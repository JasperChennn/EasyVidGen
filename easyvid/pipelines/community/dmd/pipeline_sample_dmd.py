# Copyright 2025 EasyVidGen contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""
DMD / self-forcing 因果 **前向采样** pipeline。

在 :class:`~easyvid.pipelines.wan.pipeline_t2v_casual.WanT2VCausalPipeline` 基础上对齐
外部 ``SelfForcingTrainingPipeline.inference_with_trajectory`` 的行为：

- 可选：每个 temporal 块在 ``denoising_step_list`` 内 **随机提前结束**（用于 DMD 等训练时的子步采样）。
- 可选：KV 刷新前对块输出做 ``scheduler.add_noise``（``context_noise_inject``，与外部脚本一致）。
- ``forward_sample(..., require_grad=True)``：与外部一致，**仅在每块 exit 子步**对 transformer 前向建图；KV 预热与 context 刷新为 ``no_grad``。
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import logging

from easyvid.pipelines.wan.pipeline_t2v import WanPipelineOutput
from easyvid.pipelines.wan.pipeline_t2v_casual import WanT2VCausalPipeline


logger = logging.get_logger(__name__)


class WanDMDSamplePipeline(WanT2VCausalPipeline):
    r"""
    因果块式 T2V 采样，带 DMD / self-forcing 常用开关。

    默认与 :class:`WanT2VCausalPipeline` 等价；设置 ``random_exit_per_block=True`` 时块内去噪在随机子步停止。
    """

    def __init__(
        self,
        tokenizer,
        text_encoder,
        vae,
        scheduler,
        transformer=None,
        transformer_2=None,
        boundary_ratio=None,
        expand_timesteps=False,
        num_frame_per_block: int = 21,
        independent_first_frame: bool = False,
        context_noise: float = 0.0,
        context_noise_inject: bool = False,
        same_step_across_blocks: bool = False,
        last_step_only: bool = False,
    ):
        super().__init__(
            tokenizer,
            text_encoder,
            vae,
            scheduler,
            transformer=transformer,
            transformer_2=transformer_2,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps,
            num_frame_per_block=num_frame_per_block,
            independent_first_frame=independent_first_frame,
            context_noise=context_noise,
        )
        self.register_to_config(
            context_noise_inject=context_noise_inject,
            same_step_across_blocks=same_step_across_blocks,
            last_step_only=last_step_only,
        )

    def _generate_exit_indices(
        self,
        num_blocks: int,
        num_denoising_steps: int,
        device: torch.device,
        last_step_only: bool = False,
    ) -> List[int]:
        """与 ``SelfForcingTrainingPipeline.generate_and_sync_list`` 一致：rank0 采样并 broadcast。"""
        if num_denoising_steps <= 0:
            return [0] * num_blocks
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            if last_step_only:
                idx = torch.ones(num_blocks, dtype=torch.long, device=device) * (num_denoising_steps - 1)
            else:
                idx = torch.randint(
                    low=0,
                    high=num_denoising_steps,
                    size=(num_blocks,),
                    device=device,
                )
        else:
            idx = torch.empty(num_blocks, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(idx, src=0)
        return idx.tolist()

    @staticmethod
    def _resolve_grad_step_index(
        require_grad: bool,
        exit_step_index: Optional[int],
        num_denoising_steps: int,
    ) -> Optional[int]:
        """训练时仅在某一子步建图：显式 exit 用该下标；否则跑满子步时只在最后一子步建图。"""
        if not require_grad or num_denoising_steps <= 0:
            return None
        if exit_step_index is not None:
            return int(exit_step_index)
        return num_denoising_steps - 1

    def _denoise_block(
        self,
        noisy_input: torch.Tensor,
        prompt_embeds: torch.Tensor,
        denoising_step_list: List[float],
        batch_size: int,
        current_num_frames: int,
        device: torch.device,
        transformer_dtype: torch.dtype,
        current_start_tokens: int,
        exit_step_index: Optional[int] = None,
        require_grad: bool = False,
    ) -> torch.Tensor:
        """块内子步去噪；``exit_step_index`` 非空时仅执行到该子步（含）后退出。

        与外部 self-forcing 一致：除 ``grad_step_index`` 对应子步外，transformer 前向均在 ``no_grad`` 中。
        """
        num_steps = len(denoising_step_list)
        grad_step_index = self._resolve_grad_step_index(require_grad, exit_step_index, num_steps)

        noisy = noisy_input
        denoised_pred = noisy_input
        for index, current_timestep in enumerate(denoising_step_list):
            t_val = int(current_timestep)
            timestep = torch.ones(batch_size, current_num_frames, device=device, dtype=torch.long) * t_val

            grad_here = grad_step_index is not None and index == grad_step_index
            if grad_here:
                # 与外部一致：显式在 exit 子步打开梯度（即使外层曾进入 no_grad）
                with torch.enable_grad():
                    denoised_pred = self._forward_transformer_causal(
                        noisy,
                        timestep,
                        prompt_embeds,
                        current_start_tokens,
                    )
            else:
                with torch.no_grad():
                    denoised_pred = self._forward_transformer_causal(
                        noisy,
                        timestep,
                        prompt_embeds,
                        current_start_tokens,
                    )

            if exit_step_index is not None and index == exit_step_index:
                break

            if index < len(denoising_step_list) - 1:
                with torch.no_grad():
                    next_t = denoising_step_list[index + 1]
                    next_t_val = int(next_t)
                    flat = denoised_pred.flatten(0, 1)
                    rnd = torch.randn_like(flat)
                    if hasattr(self.scheduler, "add_noise"):
                        noisy_flat = self.scheduler.add_noise(
                            flat,
                            rnd,
                            torch.full((flat.shape[0],), next_t_val, device=device, dtype=torch.long),
                        )
                    else:
                        sigma = min(max(next_t_val / 1000.0, 0.0), 1.0)
                        noisy_flat = (1.0 - sigma) * flat + sigma * rnd
                    noisy = noisy_flat.unflatten(0, denoised_pred.shape[:2]).to(transformer_dtype)

        return denoised_pred

    def _context_forward(
        self,
        denoised_pred: torch.Tensor,
        batch_size_eff: int,
        current_num_frames: int,
        context_noise: float,
        prompt_embeds: torch.Tensor,
        current_start_frame_tokens: int,
        device: torch.device,
        context_noise_inject: bool,
    ) -> None:
        """块结束后更新 KV：可选先对 latent 加噪再前向（对齐外部 self-forcing）。"""
        ctx_ts = torch.full(
            (batch_size_eff, current_num_frames),
            int(context_noise),
            device=device,
            dtype=torch.long,
        )
        with torch.no_grad():
            hidden = denoised_pred
            if context_noise_inject and int(context_noise) > 0:
                flat = hidden.flatten(0, 1)
                rnd = torch.randn_like(flat)
                if hasattr(self.scheduler, "add_noise"):
                    flat = self.scheduler.add_noise(
                        flat,
                        rnd,
                        int(context_noise)
                        * torch.ones(flat.shape[0], device=device, dtype=torch.long),
                    )
                else:
                    sigma = min(max(int(context_noise) / 1000.0, 0.0), 1.0)
                    flat = (1.0 - sigma) * flat + sigma * rnd
                hidden = flat.unflatten(0, denoised_pred.shape[:2])
            self._forward_transformer_causal(
                hidden,
                ctx_ts,
                prompt_embeds,
                current_start_frame_tokens,
            )

    def forward_sample(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        num_frame_per_block: Optional[int] = None,
        independent_first_frame: Optional[bool] = None,
        context_noise: Optional[float] = None,
        denoising_step_list: Optional[List[Union[int, float]]] = None,
        initial_latent: Optional[torch.Tensor] = None,
        warp_denoising_step: bool = False,
        random_exit_per_block: bool = False,
        same_step_across_blocks: Optional[bool] = None,
        last_step_only: Optional[bool] = None,
        context_noise_inject: Optional[bool] = None,
        exit_step_indices: Optional[List[int]] = None,
        require_grad: bool = False,
    ) -> Union[WanPipelineOutput, Tuple[Any, ...]]:
        """
        因果块式前向采样（可带梯度）。

        Args:
            random_exit_per_block: 为 True 时，每块在 ``denoising_step_list`` 的随机子步结束（需 ``exit_step_indices`` 或内部自动生成）。
            same_step_across_blocks: 所有块共用同一出口子步（覆盖 config）。
            last_step_only: 出口固定为最后一子步（覆盖 config）。
            context_noise_inject: 为 True 时，KV 刷新前对块输出 ``add_noise`` 至 ``context_noise``。
            exit_step_indices: 每块出口子步下标，长度须等于 temporal 块数；与 ``random_exit_per_block`` 二选一传入。
            require_grad: 为 True 时**仅在每块 exit 子步**（或跑满子步时的最后一子步）对 transformer 建图；KV 预热与 context 刷新不加梯度。
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        if self.transformer_2 is not None:
            raise NotImplementedError("WanDMDSamplePipeline does not support transformer_2 / two-stage denoising.")
        if self.config.expand_timesteps:
            raise NotImplementedError("WanDMDSamplePipeline does not support expand_timesteps.")
        if guidance_scale != 1.0:
            raise ValueError(
                "Causal DMD sample pipeline requires guidance_scale=1.0 (no classifier-free guidance with shared KV)."
            )
        if guidance_scale_2 is not None:
            raise ValueError("guidance_scale_2 is not used.")

        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            None,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                "`num_frames - 1` has to be divisible by %s. Rounding.",
                self.vae_scale_factor_temporal,
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        patch_size = tuple(self.transformer.config.patch_size)
        h_multiple_of = self.vae_scale_factor_spatial * patch_size[1]
        w_multiple_of = self.vae_scale_factor_spatial * patch_size[2]
        calc_height = height // h_multiple_of * h_multiple_of
        calc_width = width // w_multiple_of * w_multiple_of
        if height != calc_height or width != calc_width:
            logger.warning("Adjusting (%s, %s) -> (%s, %s).", height, width, calc_height, calc_width)
            height, width = calc_height, calc_width

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device
        num_frame_per_block = num_frame_per_block if num_frame_per_block is not None else self.config.num_frame_per_block
        independent_first_frame = (
            independent_first_frame if independent_first_frame is not None else self.config.independent_first_frame
        )
        context_noise = context_noise if context_noise is not None else self.config.context_noise
        if context_noise_inject is None:
            context_noise_inject = self.config.context_noise_inject
        same_step_across_blocks = (
            same_step_across_blocks if same_step_across_blocks is not None else self.config.same_step_across_blocks
        )
        last_step_only_eff = last_step_only if last_step_only is not None else self.config.last_step_only

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)

        num_channels_latents = self.transformer.config.in_channels
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        if latents is None:
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                torch.float32,
                device,
                generator,
                None,
            )
        else:
            latents = latents.to(device=device, dtype=torch.float32)

        noise = latents
        batch_size_eff = noise.shape[0]
        num_latent_frames_noise = noise.shape[2]
        frame_seq_length = self._frame_seq_length(height, width, num_latent_frames)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps_all = self.scheduler.timesteps
        if denoising_step_list is None:
            denoising_step_list = [float(t) for t in timesteps_all.cpu().tolist()]
        else:
            denoising_step_list = [float(x) for x in denoising_step_list]

        if warp_denoising_step and len(denoising_step_list) > 0:
            tcat = torch.cat(
                (timesteps_all.cpu().float(), torch.tensor([0.0], dtype=torch.float32)),
            )
            denoising_step_list = [float(tcat[1000 - int(i)]) for i in denoising_step_list]

        num_blocks, all_num_frames = self._causal_split_num_frames(
            num_latent_frames_noise,
            num_frame_per_block,
            independent_first_frame,
            initial_latent,
        )

        if self._kv_cache is None:
            self._allocate_caches(batch_size_eff, transformer_dtype, device, frame_seq_length, num_latent_frames_noise)
        else:
            self._reset_caches(device)

        output = torch.zeros_like(noise)

        num_denoising_steps = len(denoising_step_list)
        exit_per_block: Optional[List[int]] = None
        if random_exit_per_block:
            if exit_step_indices is not None:
                exit_per_block = list(exit_step_indices)
                if len(exit_per_block) != len(all_num_frames):
                    raise ValueError(
                        f"exit_step_indices length {len(exit_per_block)} != number of temporal blocks {len(all_num_frames)}."
                    )
            else:
                raw = self._generate_exit_indices(
                    len(all_num_frames),
                    num_denoising_steps,
                    device,
                    last_step_only=last_step_only_eff,
                )
                if same_step_across_blocks:
                    exit_per_block = [raw[0]] * len(all_num_frames)
                else:
                    exit_per_block = raw
        elif exit_step_indices is not None:
            exit_per_block = list(exit_step_indices)
            if len(exit_per_block) != len(all_num_frames):
                raise ValueError(
                    f"exit_step_indices length {len(exit_per_block)} != number of temporal blocks {len(all_num_frames)}."
                )

        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        current_start_frame = 0

        with torch.no_grad():
            if initial_latent is not None:
                init = initial_latent.to(device=device, dtype=noise.dtype)
                timestep_zero = torch.zeros(batch_size_eff, 1, device=device, dtype=torch.long)
                if independent_first_frame:
                    assert (num_input_frames - 1) % num_frame_per_block == 0, (
                        "initial_latent frames vs num_frame_per_block"
                    )
                    output[:, :, :1] = init[:, :, :1]
                    self._forward_transformer_causal(
                        init[:, :, :1],
                        timestep_zero.expand(-1, 1),
                        prompt_embeds,
                        current_start_frame * frame_seq_length,
                    )
                    current_start_frame += 1
                    num_input_blocks = (num_input_frames - 1) // num_frame_per_block
                    offset = 1
                else:
                    assert num_input_frames % num_frame_per_block == 0
                    num_input_blocks = num_input_frames // num_frame_per_block
                    offset = 0
                for _ in range(num_input_blocks):
                    chunk = init[:, :, offset : offset + num_frame_per_block]
                    output[:, :, current_start_frame : current_start_frame + num_frame_per_block] = chunk
                    ts = timestep_zero.expand(-1, num_frame_per_block)
                    self._forward_transformer_causal(
                        chunk,
                        ts,
                        prompt_embeds,
                        current_start_frame * frame_seq_length,
                    )
                    current_start_frame += num_frame_per_block
                    offset += num_frame_per_block

        for block_index, current_num_frames in enumerate(all_num_frames):
            a = current_start_frame - num_input_frames
            b = current_start_frame + current_num_frames - num_input_frames
            noisy_slice = noise[:, :, a:b, :, :].to(transformer_dtype)
            exit_idx = exit_per_block[block_index] if exit_per_block is not None else None
            denoised_pred = self._denoise_block(
                noisy_slice,
                prompt_embeds,
                denoising_step_list,
                batch_size_eff,
                current_num_frames,
                device,
                transformer_dtype,
                current_start_frame * frame_seq_length,
                exit_step_index=exit_idx,
                require_grad=require_grad,
            )

            output[:, :, current_start_frame : current_start_frame + current_num_frames] = denoised_pred

            self._context_forward(
                denoised_pred,
                batch_size_eff,
                current_num_frames,
                context_noise,
                prompt_embeds,
                current_start_frame * frame_seq_length,
                device,
                context_noise_inject,
            )

            current_start_frame += current_num_frames

        latents_out = output

        self._current_timestep = None
        if not output_type == "latent":
            latents_out = latents_out.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents_out.device, latents_out.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents_out.device, latents_out.dtype
            )
            latents_out = latents_out / latents_std + latents_mean
            # VAE 不参与训练，decode 始终 no_grad
            with torch.no_grad():
                video = self.vae.decode(latents_out, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents_out

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        num_frame_per_block: Optional[int] = None,
        independent_first_frame: Optional[bool] = None,
        context_noise: Optional[float] = None,
        denoising_step_list: Optional[List[Union[int, float]]] = None,
        initial_latent: Optional[torch.Tensor] = None,
        warp_denoising_step: bool = False,
        random_exit_per_block: bool = False,
        same_step_across_blocks: Optional[bool] = None,
        last_step_only: Optional[bool] = None,
        context_noise_inject: Optional[bool] = None,
        exit_step_indices: Optional[List[int]] = None,
    ) -> Union[WanPipelineOutput, Tuple[Any, ...]]:
        """推理入口，等价于 ``forward_sample(..., require_grad=False)``。"""
        return self.forward_sample(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            num_frame_per_block=num_frame_per_block,
            independent_first_frame=independent_first_frame,
            context_noise=context_noise,
            denoising_step_list=denoising_step_list,
            initial_latent=initial_latent,
            warp_denoising_step=warp_denoising_step,
            random_exit_per_block=random_exit_per_block,
            same_step_across_blocks=same_step_across_blocks,
            last_step_only=last_step_only,
            context_noise_inject=context_noise_inject,
            exit_step_indices=exit_step_indices,
            require_grad=False,
        )

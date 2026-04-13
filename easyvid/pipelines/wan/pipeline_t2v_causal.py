# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan T2V **causal / block-wise** 推理 pipeline（KV cache + cross-attn encoder cache），
对齐 ``tmp/pipeline_causal.py`` 的时间分块与 ``easyvid.models.wan.transformer_causal`` 的 ``forward``。

使用需加载带 ``transformer_causal.WanTransformer3DModel`` 能力（``kv_cache`` / ``cross_attn_kv_cache``）的权重。
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import logging

from easyvid.pipelines.wan.pipeline_t2v import WanPipeline, WanPipelineOutput


logger = logging.get_logger(__name__)


class WanT2VCausalPipeline(WanPipeline):
    r"""
    与 :class:`~easyvid.pipelines.wan.pipeline_t2v.WanPipeline` 相同的组件与 ``encode_prompt`` 等接口，
    但 ``__call__`` 采用 **按 latent 块** 生成：自注意力 ``kv_cache`` + cross-attn ``cross_attn_kv_cache``，
    并在每块去噪结束后用 ``context_noise`` 再跑一遍 transformer 以用干净上下文更新 KV（见 ``tmp/pipeline_causal.py``）。

    当前实现 **不支持** classifier-free guidance（``guidance_scale`` 须为 ``1.0``），且 **不支持** ``transformer_2`` / ``expand_timesteps``。
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
        num_frame_per_block: int = 3,
        independent_first_frame: bool = False,
        context_noise: float = 0.0,
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
        )
        self.register_to_config(
            num_frame_per_block=num_frame_per_block,
            independent_first_frame=independent_first_frame,
            context_noise=context_noise,
        )
        self._kv_cache: Optional[List[dict]] = None
        self._cross_attn_kv_cache: Optional[List[dict]] = None

    def _frame_seq_length(
        self,
        height: int,
        width: int,
        num_latent_frames: int,
    ) -> int:
        patch_size = tuple(self.transformer.config.patch_size)
        p_h, p_w = patch_size[1], patch_size[2]
        latent_h = int(height) // self.vae_scale_factor_spatial
        latent_w = int(width) // self.vae_scale_factor_spatial
        post_patch_h = latent_h // p_h
        post_patch_w = latent_w // p_w
        return int(post_patch_h * post_patch_w)

    def _allocate_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_seq_length: int,
        num_latent_frames: int = 31, # 默认 21 帧，与 Wan2.2 ti2v 一致
    ) -> None:
        if self.transformer is None:
            raise ValueError("transformer is required for WanT2VCausalPipeline.")
        cfg = self.transformer.config
        num_layers = cfg.num_layers
        num_heads = cfg.num_attention_heads
        head_dim = cfg.attention_head_dim
        local_attn = getattr(cfg, "local_attn_size", -1)
        local_attn = num_latent_frames if local_attn == -1 else local_attn
        kv_cache_size = local_attn * frame_seq_length

        from easyvid.models.wan.transformer_causal import WanTransformer3DModel as WanTransformer3DCasual

        self._kv_cache = WanTransformer3DCasual.init_kv_cache(
            num_layers=num_layers,
            batch_size=batch_size,
            max_sequence_length=kv_cache_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        self._cross_attn_kv_cache = WanTransformer3DCasual.init_cross_attn_kv_cache(num_layers)

    def _reset_caches(self, device: torch.device) -> None:
        if self._kv_cache is None:
            return
        for d in self._kv_cache:
            d["global_end_index"].fill_(0)
            d["local_end_index"].fill_(0)
        if self._cross_attn_kv_cache is not None:
            for d in self._cross_attn_kv_cache:
                d.clear()
                d["filled"] = False

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
        # --- causal-only ---
        num_frame_per_block: Optional[int] = None,
        independent_first_frame: Optional[bool] = None,
        context_noise: Optional[float] = None,
        denoising_step_list: Optional[List[Union[int, float]]] = None,
        initial_latent: Optional[torch.Tensor] = None,
        warp_denoising_step: bool = False,
    ):
        """
        Causal block-wise T2V。``prompt_embeds`` / ``latents`` 与父类一致。

        Args:
            num_frame_per_block: 每块 latent 帧数（默认取 ``self.config.num_frame_per_block``）。
            independent_first_frame: 首帧单独成块（无 ``initial_latent`` 时首块为 1 帧）。
            context_noise: 每块去噪结束后，用该 timestep 再前向一次以更新 KV（训练/推理脚本常用 0）。
            denoising_step_list: 块内子步 timestep 列表；若为 ``None``，则用 ``scheduler`` 的 ``num_inference_steps`` 条 ``timesteps``。
            initial_latent: 可选 ``[B, F_in, C, H_lat, W_lat]`` 条件 latent（与 ``tmp/pipeline_causal`` 一致）。
            warp_denoising_step: 若为 True，将 ``denoising_step_list`` 解释为训练步索引并映射到 ``scheduler.timesteps``。
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        if self.transformer_2 is not None:
            raise NotImplementedError("WanT2VCausalPipeline does not support transformer_2 / two-stage denoising.")
        if self.config.expand_timesteps:
            raise NotImplementedError("WanT2VCausalPipeline does not support expand_timesteps.")
        if guidance_scale != 1.0:
            raise ValueError(
                "Causal pipeline currently requires guidance_scale=1.0 (no classifier-free guidance with shared KV)."
            )
        if guidance_scale_2 is not None:
            raise ValueError("guidance_scale_2 is not used in causal pipeline.")

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
            self._allocate_caches(batch_size_eff, noise.dtype, device, frame_seq_length)
        else:
            self._reset_caches(device)

        output = torch.zeros_like(noise)

        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        current_start_frame = 0

        if initial_latent is not None:
            init = initial_latent.to(device=device, dtype=noise.dtype)
            timestep_zero = torch.zeros(batch_size_eff, 1, device=device, dtype=torch.long)
            if independent_first_frame:
                assert (num_input_frames - 1) % num_frame_per_block == 0, "initial_latent frames vs num_frame_per_block"
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

        for _, current_num_frames in enumerate(all_num_frames):
            a = current_start_frame - num_input_frames
            b = current_start_frame + current_num_frames - num_input_frames
            noisy_slice = noise[:, :, a:b, :, :]
            denoised_pred = self._denoise_block(
                noisy_slice,
                prompt_embeds,
                denoising_step_list,
                batch_size_eff,
                current_num_frames,
                device,
                transformer_dtype,
                current_start_frame * frame_seq_length,
            )

            output[:, :, current_start_frame : current_start_frame + current_num_frames] = denoised_pred

            ctx_ts = torch.full(
                (batch_size_eff, current_num_frames),
                int(context_noise),
                device=device,
                dtype=torch.long,
            )
            self._forward_transformer_causal(
                denoised_pred,
                ctx_ts,
                prompt_embeds,
                current_start_frame * frame_seq_length,
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
            video = self.vae.decode(latents_out, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents_out

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)

    def _causal_split_num_frames(
        self,
        num_latent_frames: int,
        num_frame_per_block: int,
        independent_first_frame: bool,
        initial_latent: Optional[torch.Tensor],
    ) -> Tuple[int, List[int]]:
        if not independent_first_frame or (independent_first_frame and initial_latent is not None):
            assert num_latent_frames % num_frame_per_block == 0, "num_latent_frames must align with num_frame_per_block"
            num_blocks = num_latent_frames // num_frame_per_block
            blocks = [num_frame_per_block] * num_blocks
        else:
            assert (num_latent_frames - 1) % num_frame_per_block == 0
            num_blocks = (num_latent_frames - 1) // num_frame_per_block
            blocks = [1] + [num_frame_per_block] * num_blocks
        return num_blocks, blocks

    def _forward_transformer_causal(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        current_start: int,
    ) -> torch.Tensor:
        out = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=None,
            return_dict=False,
            attention_kwargs=self._attention_kwargs,
            kv_cache=self._kv_cache,
            current_start=current_start,
            cross_attn_kv_cache=self._cross_attn_kv_cache,
        )[0]
        return out

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
    ) -> torch.Tensor:
        noisy = noisy_input
        denoised_pred = noisy_input
        for index, current_timestep in enumerate(denoising_step_list):
            t_val = int(current_timestep)
            timestep = torch.ones(batch_size, current_num_frames, device=device, dtype=torch.long) * t_val

            denoised_pred = self._forward_transformer_causal(
                noisy,
                timestep,
                prompt_embeds,
                current_start_tokens,
            )

            if index < len(denoising_step_list) - 1:
                next_t = denoising_step_list[index + 1]
                next_t_val = int(next_t)
                x0_pred = noisy - denoised_pred * (t_val / 1000.0)
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

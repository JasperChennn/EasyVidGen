"""
DMD2 风格蒸馏 Trainer 骨架：包含三个网络
  - real_score: 真实数据分布上的 score / 速度场（始终冻结）
  - fake_score: 生成分布上的 score（critic，每外层步在 G 之后更新）
  - generator_score: 待蒸馏的学生生成器（按 dfake_gen_update_ratio 决定是否在本步更新）

``args.dfake_gen_update_ratio``（整数，默认 1）：与外部 ``wan2_2_distillation.py`` 一致——每个外层
``global_step`` 若满足 ``global_step % ratio == 0`` 则先用**一批**数据训练 generator，再**总是**用
**另一批**数据训练 critic；ratio=1 时每步两次 optimizer 更新、两批数据。ratio>1 时部分外层步仅更新 critic、一批数据。

当前 ``compute_loss`` 为占位实现；正式训练时请替换为论文中的 DMD2 目标。

可通过 ``custom_model_config`` YAML 覆盖（无需改 schemas）的 critic 相关字段示例：
  - ``lr_critic``：fake_score 学习率，默认与 ``learning_rate`` 相同
  - ``adam_beta1_critic`` / ``adam_beta2_critic``：critic 的 Adam beta，默认与 ``adam_beta1`` / ``adam_beta2`` 相同
  - ``max_grad_norm_generator`` / ``max_grad_norm_critic``：分模块梯度裁剪，默认均为 ``max_grad_norm``

Generator EMA：与 ``--use_ema`` / ``--ema_decay`` 等一致；可选 ``ema_start_step``（YAML，默认 0）表示**全局步数**
达到该值后才**创建** EMA 并开始 ``step()``（省显存，对齐外部 ``ema_start_step``）。Resume 时若 checkpoint 含
``ema_generator/``，会在 load hook 里先实例化再加载。
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import math
from typing import Any, Optional

import torch
from accelerate.logging import get_logger
from accelerate.utils import DistributedType
from diffusers import AutoencoderKLWan, EMAModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, export_to_video, load_image
from tqdm.auto import tqdm

from easyvid.datasets.dataset import VideoDataset, collate_fn, RepeatLastBatchSampler
from easyvid.pipelines.wan.pipeline_i2v import WanImageToVideoPipeline
from easyvid.models.wan.transformer import WanTransformer3DModel
from easyvid.schedulers.shift_logit_norm_scheduler import ShiftedLogitNormTimestepSampler
from easyvid.trainer_utils.base_trainer import BaseTrainer
from easyvid.utils.utils import free_memory, get_memory_statistics, print_memory, unwrap_model

import torch._dynamo

torch._dynamo.config.suppress_errors = True

check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Trainer(BaseTrainer):
    """Wan I2V + DMD2：real_score 冻结；G/C 更新节奏对齐外部 dfake_gen_update_ratio（每步可 G+C、两批数据）。"""

    def __init__(self, args):
        if args.report_to == "wandb" and getattr(args, "hub_token", None) is not None:
            raise ValueError(
                "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
                " Please use `huggingface-cli login` to authenticate with the Hub."
            )
        super().__init__(args)

    def _load_vae(self):
        vae = AutoencoderKLWan.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.args.revision,
            variant=self.args.variant,
        )
        vae.to(self.accelerator.device, dtype=self.weight_dtype)
        return vae

    def _load_score_transformer(self, name: str, model_path: str = None) -> WanTransformer3DModel:
        logger.info(f"Loading score transformer: {name}")
        m = WanTransformer3DModel.from_pretrained(
            model_path or self.args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
            revision=self.args.revision,
            variant=self.args.variant,
        )
        m.to(self.accelerator.device, dtype=self.weight_dtype)
        return m

    def _load_text_image_encoder(self, weight_dtype):
        text_image_encoding_pipeline = WanImageToVideoPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path,
            transformer=None,
            vae=None,
            torch_dtype=weight_dtype,
        )
        text_image_encoding_pipeline.to(self.accelerator.device)
        return text_image_encoding_pipeline

    def _init_noise_scheduler(self):
        logger.info("Initializing noise scheduler")
        return ShiftedLogitNormTimestepSampler(
            shift=self.args.noise_shift,
            distribution_type=self.args.noise_distribution,
        )

    def _init_models(self):
        logger.info("Initializing models (real_score, fake_score, generator_score)")

        self.vae = self._load_vae()
        self.real_score = self._load_score_transformer("real_score", self.args.real_score)
        self.fake_score = self._load_score_transformer("fake_score", self.args.fake_score)
        self.generator_score = self._load_score_transformer("generator_score", self.args.generator_score)
        self.text_image_encoding_pipeline = self._load_text_image_encoder(self.weight_dtype)
        self.noise_scheduler = self._init_noise_scheduler()

    def prepare_dataset(self):
        logger.info("Initializing dataset and dataloader")

        train_dataset = VideoDataset(
            self.args.train_data_meta,
            self.args.train_data_dir,
            video_sample_size=[self.args.video_sample_height, self.args.video_sample_width],
            video_sample_n_frames=self.args.video_sample_n_frames,
        )
        sampler = RepeatLastBatchSampler(train_dataset, self.args.train_batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            drop_last=False,
        )

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader

    def _set_trainable_for_phase(self, phase: str) -> None:
        assert phase in ("fake", "generator")
        self.fake_score.requires_grad_(phase == "fake")
        self.generator_score.requires_grad_(phase == "generator")

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters (fake_score + generator_score; G/C 由 train 内分步切换)")

        if torch.backends.mps.is_available() and self.args.mixed_precision == "bf16":
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        self.vae.requires_grad_(False)
        self.real_score.requires_grad_(False)
        self.fake_score.requires_grad_(True)
        self.generator_score.requires_grad_(True)

        if self.args.gradient_checkpointing:
            self.fake_score.enable_gradient_checkpointing()
            self.generator_score.enable_gradient_checkpointing()

        self._ema_start_step = int(getattr(self.args, "ema_start_step", 0) or 0)

        if getattr(self.args, "use_ema", False):
            if self._ema_start_step > 0:
                self.ema_generator = None
                logger.info(
                    f"Generator EMA: delayed until global_step >= {self._ema_start_step} "
                    f"(decay={self.args.ema_decay}, ema_update_after_step={self.args.ema_update_after_step})"
                )
            else:
                self._create_ema_generator()
        else:
            self.ema_generator = None
            logger.info("Generator EMA disabled")

        self._prepare_ema_hooks()

    def _create_ema_generator(self) -> None:
        """实例化 ``EMAModel``（与当前 ``generator_score.parameters()`` 对齐）。"""
        logger.info(
            f"Creating generator EMA (decay={self.args.ema_decay}, "
            f"ema_update_after_step={self.args.ema_update_after_step}, every={self.args.ema_update_every})"
        )
        self.ema_generator = EMAModel(
            self.generator_score.parameters(),
            decay=self.args.ema_decay,
            update_after_step=self.args.ema_update_after_step,
            update_every=self.args.ema_update_every,
            model_cls=WanTransformer3DModel,
            model_config=self.generator_score.config,
            foreach=getattr(self.args, "foreach_ema", False),
        )
        if getattr(self.args, "offload_ema", False):
            self.ema_generator.pin_memory()
            logger.info("Generator EMA: CPU offload (pinned memory)")
        else:
            self.ema_generator.to(self.accelerator.device)
            logger.info("Generator EMA: on GPU")

    def _maybe_lazy_init_ema(self, global_step: int) -> None:
        """在达到 ``ema_start_step`` 时首次创建 EMA（训练循环内）。"""
        if not getattr(self.args, "use_ema", False):
            return
        if self.ema_generator is not None:
            return
        if global_step < self._ema_start_step:
            return
        logger.info(f"Lazy init generator EMA at global_step={global_step} (ema_start_step={self._ema_start_step})")
        self._create_ema_generator()

    def _prepare_ema_hooks(self):
        if not getattr(self.args, "use_ema", False):
            return

        def save_ema_hook(models, weights, output_dir):
            if self.ema_generator is None:
                return
            if self.accelerator.is_main_process:
                ema_dir = os.path.join(output_dir, "ema_generator")
                self.ema_generator.save_pretrained(ema_dir)
                logger.info(f"Saved generator EMA to {ema_dir}")

        def load_ema_hook(models, input_dir):
            ema_path = os.path.join(input_dir, "ema_generator")
            if not os.path.exists(ema_path):
                return
            if self.ema_generator is None:
                logger.info("Checkpoint contains EMA; instantiating generator EMA before load")
                self._create_ema_generator()
            load_m = EMAModel(
                ema_path,
                WanTransformer3DModel,
                foreach=getattr(self.args, "foreach_ema", False),
            )
            self.ema_generator.load_state_dict(load_m.state_dict())
            del load_m
            if getattr(self.args, "offload_ema", False):
                self.ema_generator.pin_memory()
            else:
                self.ema_generator.to(self.accelerator.device)
            logger.info(f"Loaded generator EMA from {ema_path}")

        self.accelerator.register_save_state_pre_hook(save_ema_hook)
        self.accelerator.register_load_state_pre_hook(load_ema_hook)

    def _maybe_ema_generator_step(self) -> None:
        """在 generator 完成一次 optimizer step 后更新 EMA（与外部 ``TRAIN_GENERATOR`` 后 ``generator_ema.update`` 对齐）。"""
        if self.ema_generator is None:
            return
        if getattr(self.args, "offload_ema", False):
            self.ema_generator.to(device=self.accelerator.device, non_blocking=True)
        self.ema_generator.step(self.generator_score.parameters())
        if getattr(self.args, "offload_ema", False):
            self.ema_generator.to(device="cpu", non_blocking=True)

    def _critic_hyperparams(self):
        """Critic 独立超参：未在 args / YAML 中设置时与 generator 共用默认值。"""
        lr_c = getattr(self.args, "lr_critic", None)
        if lr_c is None:
            lr_c = self.args.learning_rate
        else:
            lr_c = float(lr_c)
        b1c = getattr(self.args, "adam_beta1_critic", None)
        b2c = getattr(self.args, "adam_beta2_critic", None)
        if b1c is None:
            b1c = self.args.adam_beta1
        if b2c is None:
            b2c = self.args.adam_beta2
        mgn_g = getattr(self.args, "max_grad_norm_generator", None)
        if mgn_g is None:
            mgn_g = self.args.max_grad_norm
        mgn_c = getattr(self.args, "max_grad_norm_critic", None)
        if mgn_c is None:
            mgn_c = self.args.max_grad_norm
        return lr_c, b1c, b2c, float(mgn_g), float(mgn_c)

    def prepare_optimizer(self):
        logger.info("Initializing dual optimizers (generator_score + fake_score) and lr schedulers")

        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        lr_gen = float(self.args.learning_rate)
        lr_critic, beta1_c, beta2_c, max_norm_g, max_norm_c = self._critic_hyperparams()

        if self.args.scale_lr:
            scale = (
                self.args.gradient_accumulation_steps
                * self.args.train_batch_size
                * self.accelerator.num_processes
            )
            lr_gen *= scale
            lr_critic *= scale

        self.args.learning_rate = lr_gen
        self._max_grad_norm_generator = max_norm_g
        self._max_grad_norm_critic = max_norm_c

        if self.args.mixed_precision == "fp16":
            cast_training_params([self.fake_score, self.generator_score], dtype=torch.float32)

        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError as e:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                ) from e
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        n_fake = sum(p.numel() for p in self.fake_score.parameters())
        n_gen = sum(p.numel() for p in self.generator_score.parameters())
        logger.info(
            f"Dual optimizers: generator_score {n_gen / 1e6:.2f}M (lr={lr_gen}), "
            f"fake_score (critic) {n_fake / 1e6:.2f}M (lr={lr_critic}); "
            f"each optimizer step trains one module only"
        )
        logger.info(
            f"Grad clip: generator max_grad_norm={max_norm_g}, critic max_grad_norm={max_norm_c}; "
            f"critic betas=({beta1_c}, {beta2_c})"
        )

        wd = self.args.adam_weight_decay
        eps = self.args.adam_epsilon
        self.optimizer_generator = optimizer_cls(
            list(self.generator_score.parameters()),
            lr=lr_gen,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=wd,
            eps=eps,
        )
        self.optimizer_critic = optimizer_cls(
            list(self.fake_score.parameters()),
            lr=lr_critic,
            betas=(beta1_c, beta2_c),
            weight_decay=wd,
            eps=eps,
        )

        if self.args.max_train_steps is None:
            len_train_dataloader_after_sharding = math.ceil(
                len(self.train_dataloader) / self.accelerator.num_processes
            )
            num_update_steps_per_epoch = math.ceil(
                len_train_dataloader_after_sharding / self.args.gradient_accumulation_steps
            )
            num_training_steps_for_scheduler = (
                self.args.num_train_epochs * num_update_steps_per_epoch * self.accelerator.num_processes
            )
        else:
            num_training_steps_for_scheduler = self.args.max_train_steps * self.accelerator.num_processes

        ratio = max(1, int(getattr(self.args, "dfake_gen_update_ratio", 1)))
        max_train_steps_int = self._resolve_max_train_steps_for_scheduler()
        num_gen_optimizer_steps = max(1, (max_train_steps_int + ratio - 1) // ratio)
        num_training_steps_for_scheduler_generator = num_gen_optimizer_steps * self.accelerator.num_processes

        nw = self.args.lr_warmup_steps * self.accelerator.num_processes
        self.lr_scheduler_generator = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer_generator,
            num_warmup_steps=nw,
            num_training_steps=num_training_steps_for_scheduler_generator,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )
        self.lr_scheduler_critic = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer_critic,
            num_warmup_steps=nw,
            num_training_steps=num_training_steps_for_scheduler,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        self.num_training_steps_for_scheduler = num_training_steps_for_scheduler
        logger.info(
            f"dfake_gen_update_ratio={ratio}: outer steps={max_train_steps_int}, "
            f"~{num_gen_optimizer_steps} generator optimizer steps → lr scheduler total steps "
            f"(× world) = {num_training_steps_for_scheduler_generator}"
        )

    def _resolve_max_train_steps_for_scheduler(self) -> int:
        if self.args.max_train_steps is not None:
            return int(self.args.max_train_steps)
        len_train_dataloader_after_sharding = math.ceil(
            len(self.train_dataloader) / self.accelerator.num_processes
        )
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding / self.args.gradient_accumulation_steps
        )
        return int(self.args.num_train_epochs * num_update_steps_per_epoch)

    def prepare_for_training(self):
        (
            self.generator_score,
            self.fake_score,
            self.optimizer_generator,
            self.optimizer_critic,
            self.train_dataloader,
            self.lr_scheduler_generator,
            self.lr_scheduler_critic,
        ) = self.accelerator.prepare(
            self.generator_score,
            self.fake_score,
            self.optimizer_generator,
            self.optimizer_critic,
            self.train_dataloader,
            self.lr_scheduler_generator,
            self.lr_scheduler_critic,
        )
        self.transformer = self.generator_score

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            if self.num_training_steps_for_scheduler != self.args.max_train_steps * self.accelerator.num_processes:
                logger.warning(
                    "The length of the train_dataloader after accelerator.prepare may not match scheduler steps."
                )
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)
        self.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_trackers(self):
        logger.info("Initializing trackers")
        super().prepare_trackers()
        if self.accelerator.is_main_process:
            self.accelerator.print("===== Memory before training (DMD2) =====")
            free_memory(self.accelerator.device)
            print_memory(self.accelerator.device)

    def train(self):
        logger.info("Starting DMD2 training")

        memory_statistics = get_memory_statistics(logger)
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        logger.info("***** Running DMD2 training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

        if self.args.gradient_accumulation_steps != 1:
            raise ValueError(
                "dfake_gen_update_ratio 节奏（每外层步 1~2 次反传、与外部脚本一致）当前仅支持 "
                "gradient_accumulation_steps=1，请改为 1 或后续自行扩展 accumulate 逻辑。"
            )

        resume_arg = self.args.resume_from_checkpoint or None
        (
            _resume_ckpt_path,
            initial_global_step,
            global_step,
            _first_epoch,
        ) = self.get_latest_ckpt_path_to_resume_from(
            resume_arg,
            self.args.output_dir,
            self.num_update_steps_per_epoch,
        )
        if resume_arg and initial_global_step == 0:
            self.args.resume_from_checkpoint = None

        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.model_config = (
            self.generator_score.module.config if hasattr(self.generator_score, "module") else self.generator_score.config
        )

        if self.args.multi_stream:
            vae_stream = torch.cuda.Stream()
        else:
            vae_stream = None

        ratio = max(1, int(getattr(self.args, "dfake_gen_update_ratio", 1)))
        logger.info(
            f"dfake_gen_update_ratio={ratio}: when global_step % {ratio} == 0 train generator (batch A) "
            f"then always train critic (batch B); otherwise only critic with one batch."
        )

        data_iter = iter(self.train_dataloader)

        def _next_batch():
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                return next(data_iter)

        self.real_score.eval()

        while global_step < self.args.max_train_steps:
            self._maybe_lazy_init_ema(global_step)

            train_generator = global_step % ratio == 0
            loss_g = None

            if train_generator:
                self._set_trainable_for_phase("generator")
                self.generator_score.train()
                self.fake_score.eval()
                self.optimizer_generator.zero_grad(set_to_none=True)
                batch_g = _next_batch()
                with self.accelerator.accumulate(self.generator_score):
                    loss_g = self.compute_loss(batch_g, vae_stream, phase="generator")
                    self.accelerator.backward(loss_g)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.generator_score.parameters(), self._max_grad_norm_generator
                        )
                        self.optimizer_generator.step()
                        self.lr_scheduler_generator.step()
                        self._maybe_ema_generator_step()

            self._set_trainable_for_phase("fake")
            self.fake_score.train()
            self.generator_score.eval()
            self.optimizer_critic.zero_grad(set_to_none=True)
            batch_c = _next_batch()
            with self.accelerator.accumulate(self.fake_score):
                loss_c = self.compute_loss(batch_c, vae_stream, phase="fake")
                self.accelerator.backward(loss_c)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.fake_score.parameters(), self._max_grad_norm_critic)
                    self.optimizer_critic.step()
                    self.lr_scheduler_critic.step()

            progress_bar.update(1)
            global_step += 1

            self.accelerator.wait_for_everyone()
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
                if global_step % self.args.checkpointing_steps == 0:
                    save_path = self.get_intermediate_ckpt_path(
                        self.args.checkpoints_total_limit,
                        global_step,
                        self.args.output_dir,
                    )
                    self.accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            if (
                self.args.validation_epochs is not None
                and global_step % self.args.validation_epochs == 0
                and self.args.validation_prompt
            ):
                if self.ema_generator is not None:
                    if getattr(self.args, "offload_ema", False):
                        self.ema_generator.to(self.accelerator.device)
                    self.ema_generator.store(self.generator_score.parameters())
                    self.ema_generator.copy_to(self.generator_score.parameters())
                try:
                    pipe = WanImageToVideoPipeline.from_pretrained(
                        self.args.pretrained_model_name_or_path,
                        vae=self.accelerator.unwrap_model(self.vae),
                        transformer=self.accelerator.unwrap_model(self.generator_score),
                        text_encoder=self.accelerator.unwrap_model(self.text_image_encoding_pipeline.text_encoder),
                        torch_dtype=self.weight_dtype,
                        local_files_only=True,
                    )
                    pipeline_args = {
                        "prompt": self.args.validation_prompt,
                        "num_frames": self.args.video_sample_n_frames,
                        "height": self.args.video_sample_height,
                        "width": self.args.video_sample_width,
                        "num_inference_steps": 30,
                        "guidance_scale": 5.0,
                    }
                    with torch.no_grad():
                        self.log_validation(
                            pipe, self.accelerator, self.args, global_step, pipeline_args=pipeline_args
                        )
                finally:
                    if self.ema_generator is not None:
                        self.ema_generator.restore(self.generator_score.parameters())
                        if getattr(self.args, "offload_ema", False):
                            self.ema_generator.to("cpu")

            logs = {
                "loss_critic": loss_c.detach().item(),
                "lr_generator": self.lr_scheduler_generator.get_last_lr()[0],
                "lr_critic": self.lr_scheduler_critic.get_last_lr()[0],
                "train_generator": float(train_generator),
            }
            if loss_g is not None:
                logs["loss_generator"] = loss_g.detach().item()
            progress_bar.set_postfix(**logs)
            self.accelerator.log(logs, step=global_step)

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self.generator_score = unwrap_model(self.accelerator, self.generator_score)
            save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
            self.accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

        memory_statistics = get_memory_statistics(logger)
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")
        free_memory(self.accelerator.device)

        self.accelerator.end_training()

    def compute_loss(self, batch, vae_stream: Optional[Any], phase: str):
        """
        TODO: 替换为 DMD2 论文中的分布匹配 / critic 梯度项。

        ``phase`` 为 ``\"fake\"`` 时只反传 ``fake_score``；为 ``\"generator\"`` 时只反传 ``generator_score``。
        当前占位损失仅用于跑通交替训练逻辑。
        """
        pixel_values = batch["pixel_values"].to(self.weight_dtype)
        pixel_latents = self.encode_video(
            pixel_values, vae_stream, self.vae, self.args.vae_mini_batch, self.weight_dtype
        )
        if vae_stream is not None:
            torch.cuda.current_stream().wait_stream(vae_stream)

        with torch.no_grad():
            prompts = batch["captions"]
            prompt_embeds, _ = self.text_image_encoding_pipeline.encode_prompt(
                prompts,
                do_classifier_free_guidance=False,
                device=self.accelerator.device,
                dtype=self.weight_dtype,
            )

        sigmas, _ = self.noise_scheduler.sample(pixel_latents.shape[0], pixel_latents.device)
        sigmas = sigmas.unsqueeze(1).repeat(1, pixel_latents.shape[2])
        sigmas[:, :1] *= 0.01

        timesteps = torch.round(sigmas * 1000.0).long()
        num_token_per_frame = (pixel_latents.shape[-2] // 2) * (pixel_latents.shape[-1] // 2)
        timesteps = timesteps.unsqueeze(-1).repeat(1, 1, num_token_per_frame).flatten(1, 2)

        sigmas = sigmas.unsqueeze(1)
        while len(sigmas.shape) < pixel_latents.ndim:
            sigmas = sigmas.unsqueeze(-1)
        pixel_latents = pixel_latents.to(sigmas.dtype)

        noise = torch.randn_like(pixel_latents, device=self.accelerator.device, dtype=pixel_latents.dtype)
        noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
        latent_model_input = noisy_model_input.to(self.weight_dtype)

        with torch.no_grad():
            real_pred = self.real_score(
                hidden_states=latent_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

        if phase == "fake":
            fake_pred = self.fake_score(
                hidden_states=latent_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            loss = (fake_pred - real_pred).pow(2).mean()
            return loss

        with torch.no_grad():
            fake_pred = self.fake_score(
                hidden_states=latent_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

        gen_pred = self.generator_score(
            hidden_states=latent_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        pseudo_target = fake_pred - real_pred
        loss = (gen_pred - pseudo_target).pow(2).mean()
        return loss

    @staticmethod
    def encode_video(pixel_values, vae_stream, vae, vae_mini_batch, weight_dtype):
        with torch.no_grad():

            def _slice_vae(pixel_values):
                bs = vae_mini_batch
                new_pixel_values = []
                for i in range(0, pixel_values.shape[0], bs):
                    pixel_values_bs = pixel_values[i : i + bs]
                    pixel_values_bs = vae.encode(pixel_values_bs).latent_dist
                    pixel_values_bs = pixel_values_bs.sample()
                    new_pixel_values.append(pixel_values_bs)
                return torch.cat(new_pixel_values, dim=0)

            if vae_stream is not None:
                vae_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(vae_stream):
                    latents = _slice_vae(pixel_values)
            else:
                latents = _slice_vae(pixel_values)

            latents_mean = (
                torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = (latents - latents_mean) * latents_std
        return latents.to(weight_dtype)

    @torch.no_grad()
    def log_validation(
        self,
        pipeline,
        accelerator,
        args,
        step,
        pipeline_args,
        torch_dtype=torch.bfloat16,
        is_final_validation=False,
    ):
        import numpy as np
        from PIL import Image

        logger.info(
            f"Running validation... \n Generating images with prompt: {args.validation_prompt}."
        )
        for name, module in pipeline.components.items():
            if hasattr(module, "module"):
                pipeline.components[name] = accelerator.unwrap_model(module)
        pipeline = pipeline.to(accelerator.device)
        prompts = args.validation_prompt.split(args.validation_prompt_separator)
        images = args.validation_images.split(":::")
        negative_prompt = (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
            "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
            "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
            "messy background, three legs, many people in the background, walking backwards"
        )
        for prompt, image_path in zip(prompts, images):
            generator = (
                torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
            )
            pipeline_args["prompt"] = prompt
            pipeline_args["negative_prompt"] = negative_prompt
            image = load_image(image_path)
            video = pipeline(image=image, **pipeline_args, generator=generator).frames[0]
            if pipeline_args["num_frames"] > 1:
                export_to_video(
                    video,
                    os.path.join(args.validation_dir, f"{step}-{prompt.replace(' ', '_')[:20]}.mp4"),
                )
            else:
                image = Image.fromarray((video[0] * 255).astype(np.uint8))
                image.save(os.path.join(args.validation_dir, f"{step}-{prompt.replace(' ', '_')[:30]}.png"))
        del pipeline
        torch.cuda.empty_cache()
        free_memory(accelerator.device)


if __name__ == "__main__":
    from easyvid.utils.schemas import parse_args

    args = parse_args()
    trainer = Trainer(args)
    trainer.fit()

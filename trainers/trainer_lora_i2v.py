import os
import sys
import math
import json
import shutil
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from typing import List, Optional, Tuple, Union, Any

import torch

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed

import transformers
# from transformers import AutoTokenizer, UMT5EncoderModel, CLIPVisionModel

import diffusers
from diffusers import (
    AutoencoderKLWan, 
    FlowMatchEulerDiscreteScheduler, 
    # WanImageToVideoPipeline, 
    # WanTransformer3DModel
)
from diffusers.utils import check_min_version, load_image, load_video, export_to_video
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3
)

from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.nn.attention.flex_attention import flex_attention
from safetensors.torch import save_file, load_file

from src.trainers.utils import unwrap_model, get_memory_statistics, free_memory, print_memory
from src.datasets.dataset import VideoDataset, collate_fn
from src.schedulers.noise_scheduler import ShiftedLogitNormalTimestepSampler
from src.pipelines.pipeline_i2v import WanImageToVideoPipeline
from src.models.transformer import WanTransformer3DModel
from src.models.lora import WanAttnProcessorLora

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Trainer:
    def __init__(self, args):
        self.args = args
        if self.args.report_to == "wandb" and self.args.hub_token is not None:
            raise ValueError(
                "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
                " Please use `huggingface-cli login` to authenticate with the Hub."
            )

        self._init_distributed()
        self._init_logging()
        self._init_directories()
        self._init_weight_dtype()
        self._init_models()
        self._init_noise_scheduler()

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        self.args.validation_dir = os.path.join(self.args.output_dir, "validate")
        os.makedirs(self.args.validation_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)

        accelerator_project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=str(logging_dir))

        accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=accelerator_project_config
        )

        # Disable AMP for MPS. A technique for accelerating machine learning computations on iOS and macOS devices.
        if torch.backends.mps.is_available():
            logger.info("MPS is enabled. Disabling AMP.")
            accelerator.native_amp = False

        self.accelerator = accelerator

        # If passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self):
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            # DEBUG, INFO, WARNING, ERROR, CRITICAL
            level=logging.INFO,
        )

        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

    def _init_directories(self):
        # Handle the repository creation
        if self.accelerator.is_main_process:
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)

    def _init_weight_dtype(self):
        # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if self.accelerator.state.deepspeed_plugin:
            # DeepSpeed is handling precision, use what's in the DeepSpeed config
            if (
                "fp16" in self.accelerator.state.deepspeed_plugin.deepspeed_config
                and self.accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
            ):
                weight_dtype = torch.float16
            if (
                "bf16" in self.accelerator.state.deepspeed_plugin.deepspeed_config
                and self.accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
            ):
                weight_dtype = torch.bfloat16
        else:
            if self.accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif self.accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
    
    # 3D VAE
    def _load_vae(self):
        vae = AutoencoderKLWan.from_pretrained(
            self.args.pretrained_model_name_or_path, 
            subfolder="vae", 
            revision=self.args.revision, 
            variant=self.args.variant
        )
        return vae

    # Transformer
    def _load_transformer(self):
        load_dtype = torch.bfloat16 if "5b" in self.args.pretrained_model_name_or_path.lower() else torch.float16
        transformer = WanTransformer3DModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=load_dtype,
            revision=self.args.revision,
            variant=self.args.variant,
        )
        return transformer

    # Text Encoder
    def _load_text_image_encoder(self, weight_dtype):
        # Create a pipeline for text encoding. We will move this pipeline to GPU/CPU as needed.
        text_image_encoding_pipeline = WanImageToVideoPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path, 
            transformer=None, vae=None, torch_dtype=weight_dtype
        )
        return text_image_encoding_pipeline

    def _init_models(self):
        logger.info("Initializing models")

        self.vae = self._load_vae()
        self.transformer = self._load_transformer()
        self.text_image_encoding_pipeline = self._load_text_image_encoder(self.weight_dtype)

        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.transformer.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_image_encoding_pipeline = self.text_image_encoding_pipeline.to(self.accelerator.device)

    def _init_noise_scheduler(self):
        logger.info("Initializing noise scheduler")
        noise_scheduler = ShiftedLogitNormalTimestepSampler(shift=self.args.noise_shift, distribution_type=self.args.noise_distribution)

        self.noise_scheduler = noise_scheduler

    def prepare_dataset(self):
        logger.info("Initializing dataset and dataloader")

        # Prepare dataset and dataloader.
        train_dataset = VideoDataset(
            self.args.train_data_meta, self.args.train_data_dir,
            video_sample_size=[self.args.video_sample_height, self.args.video_sample_width],
            video_sample_n_frames=self.args.video_sample_n_frames
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            prefetch_factor=1,
            # pin_memory=True,
            # persistent_workers=True,
        )

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        
        if torch.backends.mps.is_available() and self.args.mixed_precision == "bf16":
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.__prepare_saving_loading_hooks()

        if self.args.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        # inject lora
        attn_processors = {}
        attn_processor_type = WanAttnProcessorLora
        for key, value in self.transformer.attn_processors.items():
            lora_modules = self.args.lora_modules if self.args.lora_modules else ['q', 'k', 'v', 'out']
            attn_processor = attn_processor_type(
                in_channels = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim,
                out_channels = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim,
                rank = self.args.rank,
                network_alpha = self.args.lora_alpha,
                device = self.accelerator.device,
                dtype = self.weight_dtype,
                lora_modules = lora_modules,
            )
            for name, param in attn_processor.named_parameters():
                param.requires_grad = True
            attn_processors[key] = attn_processor
        self.transformer.set_attn_processor(attn_processors)
        logger.info("Attn Processor initialized")
            

    def prepare_optimizer(self):
        logger.info("Initializing optimizer and lr scheduler")

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate * \
                self.args.gradient_accumulation_steps * \
                self.args.train_batch_size * \
                self.accelerator.num_processes
            )

        # Make sure the trainable params are in float32.
        if self.args.mixed_precision == "fp16":
            models = [self.transformer]
            # only upcast trainable parameters into fp32
            cast_training_params(models, dtype=torch.float32)

        # Initialize the optimizer
        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        transformer_lora_parameters = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))
        total_params = sum(p.numel() for p in transformer_lora_parameters)
        logger.info(f"Total trainable parameters: {total_params/1e6:.2f}M")

        params_to_optimize = [
            {
                'params': transformer_lora_parameters, 
                "lr": self.args.learning_rate
            }
        ]

        self.trainable_names = []
        for name, param in self.transformer.named_parameters():
            if param.requires_grad:
                self.trainable_names.append(name)
                
        optimizer = optimizer_cls(
            params_to_optimize,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        # Scheduler and math around the number of training steps.
        # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
        if self.args.max_train_steps is None:
            len_train_dataloader_after_sharding = math.ceil(len(self.train_dataloader) / self.accelerator.num_processes)
            num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / self.args.gradient_accumulation_steps)
            num_training_steps_for_scheduler = (
                self.args.num_train_epochs * num_update_steps_per_epoch * self.accelerator.num_processes
            )
        else:
            num_training_steps_for_scheduler = self.args.max_train_steps * self.accelerator.num_processes

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=num_training_steps_for_scheduler,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_training_steps_for_scheduler = num_training_steps_for_scheduler

    def prepare_for_training(self):
        # Prepare everything with our `accelerator`.
        self.transformer, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.transformer, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            if self.num_training_steps_for_scheduler != self.args.max_train_steps * self.accelerator.num_processes:
                logger.warning(
                    f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(self.train_dataloader)}) does not match "
                    f"This inconsistency may result in the learning rate scheduler not functioning properly."
                )
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)
        self.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_trackers(self):
        logger.info("Initializing trackers")

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            tracker_config = dict(vars(self.args))

            # ✅ 过滤掉不支持的类型
            filtered_config = {}
            for key, value in tracker_config.items():
                if isinstance(value, (list, tuple)):
                    filtered_config[key] = str(value)
                else:
                    filtered_config[key] = value

            self.accelerator.init_trackers(self.args.tracker_project_name, config=filtered_config)

            self.accelerator.print("===== Memory before training =====")
            free_memory(self.accelerator.device)
            print_memory(self.accelerator.device)

    def train(self):
        logger.info("Starting training")

        memory_statistics = get_memory_statistics(logger)
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        # Train!
        total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if not self.args.resume_from_checkpoint:
            initial_global_step = 0
        else:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.args.output_dir, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // self.num_update_steps_per_epoch

        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4

        # For DeepSpeed training
        self.model_config = self.transformer.module.config if hasattr(self.transformer, "module") else self.transformer.config

        if self.args.multi_stream:
            # create extra cuda streams to speedup inpaint vae computation
            vae_stream = torch.cuda.Stream()
        else:
            vae_stream = None

        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.transformer.train()
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.transformer):
                    loss = self.compute_loss(batch, vae_stream, step)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        params_to_clip = self.transformer.parameters()
                        self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    self.accelerator.wait_for_everyone()
                    # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                    if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
                        if global_step % self.args.checkpointing_steps == 0:
                            # self.accelerator.wait_for_everyone()
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if self.args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(self.args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= self.args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - self.args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(self.args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                        if global_step % self.args.validation_epochs == 0 and self.accelerator.is_main_process:
                            # self.accelerator.wait_for_everyone()
                            pipe = WanImageToVideoPipeline.from_pretrained(
                                self.args.pretrained_model_name_or_path,
                                vae=self.accelerator.unwrap_model(self.vae),
                                transformer=self.accelerator.unwrap_model(self.transformer),
                                text_encoder=self.accelerator.unwrap_model(self.text_image_encoding_pipeline.text_encoder),
                                torch_dtype=self.weight_dtype,
                                local_files_only=True,
                            )
                            pipeline_args = {"prompt": self.args.validation_prompt, "num_frames": self.args.video_sample_n_frames, "height":self.args.video_sample_height, "width":self.args.video_sample_width, "guidance_scale":5.0}
                            with torch.no_grad():
                                self.log_validation(pipe, self.accelerator, self.args, global_step, pipeline_args=pipeline_args)

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.args.max_train_steps:
                    break
            
            memory_statistics = get_memory_statistics(logger)
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")
        
        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self.transformer = unwrap_model(self.accelerator, self.transformer)
            
            save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
            self.accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

        memory_statistics = get_memory_statistics(logger)
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")
        free_memory(self.accelerator.device)

        self.accelerator.end_training()

    def fit(self):
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        self.prepare_trackers()
        self.train()

    def compute_loss(self, batch, vae_stream, step=0):
        # Convert videos and images to latent space
        pixel_values = batch["pixel_values"].to(self.weight_dtype)
        pixel_latents = self.encode_video(pixel_values, vae_stream, self.vae.to(self.accelerator.device), self.args.vae_mini_batch, self.weight_dtype)

        # wait for latents = vae.encode(pixel_values) to complete
        if vae_stream is not None:
            torch.cuda.current_stream().wait_stream(vae_stream)

        if self.args.low_vram:
            self.vae.to('cpu')
            torch.cuda.empty_cache()

        with torch.no_grad():
            prompts = batch["captions"]
            bsz = pixel_values.size(0)
            prompt_embeds, _ = self.text_image_encoding_pipeline.encode_prompt(
                prompts,
                do_classifier_free_guidance=False,
                device=self.accelerator.device,
                dtype=self.weight_dtype
            )

        if self.args.low_vram:
            self.text_image_encoding_pipeline = self.text_image_encoding_pipeline.to("cpu")
            torch.cuda.empty_cache()

        #sigmas = self.noise_scheduler.sample_for(pixel_latents).to(self.weight_dtype)
        dummy_for_sampling = torch.zeros(pixel_latents.shape[0], pixel_latents.shape[2], 1, device=pixel_latents.device)
        sigmas = self.noise_scheduler.sample_for(dummy_for_sampling)
        sigmas = sigmas.unsqueeze(1).repeat(1, pixel_latents.shape[2]) # [b, f]
        sigmas[:, :1] *= 0.01
        
        timesteps = torch.round(sigmas * 1000.0).long() # [b, f]
        num_token_per_frame = (pixel_latents.shape[-2]//2) * (pixel_latents.shape[-1]//2) # h * 2
        timesteps = timesteps.unsqueeze(-1).repeat(1,1,num_token_per_frame).flatten(1,2) # [b, f * h * w]
        
        sigmas = sigmas.unsqueeze(1) # [b, 1, f]
        while len(sigmas.shape) < pixel_latents.ndim:
            sigmas = sigmas.unsqueeze(-1) # [b, 1, f, 1, 1]
        pixel_latents = pixel_latents.to(sigmas.dtype)

        noise = torch.randn_like(
            pixel_latents, 
            device=self.accelerator.device, 
            dtype=pixel_latents.dtype
        )

        noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
        # Concatenate across channels.

        latent_model_input = noisy_model_input.to(self.weight_dtype)

        model_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        loss_masks = torch.ones_like(latent_model_input)
        loss_masks[:, :, :1] *= 0

        # flow-matching loss
        target = noise - pixel_latents
        loss = (model_pred - target).pow(2) * loss_masks
        loss = loss.mean()
        return loss

    @staticmethod
    def encode_video(pixel_values, vae_stream, vae, vae_mini_batch, weight_dtype):
        with torch.no_grad():
            # This way is quicker when batch grows up
            def _slice_vae(pixel_values):
                bs = vae_mini_batch
                new_pixel_values = []
                for i in range(0, pixel_values.shape[0], bs):
                    pixel_values_bs = pixel_values[i : i + bs]
                    pixel_values_bs = vae.encode(pixel_values_bs).latent_dist
                    pixel_values_bs = pixel_values_bs.sample()
                    new_pixel_values.append(pixel_values_bs)
                return torch.cat(new_pixel_values, dim = 0)
            if vae_stream is not None:
                vae_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(vae_stream):
                    latents = _slice_vae(pixel_values)
            else:
                latents = _slice_vae(pixel_values)

            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )

            latents = (latents - latents_mean) * latents_std
        return latents.to(weight_dtype)

    @staticmethod
    def decode_video(latents, vae_stream, vae, vae_mini_batch, weight_dtype):
        with torch.no_grad():
            # decode latents to video
            def _slice_vae_decode(latents):
                bs = vae_mini_batch
                decoded_frames = []
                for i in range(0, latents.shape[0], bs):
                    latents_bs = latents[i : i + bs]
                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, vae.config.z_dim, 1, 1, 1)
                        .to(latents_bs.device, latents_bs.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
                        1, vae.config.z_dim, 1, 1, 1
                    ).to(latents_bs.device, latents_bs.dtype)
                    
                    
                    latents_bs = latents_bs / latents_std + latents_mean
                    
                    decoded = vae.decode(latents_bs).sample
                    decoded_frames.append(decoded)
                
                return torch.cat(decoded_frames, dim=0)

            if vae_stream is not None:
                vae_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(vae_stream):
                    decoded = _slice_vae_decode(latents)
            else:
                decoded = _slice_vae_decode(latents)

            return decoded.to(weight_dtype)


    @staticmethod
    def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device=''):
        sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def __prepare_saving_loading_hooks(self):
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:

                for model in models:
                    if isinstance(
                        unwrap_model(self.accelerator, model), 
                        type(unwrap_model(self.accelerator, self.transformer))
                    ):
                        unwrapped_model_state = unwrap_model(self.accelerator, model).state_dict()
                        lora_state_dict = {k: unwrapped_model_state[k] for k in unwrapped_model_state.keys() if k in self.trainable_names}
                        save_file(
                                lora_state_dict,
                                os.path.join(output_dir, "transformer_lora.safetensors")
                        )
                        logger.info(f"Saved lora to {os.path.join(output_dir, 'transformer_lora.safetensors')}")
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            """load LoRA weights"""
            transformer_ = None

            # ✅ if not DeepSpeed 
            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()

                    if isinstance(
                        unwrap_model(self.accelerator, model), 
                        type(unwrap_model(self.accelerator, self.transformer))
                    ):
                        transformer_ = unwrap_model(self.accelerator, model)
                    else:
                        raise ValueError(f"unexpected load model: {unwrap_model(self.accelerator, model).__class__}")
            else:
                # ✅ DeepSpeed : currnet transformer
                transformer_ = unwrap_model(self.accelerator, self.transformer)

            # ✅ load LoRA weight
            lora_path = os.path.join(input_dir, "transformer_lora.safetensors")
            
            if not os.path.exists(lora_path):
                logger.warning(f"LoRA weights not found at {lora_path}, skipping loading")
                return

            lora_state_dict = load_file(lora_path)
            
            # ✅ 
            missing_keys, unexpected_keys = transformer_.load_state_dict(lora_state_dict, strict=False)
            
            # if missing_keys:
            #     logger.warning(f"Missing keys when loading LoRA: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading LoRA: {unexpected_keys}")
            
            logger.info(f"Loaded LoRA weights from {lora_path}")

            # ✅ cast trainable params to fp32
            if self.args.mixed_precision == "fp16":
                cast_training_params([transformer_], dtype=torch.float32)

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def tensor_to_pil(self, src_img_tensor):
        """
        Converts a tensor image to a PIL image.

        This function takes an input tensor with the shape (C, H, W) and converts it
        into a PIL Image format. It ensures that the tensor is in the correct data
        type and moves it to CPU if necessary.

        Parameters:
            src_img_tensor (torch.Tensor): Input image tensor with shape (C, H, W),
                where C is the number of channels, H is the height, and W is the width.

        Returns:
            PIL.Image: The converted image in PIL format.
        """

        img = src_img_tensor.clone().detach()
        if img.dtype == torch.bfloat16:
            img = img.to(torch.float32)
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
        pil_image = Image.fromarray(img)
        return pil_image

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
        logger.info(
            f"Running validation... \n Generating  images with prompt:"
            f" {args.validation_prompt}."
        )
        # 解包所有模型组件
        for name, module in pipeline.components.items():
            if hasattr(module, 'module'):
                pipeline.components[name] = accelerator.unwrap_model(module)
        pipeline = pipeline.to(accelerator.device)
        # pipeline.set_progress_bar_config(disable=True)
        prompts = args.validation_prompt.split(self.args.validation_prompt_separator)
        images = args.validation_images.split(":::")
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        for prompt, image_path in zip(prompts, images):
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
            pipeline_args["prompt"] = prompt
            pipeline_args["negative_prompt"] = negative_prompt
            image = load_image(image_path)
            video=pipeline(
                image=image,
                **pipeline_args,
                generator=generator,
            ).frames[0]
            if pipeline_args['num_frames'] > 1:
                export_to_video(video, os.path.join(args.validation_dir, f"{step}-{prompt.replace(' ', '_')[:20]}.mp4"))
            else:
                image = Image.fromarray((video[0] * 255).astype(np.uint8))
                image.save(os.path.join(args.validation_dir, f"{step}-{prompt.replace(' ', '_')[:30]}.png"))
        del pipeline
        torch.cuda.empty_cache()
        free_memory(accelerator.device)

if __name__ == '__main__':
    from src.trainers.schemas import parse_args
    args = parse_args()
    trainer = Trainer(args)
    trainer.fit()
import os
import sys
import math
import json
import shutil
import logging
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from einops import rearrange

import torch

from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed

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
)

from safetensors.torch import save_file, load_file

from src.utils.utils import unwrap_model, get_memory_statistics, free_memory
from src.datasets.dataset import VideoDataset, collate_fn
from src.schedulers.noise_scheduler import ShiftedLogitNormalTimestepSampler
from src.pipelines.wan.pipeline_i2v import WanImageToVideoPipeline
from src.models.wan.transformer import WanTransformer3DModel
from src.models.wan.lora import WanAttnProcessorLora
from src.models.wan.vae_utils import encode_to_latents, decode_to_videos
from trainers.base_trainer import BaseTrainer

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

from src.constants import LOG_LEVEL, LOG_NAME


logger = get_logger(LOG_NAME, LOG_LEVEL)


class WanTrainer(BaseTrainer):
    def __init__(self, args):
        super.__init__(args)
        self.encode_to_latents = staticmethod(encode_to_latents)
        self.decode_to_videos = staticmethod(decode_to_videos)
    
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
        """
        if add additional modules, prepare first!
        """
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
        resume_from_checkpoint_path, initial_global_step, global_step, first_epoch = self.get_latest_ckpt_path_to_resume_from(
            self.args.resume_from_checkpoint, self.args.output_dir, self.num_update_steps_per_epoch
        )

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

        if global_step % self.args.validation_epochs == 0 and self.accelerator.is_main_process and self.args.do_validation == 'true':
            self.log_validation(global_step)

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
                            save_path = self.get_intermediate_ckpt_path(self.args.checkpoints_total_limit, step=global_step, output_dir=self.args.output_dir)
                            self.accelerator.save_state(save_path)

                        if global_step % self.args.validation_epochs == 0 and self.accelerator.is_main_process and self.args.do_validation == 'true':
                            # self.accelerator.wait_for_everyone()
                            self.log_validation(global_step)
                            
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
        pixel_latents = self.encode_to_latents(pixel_values, vae_stream, self.vae.to(self.accelerator.device), self.args.vae_mini_batch, self.weight_dtype)

        # wait for latents = vae.encode(pixel_values) to complete
        if vae_stream is not None:
            torch.cuda.current_stream().wait_stream(vae_stream)

        if self.args.low_vram:
            self.vae.to('cpu')
            torch.cuda.empty_cache()

        with torch.no_grad():
            prompts = batch["captions"]
            prompt_embeds, _ = self.text_image_encoding_pipeline.encode_prompt(
                prompts,
                do_classifier_free_guidance=False,
                device=self.accelerator.device,
                dtype=self.weight_dtype
            )

        if self.args.low_vram:
            self.text_image_encoding_pipeline = self.text_image_encoding_pipeline.to("cpu")
            torch.cuda.empty_cache()

        # sample timesteps from noise_scheduler
        sigmas = self.noise_scheduler.sample(len(prompts), self.accelerator.device)
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
        noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise # torch.float32

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

    @torch.no_grad()
    def log_validation(
        self,
        step,
        pipeline_args,
        is_final_validation=False,
    ):
        logger.info(
            f"Running validation... \n Generating  images with prompt:"
            f" {self.args.validation_prompt}."
        )
        self.transformer.eval()
        pipeline = WanImageToVideoPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path,
            vae=self.accelerator.unwrap_model(self.vae),
            transformer=self.accelerator.unwrap_model(self.transformer),
            text_encoder=self.accelerator.unwrap_model(self.text_image_encoding_pipeline.text_encoder),
            torch_dtype=self.weight_dtype,
            local_files_only=True,
        )
        # unwrap all modules
        for name, module in pipeline.components.items():
            if hasattr(module, 'module'):
                pipeline.components[name] = self.accelerator.unwrap_model(module)

        pipeline = pipeline.to(self.accelerator.device)
        # pipeline.set_progress_bar_config(disable=True)
        prompts = self.args.validation_prompt.split(self.args.validation_prompt_separator)
        images = self.args.validation_images.split(":::")
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        for prompt, image_path in zip(prompts, images):
            generator = torch.Generator(device=self.accelerator.device).manual_seed(args.seed) if args.seed is not None else None
            image = load_image(image_path)
            video=pipeline(
                image=image,
                prompt = prompt,
                negative_prompt = negative_prompt,
                num_frames = self.args.video_sample_n_frames, 
                height = self.args.video_sample_height,
                width = self.args.video_sample_width,
                generator=generator,
                guidance_scale=5.0,
            ).frames[0]
            if pipeline_args['num_frames'] > 1:
                export_to_video(video, os.path.join(args.validation_dir, f"{step}-{prompt.replace(' ', '_')[:20]}.mp4"))
            else:
                image = Image.fromarray((video[0] * 255).astype(np.uint8))
                image.save(os.path.join(args.validation_dir, f"{step}-{prompt.replace(' ', '_')[:30]}.png"))
        
        self.transformer.train()
        del pipeline
        torch.cuda.empty_cache()
        free_memory(self.accelerator.device)

if __name__ == '__main__':
    from src.utils.schemas import parse_args
    args = parse_args()
    trainer = WanTrainer(args)
    trainer.fit()
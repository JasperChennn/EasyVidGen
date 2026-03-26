import json
import logging
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
import transformers
import diffusers
from typing import List, Optional, Tuple, Union, Any
from easyvid.utils.file_utils import delete_files, find_files

logger = get_logger(__name__, log_level="INFO")


class BaseTrainer:
    def __init__(self, args):
        self.args = args

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

        accelerator_project_config = ProjectConfiguration(
            project_dir=self.args.output_dir,
            logging_dir=str(logging_dir),
        )

        accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=accelerator_project_config,
        )

        if torch.backends.mps.is_available():
            logger.info("MPS is enabled. Disabling AMP.")
            accelerator.native_amp = False

        self.accelerator = accelerator

        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

    def _init_directories(self):
        if self.accelerator.is_main_process and self.args.output_dir is not None:
            os.makedirs(self.args.output_dir, exist_ok=True)

    def _init_weight_dtype(self):
        weight_dtype = torch.float32
        if self.accelerator.state.deepspeed_plugin:
            cfg = self.accelerator.state.deepspeed_plugin.deepspeed_config
            if "fp16" in cfg and cfg["fp16"]["enabled"]:
                weight_dtype = torch.float16
            if "bf16" in cfg and cfg["bf16"]["enabled"]:
                weight_dtype = torch.bfloat16
        else:
            if self.accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif self.accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

    def _init_models(self):
        raise NotImplementedError

    def _init_noise_scheduler(self):
        raise NotImplementedError

    def prepare_dataset(self):
        raise NotImplementedError

    def prepare_trainable_parameters(self):
        raise NotImplementedError

    def prepare_optimizer(self):
        raise NotImplementedError

    def compute_loss(self, batch, vae_stream):
        raise NotImplementedError

    def __prepare_saving_loading_hooks(self):
        pass

    def prepare_for_training(self):
        self.transformer, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.transformer, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)
        self.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_trackers(self):
        if self.accelerator.is_main_process:
            tracker_config = dict(vars(self.args))
            filtered_config = {k: (str(v) if isinstance(v, (list, tuple)) else v)
                               for k, v in tracker_config.items()}
            self.accelerator.init_trackers(self.args.tracker_project_name, config=filtered_config)

    def get_latest_ckpt_path_to_resume_from(
        self, resume_from_checkpoint: str | None, output_dir: str, num_update_steps_per_epoch: int
    ) -> Tuple[str | None, int, int, int]:
        if resume_from_checkpoint is None:
            initial_global_step = 0
            global_step = 0
            first_epoch = 0
            resume_from_checkpoint_path = None
        else:
            if resume_from_checkpoint != "latest":
                path = os.path.basename(resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                logger.info(
                    f"No checkpoint found for resume_from_checkpoint={resume_from_checkpoint!r}. "
                    "Starting a new training run."
                )
                initial_global_step = 0
                global_step = 0
                first_epoch = 0
                resume_from_checkpoint_path = None
            else:
                resume_from_checkpoint_path = os.path.join(output_dir, path)

                if not os.path.exists(resume_from_checkpoint_path):
                    logger.info(
                        f"Checkpoint path does not exist: {resume_from_checkpoint_path}. "
                        "Starting a new training run."
                    )
                    initial_global_step = 0
                    global_step = 0
                    first_epoch = 0
                    resume_from_checkpoint_path = None
                else:
                    logger.info(f"Resuming from checkpoint {resume_from_checkpoint_path}")
                    self.accelerator.load_state(resume_from_checkpoint_path)
                    global_step = int(os.path.basename(resume_from_checkpoint_path).split("-")[1])

                    initial_global_step = global_step
                    first_epoch = global_step // num_update_steps_per_epoch

        return resume_from_checkpoint_path, initial_global_step, global_step, first_epoch

    def get_intermediate_ckpt_path(
        self, checkpointing_limit: Optional[int], step: int, output_dir: str
    ) -> str:
        if checkpointing_limit is not None:
            checkpoints = find_files(output_dir, prefix="checkpoint")
            if len(checkpoints) >= checkpointing_limit:
                num_to_remove = len(checkpoints) - checkpointing_limit + 1
                checkpoints_to_remove = checkpoints[0:num_to_remove]
                logger.info(
                    f"{len(checkpoints)} checkpoints exist, removing {len(checkpoints_to_remove)} oldest: "
                    f"{[p.name for p in checkpoints_to_remove]}"
                )
                delete_files(checkpoints_to_remove)

        save_path = os.path.join(output_dir, f"checkpoint-{step}")
        logger.info(f"Checkpointing at step {step}, saving state to {save_path}")
        return save_path

    def train(self):
        # rely on：self.train_dataloader / self.compute_loss / self.transformer / self.optimizer / self.lr_scheduler / etc.
        raise NotImplementedError

    def fit(self):
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        self.prepare_trackers()
        self.train()

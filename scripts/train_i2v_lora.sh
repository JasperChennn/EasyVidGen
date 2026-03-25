#!/usr/bin/env bash
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    . "/opt/conda/etc/profile.d/conda.sh"
else
    export PATH="/opt/conda/bin:$PATH"
fi

# cd /mnt/nas-data-3/hujunjun.hjj/code/wm/anyi/world/Wan-Trainer
# conda activate conda activate /mnt/nas-data-3/hujunjun.hjj/code/wm/anyi/anaconda3/envs/videogen

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
NCCL_DEBUG=INFO

model_name="/mnt/nas-data-3/hujunjun.hjj/code/wm/anyi/checkpoints/Wan2.2-TI2V-5B-Diffusers"
data_path="/mnt/nas-data-3/anyi.cjt/datasets/Cakeify-Dataset/data.json"
data_root="/mnt/nas-data-3/anyi.cjt/datasets/Cakeify-Dataset"
output_dir="outputs/test"


rank=64
lora_alpha=64
noise_shift=4.0

validation_prompt="a person using a knife to cut a cake shaped like a dog"
validation_images="examples/demo.jpg"
num_validation_videos=2

validation_steps=20
num_train_epochs=10
checkpointing_steps=3000
max_train_steps=300

output_dir=${output_dir}_r${rank}

# nohup bash scripts/crouch/train_unipc.sh> logs/train_video_decouple_crouch_multi.log 2>&1 &
PORT=$MASTER_PORT NODE_RANK=$RANK NNODES=$WORLD_SIZE \
# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --config_file config/accelerate_no_deepspeed_single_gpu_0.yaml src/trainers/trainer_lora_i2v.py \
  --pretrained_model_name_or_path="$model_name" \
  --mixed_precision="bf16" \
  --vae_mini_batch=1 \
  --train_data_meta="$data_path" \
  --train_data_dir="$data_root" \
  --video_sample_height=704 \
  --video_sample_width=1280 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --dataloader_num_workers=4 \
  --allow_tf32 \
  --learning_rate=1e-04 \
  --adam_weight_decay=1e-4 \
  --adam_epsilon=1e-8 \
  --adam_beta1=0.9 \
  --adam_beta2=0.95 \
  --max_grad_norm=1.0 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=50 \
  --num_train_epochs=$num_train_epochs \
  --max_train_steps=$max_train_steps \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --checkpointing_steps=$checkpointing_steps \
  --output_dir="$output_dir" \
  --seed=42 \
  --validation_prompt="$validation_prompt" \
  --validation_images="$validation_images" \
  --validation_prompt_separator ::: \
  --validation_epochs=$validation_steps \
  --num_validation_videos $num_validation_videos \
  --max_sequence_length=512 \
  --guidance_scale=5.0 \
  --noise_distribution="normal" \
  --noise_shift=$noise_shift \
  --rank=$rank \
  --lora_alpha=$lora_alpha
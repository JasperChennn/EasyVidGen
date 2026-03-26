#!/usr/bin/env bash
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    . "/opt/conda/etc/profile.d/conda.sh"
else
    export PATH="/opt/conda/bin:$PATH"
fi

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
NCCL_DEBUG=INFO

model_name="checkpoints/Wan2.2-TI2V-5B-Diffusers"
data_path="examples/cakeify/data.json"
data_root="tmp/Open-VFX_Cake-ify/train/Cake-ify"
output_dir="tmp/train_cakeify_i2v_lora"


rank=64
lora_alpha=64
noise_shift=4.0

validation_prompt="Cut it open like you're cutting a cake."
validation_images="examples/demo.jpg"
num_validation_videos=2

validation_steps=100
num_train_epochs=10
checkpointing_steps=2000
max_train_steps=2000

output_dir=${output_dir}_r${rank}

# 单机多卡 + DeepSpeed ZeRO-2：使用下方 ACCELERATE_CONFIG（在仓库根目录执行本脚本）
# 单卡调试：设 ACCELERATE_CONFIG="" 并取消注释下面「单卡」那一行 accelerate launch。
# ACCELERATE_CONFIG="config/accelerate/accelerate_2gpu_zero2.yaml"

# nohup bash scripts/crouch/train_unipc.sh> logs/train_video_decouple_crouch_multi.log 2>&1 &
# 注意：不能在「环境变量赋值 + 反斜杠续行」后面直接写 if，会触发 syntax error。
# 多机时如需 PORT/NODE_RANK，请在调用前 export，或由 accelerate 配置读取。

TRAIN_ARGS=(
  --train_data_meta="$data_path"
  --train_data_dir="$data_root"
  --video_sample_height=480
  --video_sample_width=832
  --video_sample_n_frames=81
  --train_batch_size=1
  --dataloader_num_workers=4
  --allow_tf32
  --learning_rate=1e-04
  --adam_weight_decay=1e-2
  --adam_epsilon=1e-8
  --adam_beta1=0.9
  --adam_beta2=0.99
  --max_grad_norm=1.0
  --lr_scheduler="constant_with_warmup"
  --lr_warmup_steps=50
  --num_train_epochs=$num_train_epochs
  --max_train_steps=$max_train_steps
  --gradient_accumulation_steps=1
  --gradient_checkpointing
  --checkpointing_steps=$checkpointing_steps
  --output_dir="$output_dir"
  --seed=42
  --validation_prompt="$validation_prompt"
  --validation_images="$validation_images"
  --validation_prompt_separator :::
  --validation_epochs=$validation_steps
  --num_validation_videos
  "$num_validation_videos"
  --max_sequence_length=512
  --guidance_scale=5.0
  --noise_distribution="normal"
  --noise_shift=$noise_shift
  --rank=$rank
  --lora_alpha=$lora_alpha
)

if [ -n "$ACCELERATE_CONFIG" ]; then
  # 配置里已含 mixed_precision / num_processes / DeepSpeed，勿再传 --num_processes
  accelerate launch --config_file "$ACCELERATE_CONFIG" trainers/trainer_lora_i2v.py \
    --pretrained_model_name_or_path="$model_name" \
    "${TRAIN_ARGS[@]}"
else
  accelerate launch --num_processes=2 trainers/trainer_lora_i2v.py \
    --pretrained_model_name_or_path="$model_name" \
    --mixed_precision="bf16" \
    --vae_mini_batch=1 \
    "${TRAIN_ARGS[@]}"
fi
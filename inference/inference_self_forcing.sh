export CUDA_VISIBLE_DEVICES=1

# 请在仓库根目录执行本脚本；模型目录需已存在（见 scripts/download_models.sh）
model_name='checkpoints/Wan2.1-1.3B-Self-Forcing-Diffusers'
num_frames=81
height=480 
width=832 
num_inference_steps=4
num_frame_per_block=3
local_attn_size=21

prompt="一只毛茸茸的小怪兽跪在一根正在融化的红蜡烛旁。艺术风格为 3D 写实，注重光影和纹理。画面中的小怪物睁大眼睛、张开嘴巴凝视着火焰，充满了惊叹和好奇。它的姿势和表情传达出一种天真和活泼的气息，仿佛是第一次探索周围的世界。暖色调和戏剧性灯光进一步增强了画面的温馨氛围。"
python inference/inference_self_forcing.py \
  --model_name $model_name \
  --prompt "${prompt}" \
  --num_frames $num_frames \
  --height $height \
  --width $width \
  --num_inference_steps $num_inference_steps \
  --num_frame_per_block $num_frame_per_block \
  --local_attn_size $local_attn_size \
  --output tmp/self_forcing.mp4 \
  --fps 16 \
  --device cuda
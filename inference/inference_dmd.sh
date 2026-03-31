export CUDA_VISIBLE_DEVICES=2

# 请在仓库根目录执行本脚本；模型目录需已存在（见 scripts/download_models.sh）
model_name='checkpoints/Wan2.2-TI2V-5B-4Steps-Diffusers'
num_frames=121
height=704 
width=1280 
num_inference_steps=4
num_frame_per_block=31

prompt="A cat walking on the grass."
python inference/inference_dmd.py \
  --model_name $model_name \
  --prompt "${prompt}" \
  --num_frames $num_frames \
  --height $height \
  --width $width \
  --num_inference_steps $num_inference_steps \
  --num_frame_per_block $num_frame_per_block \
  --output tmp/demo.mp4 \
  --fps 16 \
  --device cuda
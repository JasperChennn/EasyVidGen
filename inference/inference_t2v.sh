export CUDA_VISIBLE_DEVICES=2

# 请在仓库根目录执行本脚本；模型目录需已存在（见 scripts/download_models.sh）
# model_name='checkpoints/Wan2.2-TI2V-5B-Diffusers'
model_name='checkpoints/Wan2.2-TI2V-5B-4Steps-Diffusers'

# mkdir videos
prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
image_path=assets/images/demo.jpg
python inference/inference_i2v.py \
    --model_name $model_name \
    --image_path $image_path \
    --prompt $prompt \
    --num_frames 121 \
    --height 704 \
    --width 1280 \
    --num_inference_steps 4 \
    --guidance_scale 1.0 \
    --output tmp/demo_t2v.mp4 \
    --fps 16 \
    --device cuda

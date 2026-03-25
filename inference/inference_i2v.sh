export CUDA_VISIBLE_DEVICES=2

model_name='../checkpoints/Wan2.2-TI2V-5B-Diffusers'

# mkdir videos
python inference/inference_i2v.py \
    --model_name $model_name \
    --image_path examples/demo.jpg \
    --prompt "The dog is walking happily in the road." \
    --num_frames 49 \
    --height 704 \
    --width 1280 \
    --num_inference_steps 30 \
    --output tmp/demo.mp4 \
    --fps 16 \
    --device cuda

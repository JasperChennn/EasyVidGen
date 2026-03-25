export CUDA_VISIBLE_DEVICES=2

# mkdir videos
python inference/inference_i2v.py \
    --model_name /mnt/28t/hujunjun.hjj/wm/checkpoints/Wan2.2-TI2V-5B-Diffusers \
    --image_path example/demo.jpg \
    --prompt "The dog is walking happily in the road." \
    --num_frames 81 \
    --height 704 \
    --width 1280 \
    --num_inference_steps 30 \
    --seed 42 \
    --output videos/demo.mp4 \
    --fps 16 \
    --device cuda

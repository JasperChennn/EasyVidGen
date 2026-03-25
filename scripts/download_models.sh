pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir checkpoints/Wan2.1-T2V-1.3B-Diffusers
huggingface-cli download --resume-download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir checkpoints/Wan2.2-TI2V-5B-Diffusers
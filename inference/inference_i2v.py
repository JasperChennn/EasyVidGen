import os
import sys
import json
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

import argparse
import datetime
import torch
import numpy as np
from PIL import Image
from diffusers.utils import export_to_video, load_image

from diffusers import AutoencoderKLWan, WanTransformer3DModel
from src.pipelines.pipeline_i2v import WanImageToVideoPipeline
# from src.models.transformer import WanTransformer3DModel


def _init_models(model_name, device='cuda'):
    transformer = WanTransformer3DModel.from_pretrained(
        model_name,
        subfolder='transformer',
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to(device)

    vae = AutoencoderKLWan.from_pretrained(
        model_name,
        subfolder='vae',
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to(device)

    pipeline = WanImageToVideoPipeline.from_pretrained(
        model_name,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to(device)

    return pipeline

@torch.no_grad()
def generate(pipeline, params):
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    output = pipeline(
        image=load_image(params.image_path),
        prompt=params.prompt,
        negative_prompt=negative_prompt,
        num_frames=params.num_frames,
        height=params.height,
        width=params.width,
        num_inference_steps=params.num_inference_steps,
        generator=(
            torch.Generator(device='cuda').manual_seed(params.seed)
            if params.seed else None
        ),
    ).frames[0]

    output_path = params.output or "output.mp4"
    export_to_video(output, output_path, fps=params.fps)
    print(f"Video saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Image to Video Generator")
    parser.add_argument("--model_name", type=str, required=True, help="预训练模型名称或路径")
    parser.add_argument("--image_path", type=str, required=True, help="输入图片路径")
    parser.add_argument("--prompt", type=str, required=True, help="生成视频的描述文本")
    parser.add_argument("--num_frames", type=int, default=49, help="基础帧数（最终视频帧数为 num_frames*4+1）")
    parser.add_argument("--height", type=int, default=704, help="视频帧的高度")
    parser.add_argument("--width", type=int, default=1280, help="视频帧的宽度")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="推理步数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--output", type=str, default="output.mp4", help="输出视频文件路径")
    parser.add_argument("--fps", type=int, default=16, help="输出视频帧率")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备 cuda 或 cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    pipeline = _init_models(args.model_name, device=args.device)
    generate(pipeline, args)

if __name__ == "__main__":
    main()


import os
import sys

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

import argparse
import torch
from diffusers import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, UMT5EncoderModel

from easyvid.models.wan.transformer_casual import WanTransformer3DModel
from easyvid.pipelines.community.dmd.pipeline_sample_dmd import WanSelfForcingPipeline


def _init_models(model_name, device="cuda", params=None):
    dtype = torch.bfloat16
    kwargs = {"local_files_only": True}

    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer", **kwargs)
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder", dtype=dtype, **kwargs
    )
    scheduler = UniPCMultistepScheduler.from_pretrained(
        model_name, subfolder="scheduler", shift=5.0, **kwargs
    )
    transformer = WanTransformer3DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=dtype, **kwargs
    )
    vae = AutoencoderKLWan.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype, **kwargs)
    transformer.config.local_attn_size = params.local_attn_size
    pipeline = WanSelfForcingPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        transformer=transformer,
        transformer_2=None,
        boundary_ratio=None,
        expand_timesteps=False,
        num_frame_per_block=params.num_frame_per_block,
        independent_first_frame=params.independent_first_frame,
        context_noise=0.0,
        context_noise_inject=params.context_noise_inject,
        same_step_across_blocks=False,
        last_step_only=False,
    )
    return pipeline.to(device)


@torch.no_grad()
def generate(pipeline, params):
    gen = (
        torch.Generator(device=params.device).manual_seed(params.seed)
        if params.seed is not None
        else None
    )

    output = pipeline(
        prompt=params.prompt,
        height=params.height,
        width=params.width,
        num_frames=params.num_frames,
        num_inference_steps=params.num_inference_steps,
        num_frame_per_block=params.num_frame_per_block,
        independent_first_frame=params.independent_first_frame,
        context_noise=params.context_noise,
        context_noise_inject=params.context_noise_inject,
        random_exit_per_block=params.random_exit_per_block,
        generator=gen,
    ).frames[0]

    output_path = params.output or "output_dmd.mp4"
    export_to_video(output, output_path, fps=params.fps)
    print(f"Video saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="DMD 因果 T2V 推理（WanDMDSamplePipeline）")
    parser.add_argument("--model_name", type=str, required=True, help="预训练模型目录或 HuggingFace id")
    parser.add_argument("--prompt", type=str, required=True, help="文本描述")
    parser.add_argument("--num_frames", type=int, default=81, help="视频帧数")
    parser.add_argument("--height", type=int, default=480, help="高度")
    parser.add_argument("--width", type=int, default=832, help="宽度")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--num_frame_per_block", type=int, default=3, help="每块 latent 帧数")
    parser.add_argument("--local_attn_size", type=int, default=21, help="local attention size")
    parser.add_argument("--independent_first_frame", action="store_true", help="首帧单独成块")
    parser.add_argument("--context_noise", type=float, default=0.0, help="KV 刷新 timestep")
    parser.add_argument("--context_noise_inject", action="store_true", help="KV 刷新前对块输出加噪")
    parser.add_argument("--random_exit_per_block", action="store_true", help="每块随机子步出口")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--output", type=str, default="output_dmd.mp4", help="输出 mp4")
    parser.add_argument("--fps", type=int, default=16, help="帧率")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = _init_models(args.model_name, device=args.device, params=args)
    generate(pipeline, args)


if __name__ == "__main__":
    main()

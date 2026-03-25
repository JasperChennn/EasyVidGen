import os
import sys
import warnings

import torch
from diffusers import AutoencoderKLWan

import numpy as np
from pathlib import Path
# workspace path for local imports
current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent))
os.environ["TRANSFORMERS_ALLOW_TORCH_LOAD_WITH_UNSAFE_WEIGHTS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

def _load_wan_vae(model_path, *, subfolder: str = "vae"):
    """
    Load Wan VAE weights. Upstream ``vae/config.json`` may list keys (e.g. ``clip_output``)
    that this diffusers build does not declare; passing a cleaned ``config`` into
    ``from_pretrained`` collides with internal ``from_config(config, ...)`` and raises
    ``multiple values for argument 'config'``. We load normally and silence that benign notice.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*config attributes.*not expected and will be ignored.*",
        )
        return AutoencoderKLWan.from_pretrained(
            model_path,
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )


class VAEEncoder:
    def __init__(self, model_path):
        self.model = _load_wan_vae(model_path)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def encode_to_latents(self, video, vae_mini_batch=1):
        """
        Wan VAE time grouping (along the frame axis):
        - Chunk 1: source frame [0] -> latent[0]
        - Chunk 2: frames [1:5] -> latent[1]
        - Chunk 3: frames [5:9] -> latent[2]
        - ...
        Number of latent time steps: 1 + (F - 1) // 4 (e.g. 140 frames -> 35 steps).
        """
        video = video.to(self.model.device, self.model.dtype)  # e.g. [1, 3, 140, H, W]
        def _slice_vae(pixel_values):
            bs = vae_mini_batch
            new_pixel_values = []
            for i in range(0, pixel_values.shape[0], bs):
                pixel_values_bs = pixel_values[i : i + bs]
                pixel_values_bs = self.model.encode(pixel_values_bs).latent_dist
                pixel_values_bs = pixel_values_bs.sample()
                new_pixel_values.append(pixel_values_bs)
            return torch.cat(new_pixel_values, dim = 0)
        latents = _slice_vae(video)
        latents_mean = (
            torch.tensor(self.model.config.latents_mean)
            .view(1, self.model.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.model.config.latents_std).view(1, self.model.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * latents_std
        return latents  # e.g. [1, z_dim, T_latent, h, w]

    @torch.no_grad()
    def decode_to_video(self, latents, vae_mini_batch=1, to_save=False):
        # Match model device/dtype (often GPU + bfloat16) for faster decode
        latents = latents.to(self.model.device, self.model.dtype)  # e.g. [1, z_dim, T, h, w]
        def _slice_vae_decode(latents):
            bs = vae_mini_batch
            decoded_frames = []
            for i in range(0, latents.shape[0], bs):
                latents_bs = latents[i : i + bs]
                latents_mean = (
                    torch.tensor(self.model.config.latents_mean)
                    .view(1, self.model.config.z_dim, 1, 1, 1)
                    .to(latents_bs.device, latents_bs.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.model.config.latents_std).view(
                    1, self.model.config.z_dim, 1, 1, 1
                ).to(latents_bs.device, latents_bs.dtype)
                
                
                latents_bs = latents_bs / latents_std + latents_mean
                
                decoded = self.model.decode(latents_bs).sample
                decoded_frames.append(decoded)
            return torch.cat(decoded_frames, dim=0)
        video = _slice_vae_decode(latents)  # e.g. [1, 3, T_frames, H, W]

        if to_save:
            video_list = []
            for sub_video in video:
                video_np = (sub_video.permute(1,2,3,0).cpu().float() * 127.5 + 127.5).clamp(0,255).numpy().astype(np.uint8)
                video_list.append(video_np)
            return video_list  # list of (T, H, W, 3) uint8 arrays per batch item

        return video


if __name__ == "__main__":
    import numpy as np
    import imageio
    from diffusers.utils import load_video, export_to_video
    model_id = "checkpoints/Wan2.2-TI2V-5B-Diffusers"
    vae_encoder = VAEEncoder(model_id)
    vae_encoder.model.to("cuda")
    video_path = "EasyVidGen/tmp/demo.mp4"
    video = load_video(video_path)
    video = np.stack([np.array(frame) for frame in video], axis=0)  # (T, H, W, 3)
    video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, T, H, W]
    video = (video.float() / 127.5) - 1.0
    latents = vae_encoder.encode_to_latents(video, vae_mini_batch=1)
    print(f"Latents shape: {latents.shape}")
    video_np = vae_encoder.decode_to_video(latents, vae_mini_batch=1, to_save=True)
    imageio.mimwrite("EasyVidGen/tmp/debug.mp4", video_np[0], fps=30, codec='libx264')
    # print(f"Decoded video saved to output_video.mp4")
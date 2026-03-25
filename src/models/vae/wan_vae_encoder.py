import torch
from diffusers import AutoencoderKLWan

import os, sys
import numpy as np
from pathlib import Path
# get current workspace
current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent))
os.environ["TRANSFORMERS_ALLOW_TORCH_LOAD_WITH_UNSAFE_WEIGHTS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

class VAEEncoder:
    def __init__(self, model_path):
        self.model = AutoencoderKLWan.from_pretrained(
            model_path, 
            subfolder="vae", 
        )
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def encode_to_latents(self, video, vae_mini_batch=1):
        """
        VAE的时间压缩规则：
        - 第1次编码: 原始帧[0] → latent[0]
        - 第2次编码: 原始帧[1:5] → latent[1]
        - 第3次编码: 原始帧[5:9] → latent[2]
        - ...
        - 总压缩比: 1 + (F-1)//4，例如140帧 → 35个latent时间步)
        """
        video = video.to(self.model.device, self.model.dtype) # [1, 3, 140, 240, 320]
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
        return latents # [1, 16, 35, 30, 40]

    @torch.no_grad()
    def decode_to_video(self, latents, vae_mini_batch=1, to_save=False):
        # 把 latents 转到和 model 一样的 device 和 dtype 上 (通常是 GPU 和 bfloat16)，以加速解码过程
        latents = latents.to(self.model.device, self.model.dtype) # [1, 16, 35, 30, 40]
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
        video = _slice_vae_decode(latents) # [1, 3, 137, 240, 320]

        if to_save:
            video_list = []
            for sub_video in video:
                video_np = (sub_video.permute(1,2,3,0).cpu().float() * 127.5 + 127.5).clamp(0,255).numpy().astype(np.uint8)
                video_list.append(video_np)
            return video_list # (137, 240, 320, 3)

        return video


if __name__ == "__main__":
    import numpy as np
    import imageio
    from diffusers.utils import load_video, export_to_video
    model_id = ""
    vae_encoder = VAEEncoder(model_id)
import torch

def encode_to_latents(pixel_values, vae_stream, vae, vae_mini_batch, weight_dtype):
    with torch.no_grad():
        # This way is quicker when batch grows up
        def _slice_vae(pixel_values):
            bs = vae_mini_batch
            new_pixel_values = []
            for i in range(0, pixel_values.shape[0], bs):
                pixel_values_bs = pixel_values[i : i + bs]
                pixel_values_bs = vae.encode(pixel_values_bs).latent_dist
                pixel_values_bs = pixel_values_bs.sample()
                new_pixel_values.append(pixel_values_bs)
            return torch.cat(new_pixel_values, dim = 0)
        if vae_stream is not None:
            vae_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(vae_stream):
                latents = _slice_vae(pixel_values)
        else:
            latents = _slice_vae(pixel_values)

        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        latents = (latents - latents_mean) * latents_std
    return latents.to(weight_dtype)

def decode_to_videos(latents, vae_stream, vae, vae_mini_batch, weight_dtype):
    with torch.no_grad():
        # decode latents to video
        def _slice_vae_decode(latents):
            bs = vae_mini_batch
            decoded_frames = []
            for i in range(0, latents.shape[0], bs):
                latents_bs = latents[i : i + bs]
                latents_mean = (
                    torch.tensor(vae.config.latents_mean)
                    .view(1, vae.config.z_dim, 1, 1, 1)
                    .to(latents_bs.device, latents_bs.dtype)
                )
                latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
                    1, vae.config.z_dim, 1, 1, 1
                ).to(latents_bs.device, latents_bs.dtype)
                
                
                latents_bs = latents_bs / latents_std + latents_mean
                
                decoded = vae.decode(latents_bs).sample
                decoded_frames.append(decoded)
            
            return torch.cat(decoded_frames, dim=0)

        if vae_stream is not None:
            vae_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(vae_stream):
                decoded = _slice_vae_decode(latents)
        else:
            decoded = _slice_vae_decode(latents)

        return decoded.to(weight_dtype)

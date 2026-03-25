import torch
from torch.distributions import Beta


class BetaTimestepSampler:
    """Beta 分布时间步采样。"""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.5,
        noise_s: float = 0.998,
        num_train_timesteps: int = 1000,
    ):
        self.distribution_type = "beta"
        self.alpha = alpha
        self.beta = beta
        self.noise_s = noise_s
        self.num_train_timesteps = num_train_timesteps
        self.dist = Beta(
            torch.tensor(self.alpha),
            torch.tensor(self.beta),
        )

    def sample(self, batch_size: int, device: torch.device = None):
        timesteps = self.dist.sample((batch_size,))
        timesteps = (self.noise_s - timesteps) / self.noise_s
        if device is not None:
            timesteps = timesteps.to(device=device)
        timesteps = torch.clamp(timesteps, 0.005, 0.998)
        return timesteps, torch.round(timesteps * self.num_train_timesteps).long()

    def sample_for(self, x: torch.Tensor):
        sigmas, _ = self.sample(x.shape[0], x.device)
        return sigmas

    def set_timesteps(self, num_inference_timesteps: int):
        self.num_inference_timesteps = num_inference_timesteps
        self.timesteps = torch.linspace(self.num_train_timesteps, 0, num_inference_timesteps + 1)
        self.timesteps = self.timesteps[:-1]
        self.sigmas = self.timesteps / self.num_train_timesteps
        return self.timesteps

    def add_noise(self, original_samples, noise, timestep):
        sigma = timestep / self.num_train_timesteps
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        return (1 - sigma) * original_samples + sigma * noise

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        device = sample.device
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0.0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * ((sigma_ - sigma).to(device))
        return prev_sample

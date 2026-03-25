import torch
from torch.distributions import Normal, Uniform


class ShiftedLogitNormTimestepSampler:
    """移位 Logit 正态或均匀分布的时间步采样（不含 Beta）。"""

    def __init__(
        self,
        distribution_type: str = "normal",
        std: float = 1.0,
        shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ):
        assert distribution_type in ("normal", "uniform"), (
            f"ShiftedLogitNorm 仅支持 normal/uniform，收到：{distribution_type}"
        )
        self.distribution_type = distribution_type
        self.std = std
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps

        if distribution_type == "normal":
            self.dist = Normal(0.0, 1.0)
        else:
            self.dist = Uniform(0.0, 1.0)

    def sample(self, batch_size: int, device: torch.device = None):
        timesteps = self.dist.sample((batch_size,))
        if self.distribution_type == "normal":
            timesteps = timesteps * self.std
            timesteps = torch.sigmoid(timesteps)
        timesteps = (timesteps * self.shift) / (1 + (self.shift - 1) * timesteps)
        if device is not None:
            timesteps = timesteps.to(device=device)
        timesteps = torch.clamp(timesteps, 0.005, 0.998)
        return timesteps, torch.round(timesteps * self.num_train_timesteps).long()


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

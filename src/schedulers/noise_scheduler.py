import torch

class ShiftedLogitNormalTimestepSampler:
    """
    Samples timesteps from a shifted logit-normal distribution,
    where the shift is determined by the sequence length.
    """

    def __init__(self, std: float = 1.0, shift: float = 3.0, distribution_type: str = "normal"):
        self.distribution_type = distribution_type
        if distribution_type == "normal":
            self.dist = torch.distributions.normal.Normal(0, 1)
        elif distribution_type == "uniform":
            self.dist = torch.distributions.uniform.Uniform(0, 1)
        self.std = std
        self.shift = shift

    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """Sample timesteps for a batch from a shifted logit-normal distribution.

        Args:
            batch_size: Number of timesteps to sample
            seq_length: Length of the sequence being processed, used to determine the shift
            device: Device to place the samples on

        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by seq_length
        """
        # shift = self._get_shift_for_sequence_length(seq_length)
        if self.distribution_type == "normal":
            timesteps = self.dist.sample((batch_size,)).to(device=device)
            timesteps = timesteps * self.std
            timesteps = torch.sigmoid(timesteps)
        elif self.distribution_type == "uniform":
            timesteps = self.dist.sample((batch_size,)).to(device=device)

        timesteps = (timesteps * self.shift) / (1 + (self.shift - 1) * timesteps)
        return timesteps

    def sample_for(self, batch: torch.Tensor, rate=None) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by the sequence length
            of the input batch

        Raises:
            ValueError: If the input batch does not have 3 dimensions
        """
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, seq_length, device=batch.device, rate=rate)


SAMPLERS = {
    "shifted_logit_normal": ShiftedLogitNormalTimestepSampler,
}


def example() -> None:
    import matplotlib.pyplot as plt  # type: ignore

    # sampler = ShiftedLogitNormalTimestepSamplerV2(shift=3.0, std=1.0, distribution_type='uniform')
    sampler = ShiftedLogitNormalTimestepSampler(shift=1.5, std=1.0, distribution_type='uniform')
    # sampler = ShiftedLogitNormalTimestepSampler()
    dummy_for_sampling = torch.zeros(20000, 1, 1)
    # print(sampler._get_shift_for_sequence_length(seq_length=1))
    for seq_length in [1]:
        # samples = sampler.sample(batch_size=1_000, seq_length=seq_length)
        samples = sampler.sample_for(dummy_for_sampling, rate=0.08)#.to(torch.bfloat16)
        # plot the histogram of the samples
        plt.hist(samples.float().numpy(), bins=100, density=True)
        plt.title(f"Timestep Samples for Sequence Length {seq_length}")
        plt.xlabel("Timestep")
        plt.ylabel("Density")
        plt.show()
        plt.savefig(f"biased_stage2.png")


if __name__ == "__main__":
    example()

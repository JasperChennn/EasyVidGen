import torch

from .beta_timestep_sampler import BetaTimestepSampler
from .shift_logit_norm import ShiftedLogitNormTimestepSampler


class NoiseTimestepSampler:
    """
    统一的时间步采样器，支持以下分布类型：
    - normal: 移位Logit正态分布
    - uniform: 移位Logit均匀分布
    - beta: Beta分布（见 BetaTimestepSampler）
    """

    def __init__(
        self,
        distribution_type: str = "beta",
        std: float = 1.0,
        shift: float = 3.0,
        alpha: float = 1.0,
        beta: float = 1.5,
        noise_s: float = 0.998,
        num_train_timesteps: int = 1000,
    ):
        assert distribution_type in ("normal", "uniform", "beta"), (
            f"不支持的分布类型：{distribution_type}，可选：normal/uniform/beta"
        )
        self.distribution_type = distribution_type
        self.num_train_timesteps = num_train_timesteps

        if distribution_type in ("normal", "uniform"):
            self._impl = ShiftedLogitNormTimestepSampler(
                distribution_type=distribution_type,
                std=std,
                shift=shift,
                num_train_timesteps=num_train_timesteps,
            )
        else:
            self._impl = BetaTimestepSampler(
                alpha=alpha,
                beta=beta,
                noise_s=noise_s,
                num_train_timesteps=num_train_timesteps,
            )

    def __getattr__(self, name):
        return getattr(self._impl, name)


# 历史命名：训练脚本在 beta 等分布下仍从此名导入
ShiftedLogitNormalTimestepSampler = NoiseTimestepSampler


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    SAMPLE_NUM = 10000
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PLOT_BINS = 50
    PLOT_FIGSIZE = (15, 5)

    normal_sampler = NoiseTimestepSampler(
        distribution_type="normal",
        std=1.0,
        shift=3.0,
        num_train_timesteps=1000,
    )
    _, normal_samples = normal_sampler.sample(batch_size=SAMPLE_NUM, device=DEVICE)

    uniform_sampler = NoiseTimestepSampler(
        distribution_type="uniform",
        shift=3.0,
        num_train_timesteps=1000,
    )
    _, uniform_samples = uniform_sampler.sample(batch_size=SAMPLE_NUM, device=DEVICE)

    beta_sampler = NoiseTimestepSampler(
        distribution_type="beta",
        alpha=1.5,
        beta=1.0,
        noise_s=0.999,
        num_train_timesteps=1000,
    )
    _, beta_samples = beta_sampler.sample(batch_size=SAMPLE_NUM, device=DEVICE)

    normal_samples_np = normal_samples.cpu().numpy()
    uniform_samples_np = uniform_samples.cpu().numpy()
    beta_samples_np = beta_samples.cpu().numpy()

    print("=== 采样结果统计信息 ===")
    print(f"Normal分布 - 均值：{np.mean(normal_samples_np):.4f}，标准差：{np.std(normal_samples_np):.4f}")
    print(f"Uniform分布 - 均值：{np.mean(uniform_samples_np):.4f}，标准差：{np.std(uniform_samples_np):.4f}")
    print(f"Beta分布 - 均值：{np.mean(beta_samples_np):.4f}，标准差：{np.std(beta_samples_np):.4f}")

    plt.rcParams["font.size"] = 12
    fig, axes = plt.subplots(1, 3, figsize=PLOT_FIGSIZE)

    axes[0].hist(normal_samples_np, bins=PLOT_BINS, density=True, alpha=0.7, color="#1f77b4", edgecolor="black")
    axes[0].set_title("Shifted Logit Normal Distribution", fontsize=14)
    axes[0].set_xlabel("Timestep Value", fontsize=12)
    axes[0].set_ylabel("Probability Density", fontsize=12)
    axes[0].grid(alpha=0.3)

    axes[1].hist(uniform_samples_np, bins=PLOT_BINS, density=True, alpha=0.7, color="#ff7f0e", edgecolor="black")
    axes[1].set_title("Shifted Logit Uniform Distribution", fontsize=14)
    axes[1].set_xlabel("Timestep Value", fontsize=12)
    axes[1].set_ylabel("Probability Density", fontsize=12)
    axes[1].grid(alpha=0.3)

    axes[2].hist(beta_samples_np, bins=PLOT_BINS, density=True, alpha=0.7, color="#2ca02c", edgecolor="black")
    axes[2].set_title("Beta Distribution", fontsize=14)
    axes[2].set_xlabel("Timestep Value", fontsize=12)
    axes[2].set_ylabel("Probability Density", fontsize=12)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("timestep_distributions.png", dpi=300, bbox_inches="tight")
    plt.show()

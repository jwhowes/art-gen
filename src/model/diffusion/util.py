import torch

from torch import nn
from typing import Optional

from ..util import LayerNorm2d


class NoiseScheduler(nn.Module):
    def __init__(
            self, num_timesteps: int = 1000, schedule: str = "linear"
    ):
        super(NoiseScheduler, self).__init__()
        self.num_timesteps = num_timesteps

        if schedule == "linear":
            self.register_buffer(
                "beta",
                torch.linspace(1e-4, 20 / num_timesteps, num_timesteps),
                persistent=False
            )
        else:
            raise NotImplementedError("Schedule method not found")

        self.register_buffer(
            "alpha",
            (1 - self.beta).cumprod(0),
            persistent=False
        )
        self.register_buffer(
            "sqrt_alpha",
            self.alpha.sqrt(),
            persistent=False
        )
        self.register_buffer(
            "sqrt_one_minus_alpha",
            (1 - self.alpha).sqrt(),
            persistent=False
        )

    def add_noise(self, x, eps=None, t=None):
        B = x.shape[0]

        return_eps = (eps is None)
        return_t = (t is None)

        if eps is None:
            eps = torch.randn_like(x)

        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,))

        x_t = x * self.sqrt_alpha[t].view(B, 1, 1, 1) + eps * self.sqrt_one_minus_alpha[t].view(B, 1, 1, 1)

        if return_eps and return_t:
            return x_t, eps, t

        if return_eps:
            return x_t, eps

        if return_t:
            return x_t, t

        return x_t


class FiLM2d(nn.Module):
    def __init__(self, d_model: int, d_t: int, *norm_args, **norm_kwargs):
        super(FiLM2d, self).__init__()
        self.norm = LayerNorm2d(d_model, *norm_args, elementwise_affine=False, bias=False, **norm_kwargs)
        self.gamma = nn.Linear(d_t, d_model)
        self.beta = nn.Linear(d_t, d_model)

    def forward(self, x, t):
        B = x.shape[0]
        g = self.gamma(t).view(B, -1, 1, 1)
        b = self.beta(t).view(B, -1, 1, 1)

        return g * self.norm(x) + b


class ConvNeXtFiLMBlock(nn.Module):
    def __init__(self, d_model: int, d_t: int, d_hidden: Optional[int] = None, norm_eps: float = 1e-6):
        super(ConvNeXtFiLMBlock, self).__init__()
        d_hidden = 4 * d_model if d_hidden is None else d_hidden

        self.dwconv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.norm = FiLM2d(d_model, d_t, eps=norm_eps)

        self.pw_module = nn.Sequential(
            nn.Conv2d(d_model, d_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_hidden, d_model, kernel_size=1)
        )

    def forward(self, x, t):
        return x + self.pw_module(
            self.norm(
                self.dwconv(x), t
            )
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, d_model: int, base: float = 1e5):
        super(SinusoidalPosEmb, self).__init__()
        assert d_model % 2 == 0

        self.register_buffer(
            "theta",
            1.0 / (base ** (2 * torch.arange(d_model // 2) / d_model)),
            persistent=False
        )

    def forward(self, x):
        x = x.float().view(-1, 1) * self.theta

        return torch.stack((
            x.cos(),
            x.sin()
        ), dim=-1).flatten(1)

from typing import Optional, Tuple, Literal

from torch import nn, Tensor


class ResNetBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: Optional[int] = None, num_groups: int = 32, norm_eps: float = 1e-6,
            sample: Optional[Literal["down"] | Literal["up"]] = None
    ):
        super(ResNetBlock, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels, eps=norm_eps),
            nn.SiLU()
        )

        if sample == "down":
            self.sample = nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        elif sample == "up":
            self.sample = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        elif out_channels != in_channels:
            self.sample = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            )
        else:
            self.sample = nn.Identity()

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.sample(x)

        x = self.sample(self.block1(x))

        return residual + self.block2(x)


class ResNet(nn.Module):
    def __init__(
            self, in_channels: int, dims: Tuple[int, ...], depths: Tuple[int, ...],
            sample: Literal["down"] | Literal["up"]
    ):
        super(ResNet, self).__init__()
        assert len(dims) == len(depths), "Must specify depth and dim for each layer."
        assert min(depths) >= 1, "Each layer must have depth at least one."

        layers = [
            nn.Conv2d(in_channels, dims[0], kernel_size=5, padding=2)
        ]

        for i in range(len(dims) - 1):
            layers += [
                ResNetBlock(dims[i])
                for _ in range(depths[i] - 1)
            ] + [
                ResNetBlock(dims[i], dims[i + 1], sample=sample)
            ]

        layers += [
            ResNetBlock(dims[-1])
            for _ in range(depths[-1])
        ]

        self.module = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)

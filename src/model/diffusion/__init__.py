# TODO:
#   1. FiLM
#   2. FiLMUNet
#   3. Diffusion Noise Scheduler
#   4. DDPM Loss (with pretrained VAE)
#   5. Diffusion Sampling
from torch import nn

from .util import ConvNeXtFiLMBlock, SinusoidalPosEmb


class DiffusionModel(nn.Module):
    def __init__(
            self, image_channels: int
    ):
        pass

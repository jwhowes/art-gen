import os

import torch
from pydantic import BaseModel

from ..data import DatasetConfig
from ..model.dit import DiTConfig, DiT
from ..model.vae import VAEEncoder
from ..config import Config

from .vae import VAETrainConfig
from .abstract import AbstractTrainer


class FlowMatchTrainConfig(Config):
    class FlowMatchTrain(BaseModel):
        num_epochs: int = 32
        log_interval: int = 100

        lr: float = 1e-4
        weight_decay: float = 0.0
        clip_grad: float = 1.0

        sigma_min: float = 1e-4

        lognorm_mean: float = -0.5
        lognorm_var: float = 1.0

    model: DiTConfig

    train: FlowMatchTrain
    data: DatasetConfig


class FlowMatchTrainer(AbstractTrainer):
    metrics = ["loss"]
    config_cls = FlowMatchTrainConfig

    def train(self):
        vae_config = VAETrainConfig.from_yaml(
            os.path.join("experiments", self.config.model.vae.experiment)
        )

        encoder = VAEEncoder.from_config(vae_config.encoder)
        state_dict = torch.load(
            os.path.join(
                "experiments",
                self.config.model.vae.experiment,
                f"checkpoint_{self.config.model.vae.checkpoint:03}.pt"
            ),
            weights_only=True,
            map_location="cpu"
        )

        encoder.load_state_dict(state_dict["encoder"])
        del state_dict

        encoder.eval()
        encoder.requires_grad_(False)

        model = DiT.from_config(self.config.model)

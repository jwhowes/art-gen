import os
from abc import ABC, abstractmethod
from typing import Type, Tuple, Self, Dict
from datetime import datetime

import torch
from torch import Tensor

from ..config import Config


class AbstractTrainer(ABC):
    metrics: Tuple[str]
    config_cls: Type[Config]

    def __init__(self, exp_dir: str, ckpt_dir: str, log_path: str, checkpoint: int, config: Config):
        self.exp_dir = exp_dir
        self.ckpt_dir = ckpt_dir
        self.log_path = log_path
        self.checkpoint = checkpoint
        self.config = config

    @classmethod
    def from_config(
            cls, config_path: str,
            checkpoint: int = 1
    ) -> Self:
        assert os.path.isfile(config_path), f"Director {config_path} not found."

        exp_name = os.path.splitext(os.path.basename(config_path))[0]
        exp_dir = os.path.join("experiments", exp_name)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        log_path = os.path.join(exp_dir, "log.csv")

        config = cls.config_cls.from_yaml(config_path)

        if checkpoint == 1:
            if not os.path.isdir(ckpt_dir):
                os.makedirs(ckpt_dir)

            config.to_yaml(os.path.join(exp_dir, "config.yaml"))

            with open(log_path, "w+") as f:
                f.write(f"epoch,{','.join(cls.metrics)},timestamp\n")

        return cls(
            exp_dir=exp_dir,
            ckpt_dir=ckpt_dir,
            log_path=log_path,
            config=config,
            checkpoint=checkpoint
        )

    def save_checkpoint(self, state_dict: Dict[str, Tensor | Dict[str, Tensor]], *metrics: float):
        with open(self.log_path, "a") as f:
            f.write(f"{self.checkpoint},{','.join([f'{m}' for m in metrics])},"
                    f"{datetime.now()}\n")

        torch.save(
            state_dict,
            os.path.join(self.ckpt_dir, f"checkpoint_{self.checkpoint:03}.pt")
        )

        self.checkpoint += 1

    @abstractmethod
    def train(self):
        ...

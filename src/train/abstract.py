import os
from abc import ABC, abstractmethod
from typing import Type, Tuple, Self, Dict
from dataclasses import dataclass
from datetime import datetime

import torch
from torch import Tensor

from ..config import Config


class AbstractTrainConfig(ABC, Config):
    metrics: Tuple[str]


@dataclass
class AbstractTrainer(ABC):
    exp_dir: str
    ckpt_dir: str
    log_path: str

    model_config: Config
    data_config: Config
    train_config: AbstractTrainConfig

    checkpoint: int

    @classmethod
    def from_dir(
            cls, config_dir: str,
            model_type: Type[Config], data_type: Type[Config], train_type: Type[AbstractTrainConfig],
            checkpoint: int = 1
    ) -> Self:
        assert os.path.isdir(config_dir), f"Director {config_dir} not found."

        exp_name = os.path.basename(config_dir)
        exp_dir = os.path.join("experiments", exp_name)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        log_path = os.path.join(exp_dir, "log.csv")

        model_config = model_type.from_yaml(os.path.join(config_dir, "model.yaml"))
        data_config = data_type.from_yaml(os.path.join(config_dir, "data.yaml"))
        train_config = train_type.from_yaml(os.path.join(config_dir, "train.yaml"))

        if checkpoint == 1:
            if not os.path.isdir(ckpt_dir):
                os.makedirs(ckpt_dir)

            model_config.to_yaml(os.path.join(exp_dir, "model.yaml"))
            data_config.to_yaml(os.path.join(exp_dir, "data.yaml"))
            train_config.to_yaml(os.path.join(exp_dir, "train.yaml"))

            with open(log_path, "w+") as f:
                f.write(f"epoch,{','.join(train_config.metrics)},timestamp\n")

        return AbstractTrainer(
            exp_dir=exp_dir,
            ckpt_dir=ckpt_dir,
            log_path=log_path,
            model_config=model_config,
            data_config=data_config,
            train_config=train_config,
            checkpoint=checkpoint
        )

    def log(self, state_dict: Dict[str, Tensor | Dict[str, Tensor]], metrics: Dict[str, float]):
        with open(self.log_path, "a") as f:
            f.write(f"{self.checkpoint},{','.join([f'{metrics[m]:.4f}' for m in self.train_config.metrics])},"
                    f"{datetime.now()}\n")

        torch.save(
            state_dict,
            os.path.join(self.ckpt_dir, f"checkpoint_{self.checkpoint:03}.pt")
        )

        self.checkpoint += 1

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from ..config import Config


class AbstractModel(ABC, nn.Module):
    @staticmethod
    @abstractmethod
    def from_config(config: Config) -> AbstractModel:
        ...

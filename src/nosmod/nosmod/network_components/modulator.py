from enum import Enum, auto
from typing import Optional, Type, Union

import torch
from pydantic import Field
from torch import Tensor, nn

from ..utils import BaseConfig


class ModulatorParams(BaseConfig[Union["PSKModulator", "AMModulator"]]):
    target: "ModulationType" = Field(default_factory=lambda: ModulationType.PSK)
    num_constellation_points: Optional[int] = 4


class PSKModulator(nn.Module):
    def __init__(self, params: ModulatorParams):
        super().__init__()
        self.params = params

    def forward(self, x_x: Tensor, x_y: Tensor) -> Tensor:
        x_x_mod = torch.exp(
            1j * 2 * torch.pi * x_x / self.params.num_constellation_points
        )
        x_y_mod = torch.exp(
            1j * 2 * torch.pi * x_y / self.params.num_constellation_points
        )
        return torch.stack((x_x_mod, x_y_mod), dim=-1)


class AMModulator(nn.Module):
    def __init__(self, params: ModulatorParams):
        super().__init__()
        self.params = params

    def forward(self, x_x: Tensor, x_y: Tensor) -> Tensor:
        levels = torch.linspace(-1, 1, self.params.num_constellation_points)
        x_x_mod = levels[x_x.long()]
        x_y_mod = levels[x_y.long()]
        return torch.stack((x_x_mod, x_y_mod), dim=-1).to(torch.complex64)


class ModulationType(Enum):
    PSK = auto()
    AM = auto()
    # ARBITRARY = auto()

    @classmethod
    def setup_target(
        cls, params: ModulatorParams, constpoints: Optional[Tensor] = None
    ) -> nn.Module:
        match params.target:
            case cls.PSK:
                return PSKModulator(params)
            case cls.AM:
                return AMModulator(params)
            case _:
                raise ValueError(f"Unknown modulation type: {params.target}")

from enum import Enum, auto
from typing import Optional, Union

import torch
from torch import Tensor, nn

from ..utils import BaseConfig


class ModulatorParams(BaseConfig[Union["PSKModulator", "AMModulator"]]):
    target: "ModulationType"
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


# class ArbitraryModulator(nn.Module):
#     def __init__(self, constpoints: Tensor):
#         super().__init__()
#         self.constpoints = constpoints

#     def forward(self, x_x: Tensor, x_y: Tensor) -> Tensor:
#         sps = self.constpoints.shape[-1]

#         indices_x = x_x.long().unsqueeze(-1).expand(-1, sps)
#         indices_y = x_y.long().unsqueeze(-1).expand(-1, sps)

#         x_x_mod = self.constpoints[0, 0, indices_x]
#         x_y_mod = self.constpoints[1, 0, indices_y]

#         return torch.stack((x_x_mod, x_y_mod), dim=-1).to(torch.complex64)


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
            # case cls.ARBITRARY:
            #     assert (
            #         constpoints is not None
            #     ), "constpoints must be provided for ARBITRARY modulation"
            #     return ArbitraryModulator(constpoints)
            case _:
                raise ValueError(f"Unknown modulation type: {params.target}")

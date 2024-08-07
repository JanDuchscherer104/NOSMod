from enum import Enum, auto
from typing import Optional, Type, Union

import torch
from pydantic import Field
from torch import Tensor, nn

from ..utils import BaseConfig
from .shared_params import TemporalParams

"""
TODO: change modulator to operate in time domain
"""


class ModulatorParams(BaseConfig[Union["PSKModulator", "AMModulator"]]):
    target: "ModulationType" = Field(default_factory=lambda: ModulationType.PSK)
    num_constellation_points: Optional[int] = 4

    temporal: TemporalParams = Field(default_factory=TemporalParams)


class PSKModulator(nn.Module):
    def __init__(self, params: ModulatorParams):
        super().__init__()
        self.params = params

        # Generate continuous wave (CW) carrier signals for both polarizations
        time_ax = torch.from_numpy(self.params.temporal.time_ax)
        self.carrier = torch.exp(
            1j * 2 * torch.pi * time_ax * self.params.temporal.center_freq
        ).unsqueeze(
            0
        )  # Shape: (1, nos)

    def forward(self, x_xy: Tensor) -> Tensor:
        """
        Apply PSK modulation using a Mach-Zehnder Modulator (MZM) in the time domain.

        Args:
            x_xy (Tensor["B, 2, nos", torch.float32]): Input electrical signals for x and y polarizations.

        Returns:
            Tensor["B, 2, nos * sps", torch.complex64]: Modulated optical signals for x and y polarizations.
        """

        # Combine the carrier signals for both polarizations
        carrier_xy = self.carrier.expand(x_xy.size(0), -1)  # Shape: (B, nos)

        # Modulate the carrier signals with the input x and y polarization signals
        modulated_signal = carrier_xy * torch.stack(
            (
                torch.cos(
                    torch.pi
                    * x_xy[:, 0, :]
                    / self.params.num_constellation_points  # phase_x
                ),
                torch.sin(
                    torch.pi
                    * x_xy[:, 1, :]
                    / self.params.num_constellation_points  # phase_y
                ),
            ),
            dim=1,
        )

        return modulated_signal


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

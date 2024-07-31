from enum import Enum, auto
from typing import Optional, Type, Union

import torch
from pydantic import Field
from torch import Tensor, nn

from ..utils import BaseConfig


class DemodulatorParams(BaseConfig[Union["PSKDemodulator", "AMDemodulator"]]):
    target: "DemodulationType" = Field(default_factory=lambda: DemodulationType.PSK)
    constellation_points: int = 4


class PSKDemodulator(nn.Module):
    def __init__(self, params: DemodulatorParams):
        super().__init__()
        self.params = params

    def forward(self, x: Tensor) -> Tensor:
        """
        Differentiable demodulate a PSK modulated signal.

        Args:
            x (Tensor["B, 2", complex64]): Complex input tensor representing the PSK modulated signal.

        Returns:
            Tensor["B, 2, constellation_points", float32]: Soft demodulated constellation points.
        """
        # Calculate the angles
        angles = torch.angle(x)

        # Calculate the distances to each constellation point
        constellation_angles = (
            2
            * torch.pi
            * torch.arange(self.params.constellation_points)
            / self.params.constellation_points
        )
        distances = (angles.unsqueeze(-1) - constellation_angles).abs()

        # Use softmax to get a differentiable approximation of the closest constellation point
        constellation_points = (-distances).softmax(dim=-1)
        return constellation_points


class AMDemodulator(nn.Module):
    def __init__(self, params: DemodulatorParams):
        super().__init__()
        self.params = params

    def forward(self, x: Tensor) -> Tensor:
        """
        Demodulate an AM modulated signal.

        Args:
            x (Tensor): Input tensor representing the AM modulated signal.

        Returns:
            Tensor: Demodulated symbols.
        """
        levels = torch.linspace(-1, 1, self.params.constellation_points)
        symbols = torch.argmin(torch.abs(x.unsqueeze(-1) - levels), dim=-1)
        return symbols


class DemodulationType(Enum):
    PSK = auto()
    AM = auto()

    @classmethod
    def setup_target(cls, params: DemodulatorParams) -> nn.Module:
        """
        Get the appropriate demodulator based on the specified type.

        Args:
            params (DemodulatorParams): Parameters for demodulation.

        Returns:
            nn.Module: An instance of the appropriate demodulator.
        """
        match params.target:
            case cls.PSK:
                return PSKDemodulator(params)
            case cls.AM:
                return AMDemodulator(params)
            case _:
                raise ValueError(f"Unsupported demodulation type: {params.target}")

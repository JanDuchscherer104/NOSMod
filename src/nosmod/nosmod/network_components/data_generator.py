from enum import Enum, auto
from typing import Optional, Tuple, Type

import torch
from pydantic import Field, field_validator
from torch import Tensor
from torch.utils.data import Dataset

from ..utils import CONSOLE, BaseConfig, Stage
from .shared_params import TemporalParams

"""
TODO: Split in (test, val): do not generate new data for each epoch!
"""


class NosmodDataGeneratorParams(BaseConfig["NosmodDataGenerator"]):
    num_epochs: int
    batch_size: int
    target: Type["NosmodDataGenerator"] = Field(
        default_factory=lambda: NosmodDataGenerator
    )
    split: Stage = Field(default=Stage.TRAIN)
    generate_per_epoch: bool = True
    alphabet: "AlphabetType" = Field(default_factory=lambda: AlphabetType.DIGITAL)
    temporal: TemporalParams = Field(default_factory=TemporalParams)

    alphabet_size: int = 4

    @field_validator("alphabet_size")
    def __val_alphabet_size(cls, v, _):
        if v < 2:
            raise ValueError("alphabet_size must be greater than or equal to 2")
        return v


class DigitalAlphabet(Dataset):
    def __init__(self, params: NosmodDataGeneratorParams):
        self.params = params

    def __len__(self):
        return self.params.batch_size * self.params.num_epochs

    def __getitem__(self, idx) -> Tensor:
        """Generate a batch of digital symbols.

        Args:
            idx (_type_): _description_

        Returns:
            Tensor["B, 2, num_symbols", float32]: Batch of digital symbols.
        """
        num_symbols = self.params.temporal.num_symbols
        # Generate random integer values representing symbols for x and y polarization
        x_xy = (
            torch.stack(
                (
                    torch.randint(
                        0,
                        self.params.alphabet_size,
                        (
                            self.params.batch_size,
                            num_symbols,
                        ),
                    ),
                    torch.randint(
                        0,
                        self.params.alphabet_size,
                        (
                            self.params.batch_size,
                            num_symbols,
                        ),
                    ),
                ),
                dim=0,  # Stack along the first dimension (for x and y)
            )
            .unsqueeze(0)
            .to(torch.float32)
        )  # Add a new batch dimension

        return x_xy


class ContinuousAlphabet(Dataset):
    def __init__(self, params: NosmodDataGeneratorParams):
        self.params = params

    def __len__(self):
        return self.params.num_samples

    def __getitem__(self, idx) -> Tensor:
        """Generate a pair of continuous valued symbols.

        Args:
            idx (int, Tensor[int])

        Returns:
            Tensor["num_symbols, 2", float32]: Continuous valued symbols.
        """
        # Generate continuous valued data in the range [0, alphabet_size)
        x_xy = torch.stack(
            (
                torch.rand(self.params.num_symbols) * self.params.alphabet_size,
                torch.rand(self.params.num_symbols) * self.params.alphabet_size,
            ),
            dim=-1,
        ).to(torch.float32)

        return x_xy


class AlphabetType(Enum):
    DIGITAL = auto()
    CONTINUOUS = auto()

    @classmethod
    def get_dataset(cls, params: NosmodDataGeneratorParams) -> Dataset:
        if params.alphabet == cls.DIGITAL:
            return DigitalAlphabet(params)
        elif params.alphabet == cls.CONTINUOUS:
            return ContinuousAlphabet(params)
        else:
            raise ValueError(f"Unknown alphabet type: {params.alphabet}")


class NosmodDataGenerator(Dataset):
    def __init__(self, params: NosmodDataGeneratorParams):
        self.params = params
        self.dataset = AlphabetType.get_dataset(params)

        CONSOLE.log(
            f"Instantiated {self.__class__.__name__} with {len(self)} samples and alphabet {self.params.alphabet}."
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tensor:
        return self.dataset[idx]

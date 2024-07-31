from enum import Enum, auto
from typing import Tuple, Type

import torch
from pydantic import Field, field_validator
from torch import Tensor
from torch.utils.data import Dataset

from ..utils import CONSOLE, BaseConfig, Stage

"""
TODO: Split in (test, val): do not generate new data for each epoch!
"""


class NosmodDataGeneratorParams(BaseConfig["NosmodDataGenerator"]):
    target: Type["NosmodDataGenerator"] = Field(
        default_factory=lambda: NosmodDataGenerator
    )
    split: Stage = Field(default=Stage.TRAIN)
    alphabet: "AlphabetType" = Field(default_factory=lambda: AlphabetType.DIGITAL)
    num_symbols: int = 128
    num_samples: int = 1000
    samples_per_symbol: int = 32
    alphabet_size: int = 4
    generate_per_epoch: bool = True

    @field_validator("alphabet_size")
    def __val_alphabet_size(cls, v, _):
        if v < 2:
            raise ValueError("alphabet_size must be greater than or equal to 2")
        return v


class DigitalAlphabet(Dataset):
    def __init__(self, params: NosmodDataGeneratorParams):
        self.params = params

    def __len__(self):
        return self.params.num_samples

    def __getitem__(self, idx) -> Tensor:
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            Tensor["num_symbols, 2", float32]: _description_
        """
        x_xy = torch.stack(
            (
                torch.randint(0, self.params.alphabet_size, (self.params.num_symbols,)),
                torch.randint(0, self.params.alphabet_size, (self.params.num_symbols,)),
            ),
            dim=-1,
        ).to(torch.float32)

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

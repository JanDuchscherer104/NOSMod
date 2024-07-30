from enum import Enum, auto
from typing import Tuple

import torch
from pydantic import Field, field_validator
from torch import Tensor
from torch.utils.data import Dataset

from ..utils import CONSOLE, BaseConfig, Stage

"""
TODO: Split in (test, val): do not generate new data for each epoch!
"""


class NosmodDatasetParams(BaseConfig):
    split: Stage
    alphabet: "AlphabetType"
    num_samples: int = 1000
    num_symbols: int = 128
    sps: int = 32
    alphabet_size: int = 4
    generate_per_epoch: bool = True

    @field_validator("alphabet_size")
    def __val_alphabet_size(cls, v, _):
        if v < 2:
            raise ValueError("alphabet_size must be greater than or equal to 2")
        return v


class DigitalAlphabet(Dataset):
    def __init__(self, params: NosmodDatasetParams):
        self.params = params

    def __len__(self):
        return self.params.num_samples

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        # Generate random symbols in the range [0, alphabet_size)
        x_x = torch.randint(0, self.params.alphabet_size, (self.params.num_symbols,))
        x_y = torch.randint(0, self.params.alphabet_size, (self.params.num_symbols,))
        return x_x, x_y


class ContinuousAlphabet(Dataset):
    def __init__(self, params: NosmodDatasetParams):
        self.params = params

    def __len__(self):
        return self.params.num_samples

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        """Generate a pair of continuous valued symbols.

        Args:
            idx (int, Tensor[int])

        Returns:
            Tuple[Tensor, Tensor]: Tuple of two tensors containing the x and y symbols.
        """
        # Generate continuous valued data in the range [0, alphabet_size)
        x_x = torch.rand(self.params.num_symbols) * self.params.alphabet_size
        x_y = torch.rand(self.params.num_symbols) * self.params.alphabet_size
        return x_x, x_y


class AlphabetType(Enum):
    DIGITAL = auto()
    CONTINUOUS = auto()

    @classmethod
    def get_dataset(cls, params: NosmodDatasetParams) -> Dataset:
        if params.alphabet == cls.DIGITAL:
            return DigitalAlphabet(params)
        elif params.alphabet == cls.CONTINUOUS:
            return ContinuousAlphabet(params)
        else:
            raise ValueError(f"Unknown alphabet type: {params.alphabet}")


class NosmodDataset(Dataset):
    def __init__(self, params: NosmodDatasetParams):
        self.params = params
        self.dataset = AlphabetType.get_dataset(params)

        CONSOLE.log(
            f"Instantiated {self.__class__.__name__} with {len(self)} samples and alphabet {self.params.alphabet}."
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.dataset[idx]  # type: ignore

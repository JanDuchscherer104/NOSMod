from typing import Optional, Type

import torch
from pydantic import Field
from torch import Tensor, nn

from ..utils import BaseConfig


class SymbolPredictorParams(BaseConfig["SymbolPredictor"]):
    target: Type["SymbolPredictor"] = Field(default_factory=lambda: SymbolPredictor)
    alphabet_size: Optional[int] = None


class SymbolPredictor(nn.Module):
    def __init__(self, alphabet_size: int):
        super().__init__()
        self.alphabet_size = alphabet_size

    def forward(self, soft_symbols: Tensor) -> Tensor:
        """
        Predict the original symbols from the soft demodulated symbols.

        Args:
            soft_symbols (Tensor["B, alphabet_size", float32]): Soft demodulated symbols.

        Returns:
            Tensor["B, alphabet_size", float32]: Predicted symbols in the original alphabet space.
        """
        predicted_symbols = soft_symbols @ torch.arange(
            self.alphabet_size, dtype=torch.float64
        )
        return predicted_symbols

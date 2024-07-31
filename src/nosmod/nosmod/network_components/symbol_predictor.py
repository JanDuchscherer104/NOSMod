import torch
from torch import Tensor, nn


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
        predicted_symbols = soft_symbols @ torch.arange(self.alphabet_size).float()
        return predicted_symbols

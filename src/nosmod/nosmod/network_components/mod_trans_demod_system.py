from typing import Type

from pydantic import Field
from torch import Tensor, nn

from ..utils import CONSOLE, BaseConfig
from .data_generator import NosmodDataGeneratorParams
from .demodulator import DemodulatorParams
from .modulator import ModulatorParams
from .raised_cos_filter import RaisedCosParams
from .symbol_predictor import SymbolPredictor


class NosmodSystemParams(BaseConfig["NosmodSystem"]):
    target: Type["NosmodSystem"] = Field(default_factory=lambda: NosmodSystem)

    mod_params: ModulatorParams = Field(default_factory=ModulatorParams)
    raised_cos_params: RaisedCosParams = Field(default_factory=RaisedCosParams)
    demod_params: DemodulatorParams = Field(default_factory=DemodulatorParams)
    data_gen_params: NosmodDataGeneratorParams = Field(
        default_factory=NosmodDataGeneratorParams
    )


class NosmodSystem(nn.Module):
    def __init__(
        self,
        params: NosmodSystemParams,
    ):
        super().__init__()
        self.params = params

        self.modulator = self.params.mod_params.setup_target()
        self.channel = self.params.raised_cos_params.setup_target()
        self.demodulator = self.params.demod_params.setup_target()
        self.symbol_predictor = SymbolPredictor(
            self.params.data_gen_params.alphabet_size
        )

        CONSOLE.log(f"Initialized {self.__class__.__name__}")

    def forward(self, x_xy: Tensor) -> Tensor:
        """
        Forward pass through the system.

        Args:
            x_xy (Tensor["B, 2", float32]): Input tensor representing the symbols.

        Returns:
            Tensor["B, 2, alphabet_size", float32]: Soft predicted symbols.
        """

        modulated_signal = self.modulator(x_xy)
        transmitted_signal = self.channel(modulated_signal)
        soft_symbols = self.demodulator(transmitted_signal)
        predicted_symbols = self.symbol_predictor(soft_symbols)

        return predicted_symbols

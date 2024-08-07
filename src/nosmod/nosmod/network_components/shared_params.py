from typing import Type

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from ..utils import _SharedParams


class TemporalParams(_SharedParams):
    # target: Type["SharedParams"] = Field(default_factory=lambda: SharedParams)
    symbol_rate: float = 1e4
    """[symbol_rate] = Hz"""

    center_freq: float = 1e3
    """[center_freq] = Hz"""

    samples_per_symbol: int = 1
    """The number of samples per symbol period."""

    num_symbols: int = int(1e3)
    """The total number of symbols in the transmission."""

    @property
    def freq_range(self) -> float:
        """
        Calculates the frequency range for the system based on the symbol rate
        and the number of samples per symbol.

        Returns:
            float: Frequency range in Hz.
        """
        return self.samples_per_symbol * self.symbol_rate

    @property
    def sample_time(self) -> float:
        """
        float: Sample time in seconds (time duration of a single sample).
        """
        return 1.0 / (self.symbol_rate * self.samples_per_symbol)

    @property
    def freq_res(self) -> float:
        """
        float: Frequency resolution in Hz.
        """
        return self.symbol_rate / self.num_symbols

    @property
    def freq_ax(self) -> NDArray[np.float64]:
        """
        NDArray["sps*nos,", float64]: The frequency axis in Hz.
        """
        return (
            np.arange(-self.freq_range / 2.0, self.freq_range / 2.0, self.freq_res)
            + self.center_freq
        )

    @property
    def time_ax(self) -> NDArray[np.float64]:
        """
        NDArray["sps*nos,", float64]: The time axis in seconds.
        """
        return (
            np.linspace(
                self.sample_time,
                1.0 / self.symbol_rate * self.num_symbols,
                self.samples_per_symbol * self.num_symbols,
            )
            - self.sample_time
        )

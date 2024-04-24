from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pydantic import BaseModel, Field, field_validator, model_validator


class RaisedCosineParameters(BaseModel):
    fc: float = Field(..., description="Center frequency in Hz")
    B: float = Field(..., description="Bandwidth in Hz", gt=0)
    alpha: float = Field(
        ..., description="Roll-off factor, 0 <= alpha <= 1", ge=0, le=1
    )
    fs: float = Field(..., description="Sampling frequency in Hz")
    nos: int = Field(128, description="Number of symbols")
    sps: int = Field(32, description="Samples per symbol")
    N: int = Field(init=False, description="Number of samples")
    attenuation_db: float = Field(0, description="Attenuation in dBm")

    @model_validator(mode="after")
    def __set_N(self) -> "RaisedCosineParameters":
        self.N = self.nos * self.sps

        assert (self.N) & (self.N - 1) == 0, "N must be a power of 2"

        return self


class RaisedCosineFilter:
    def __init__(self, params: RaisedCosineParameters):
        self.params = params
        self.frequency_response = self.calculate_frequency_response()
        self.impulse_response = self.calculate_impulse_response()

    def calculate_frequency_response(self) -> torch.Tensor:
        """_summary_

        Returns:
            torch.Tensor: _description_
        """
        f = torch.linspace(
            -self.params.fs / 2, self.params.fs / 2, self.params.N, dtype=torch.float64
        )

        freq_response = torch.zeros_like(f)

        f_shifted = torch.abs(f - self.params.fc)

        # Define the response based on conditions
        # Passband
        cond1 = f_shifted <= self.params.fs * (1 - self.params.alpha) / 2
        freq_response[cond1] = 1

        # Transition band
        cond2 = (f_shifted > self.params.fs * (1 - self.params.alpha) / 2) & (
            f_shifted <= self.params.fs * (1 + self.params.alpha) / 2
        )
        freq_response[cond2] = 0.5 * (
            1
            + torch.cos(
                np.pi
                / (self.params.fs * self.params.alpha)
                * (f_shifted[cond2] - self.params.fs * (1 - self.params.alpha) / 2)
            )
        )

        return freq_response * 10 ** (-self.params.attenuation_db / 10)

    def calculate_impulse_response(self):
        # Shift the zero-frequency component to the start of the array
        freq_response_shifted = torch.fft.ifftshift(self.frequency_response)

        # Calculate the impulse response by IFFT of the frequency response
        h = torch.fft.ifft(freq_response_shifted)
        h_centered = h / len(self.frequency_response)
        return h_centered

    def plot_frequency_response(
        self,
        input_signal: Optional[torch.Tensor] = None,
        filtered_signal: Optional[torch.Tensor] = None,
    ):
        f = np.linspace(-self.params.fs / 2, self.params.fs / 2, self.params.N)

        if input_signal is not None and filtered_signal is not None:
            input_signal_fft = torch.fft.fftshift(torch.fft.fft(input_signal))
            filtered_signal_fft = torch.fft.fftshift(torch.fft.fft(filtered_signal))
            magnitude_input = np.abs(input_signal_fft.numpy())
            magnitude_filtered = np.abs(filtered_signal_fft.numpy())
            title = "Frequency Response of Input and Filtered Signals"
        else:
            magnitude = torch.abs(self.frequency_response).numpy()
            title = "Frequency Response of Raised Cosine Filter"

        plt.figure(figsize=(10, 5))
        sns.set_style("whitegrid")

        if input_signal is not None and filtered_signal is not None:
            sns.lineplot(x=f, y=magnitude_input, label="Input Signal")
            sns.lineplot(x=f, y=magnitude_filtered, label="Filtered Signal")
        else:
            sns.lineplot(x=f, y=magnitude)

        plt.title(title)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")

        sns.despine()
        plt.tight_layout()
        plt.show()

    def plot_impulse_response(
        self,
        input_signal: Optional[torch.Tensor] = None,
        filtered_signal: Optional[torch.Tensor] = None,
        x_lim: Optional[float] = None,
    ):
        # t = np.arange(-self.params.N // 2, self.params.N // 2)
        t = np.linspace(
            1 / (-2 * self.params.fs), 1 / (2 * self.params.fs), self.params.N
        )

        if input_signal is not None and filtered_signal is not None:
            input_signal_np = input_signal.numpy().real
            filtered_signal_np = filtered_signal.numpy().real
            title = "Time Response of Input and Filtered Signals"
        else:
            impulse = self.impulse_response.numpy().real
            title = "Impulse Response of Raised Cosine Filter"

        plt.figure(figsize=(10, 5))
        sns.set_style("whitegrid")

        if input_signal is not None and filtered_signal is not None:
            sns.lineplot(x=t, y=input_signal_np, label="Input Signal")
            sns.lineplot(x=t, y=filtered_signal_np, label="Filtered Signal")
        else:
            sns.lineplot(x=t, y=impulse)

        plt.title(title)
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")

        if x_lim:
            plt.xlim(-x_lim, x_lim)

        sns.despine()
        plt.show()

    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        """Applies the filter in the frequency domain

        Args:
            signal (torch.Tensor): An input signal of shape (N,) or (N, 2)

        Returns:
            torch.Tensor: The filtered signal
        """
        signal_fft = torch.fft.fft(signal)
        filtered_signal_fft = signal_fft * self.frequency_response
        filtered_signal = torch.fft.ifft(filtered_signal_fft) / self.params.N
        return filtered_signal

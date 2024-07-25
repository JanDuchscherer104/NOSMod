from typing import Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pydantic import Field, ValidationInfo, field_validator, model_validator
from torch import Tensor, nn
from typing_extensions import Annotated

from ..utils import BaseConfig


class RaisedCosineParams(BaseConfig["RaisedCosineFilter"]):
    target: Annotated[
        Type["RaisedCosineFilter"],
        Field(..., default_factory=lambda: RaisedCosineFilter),
    ]
    center_freq: float = Field(..., description="Center frequency in Hz", gt=0)
    rolloff_fact: float = Field(
        None, description="Roll-off factor, 0 <= alpha <= 1", ge=0, le=1
    )
    sampling_freq: float = Field(..., description="Sampling frequency in Hz")
    nos: int = Field(128, description="Number of symbols")
    sps: int = Field(32, description="Samples per symbol")
    attenuation_db: float = Field(0, description="Attenuation in dBm")
    num_samples: int = Field(None, description="Number of samples, N=SPS*NOS")
    bandwidth: Annotated[
        float, Field(None, description="Bandwidth in Hz, B=fs/2*(1+rolloff)")
    ]

    @field_validator("bandwidth", "rolloff_fact", mode="before")
    def __validate_bandwidth_and_rolloff(
        cls, v: Optional[float], info: ValidationInfo
    ) -> float:
        values = info.data
        if info.field_name == "bandwidth":
            if v is None and values.get("rolloff_fact") is None:
                raise ValueError("Either 'bandwidth' or 'rolloff_fact' must be set.")
            if v is None:
                v = values["sampling_freq"] / 2 * (1 + values["rolloff_fact"])
        elif info.field_name == "rolloff_fact":
            if v is None and values.get("bandwidth") is None:
                raise ValueError("Either 'bandwidth' or 'rolloff_fact' must be set.")
            if v is None:
                v = 2 * values["bandwidth"] / values["sampling_freq"] - 1
        return v  # type: ignore

    @field_validator("num_samples", "nos", "sps", mode="before")
    def __validate_num_samples(cls, v: Optional[int], info: ValidationInfo) -> int:
        values = info.data
        if v is None:
            v = values["nos"] * values["sps"]
        assert (v) & (v - 1) == 0, "N must be a power of 2"
        return v


class RaisedCosineFilter(nn.Module):
    params: RaisedCosineParams
    frequency_response: Optional[Tensor]
    impulse_response: Optional[Tensor]

    def __init__(self, params: RaisedCosineParams):
        super().__init__()

        self.params = params
        self.frequency_response = self.calculate_frequency_response()
        self.impulse_response = self.calculate_impulse_response()

    def forward(self, x: Tensor) -> Tensor:
        signal_fft = torch.fft.fft(x)
        filtered_signal_fft = signal_fft * self.frequency_response
        filtered_signal = torch.fft.ifft(filtered_signal_fft) / self.params.num_samples
        return filtered_signal

    def calculate_frequency_response(self) -> Tensor:
        f = torch.linspace(
            -self.params.sampling_freq / 2,
            self.params.sampling_freq / 2,
            self.params.num_samples,
            dtype=torch.float64,
        )
        freq_response = torch.zeros_like(f)
        f_shifted = torch.abs(f - self.params.center_freq)

        # Define the response based on the filters behavior in different frequency ranges
        # Passband
        cond1 = (
            f_shifted <= self.params.sampling_freq * (1 - self.params.rolloff_fact) / 2
        )
        freq_response[cond1] = 1

        # Transition band
        cond2 = (
            f_shifted > self.params.sampling_freq * (1 - self.params.rolloff_fact) / 2
        ) & (
            f_shifted <= self.params.sampling_freq * (1 + self.params.rolloff_fact) / 2
        )
        freq_response[cond2] = 0.5 * (
            1
            + torch.cos(
                np.pi
                / (self.params.sampling_freq * self.params.rolloff_fact)
                * (
                    f_shifted[cond2]
                    - self.params.sampling_freq * (1 - self.params.rolloff_fact) / 2
                )
            )
        )

        return freq_response * 10 ** (-self.params.attenuation_db / 10)

    def calculate_impulse_response(self):
        assert self.frequency_response is not None
        freq_response_shifted = torch.fft.ifftshift(self.frequency_response)
        h = torch.fft.ifft(freq_response_shifted)
        h_centered = h / len(self.frequency_response)
        return h_centered

    def plot_frequency_response(
        self,
        input_signal: Optional[Tensor] = None,
        filtered_signal: Optional[Tensor] = None,
    ):
        f = np.linspace(
            -0.5,
            0.5,
            self.params.num_samples,
        )  # Normalized frequency axis
        plt.figure(figsize=(10, 5))
        sns.set_style("whitegrid")

        if input_signal is not None and filtered_signal is not None:
            input_signal_fft = torch.fft.fftshift(torch.fft.fft(input_signal))
            filtered_signal_fft = torch.fft.fftshift(torch.fft.fft(filtered_signal))
            magnitude_input = np.abs(input_signal_fft.numpy())
            magnitude_filtered = np.abs(filtered_signal_fft.numpy())
            sns.lineplot(x=f, y=magnitude_input, label="Input Signal")
            sns.lineplot(x=f, y=magnitude_filtered, label="Filtered Signal")
            title = "Frequency Response of Input and Filtered Signals"
        else:
            magnitude = torch.abs(self.frequency_response).numpy()
            sns.lineplot(x=f, y=magnitude)
            title = "Frequency Response of Raised Cosine Filter"

        plt.title(title)
        plt.xlabel("Normalized Frequency (f / f_s)")
        plt.ylabel("Magnitude")
        sns.despine()
        plt.tight_layout()
        plt.show()

    def plot_impulse_response(
        self,
        input_signal: Optional[Tensor] = None,
        filtered_signal: Optional[Tensor] = None,
        x_lim: Optional[float] = None,
    ):
        t = (
            np.linspace(
                -self.params.num_samples // 2,
                self.params.num_samples // 2,
                self.params.num_samples,
            )
            / self.params.sampling_freq
        ) / (
            1 / self.params.sps
        )  # Normalized time axis
        plt.figure(figsize=(10, 5))
        sns.set_style("whitegrid")

        if input_signal is not None and filtered_signal is not None:
            input_signal_np = input_signal.numpy().real
            filtered_signal_np = filtered_signal.numpy().real
            sns.lineplot(x=t, y=input_signal_np, label="Input Signal")
            sns.lineplot(x=t, y=filtered_signal_np, label="Filtered Signal")
            title = "Time Response of Input and Filtered Signals"
        else:
            impulse = torch.fft.fftshift(self.impulse_response).numpy().real
            sns.lineplot(x=t, y=impulse)
            title = "Impulse Response of Raised Cosine Filter"

        plt.title(title)
        plt.xlabel("Normalized Time (t / T)")
        plt.ylabel("Amplitude")

        if x_lim:
            plt.xlim(-x_lim, x_lim)

        sns.despine()
        plt.tight_layout()
        plt.show()

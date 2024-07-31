from typing import Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pydantic import Field, ValidationInfo, field_validator, model_validator
from torch import Tensor, nn
from typing_extensions import Annotated

from ..utils import CONSOLE, BaseConfig


class RaisedCosParams(BaseConfig["RaisedCosFilter"]):
    """
    Configuration parameters for the Raised Cosine Filter.
    """

    target: Type["RaisedCosFilter"] = Field(default_factory=lambda: RaisedCosFilter)
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
    input_is_time_domain: bool = Field(False, description="Input is in time domain")
    conv_in_time_domain: bool = Field(
        False, description="Conv(impulse_response, signal) if input_is_time_domain=True"
    )

    @field_validator("bandwidth", "rolloff_fact", mode="before")
    def __validate_bandwidth_and_rolloff(
        cls, v: Optional[float], info: ValidationInfo
    ) -> float:
        """
        Validate and calculate the bandwidth and roll-off factor.

        Args:
            v (Optional[float]): Value to validate.
            info (ValidationInfo): Information about the validation context.

        Returns:
            float: Validated and calculated value.
        """
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
        """
        Validate and calculate the number of samples.

        Args:
            v (Optional[int]): Value to validate.
            info (ValidationInfo): Information about the validation context.

        Returns:
            int: Validated and calculated value.
        """
        values = info.data
        if v is None:
            v = values["nos"] * values["sps"]
        assert (v) & (v - 1) == 0, "N must be a power of 2"
        return v


class RaisedCosFilter(nn.Module):
    """
    Raised Cosine Filter implemented as a PyTorch module.
    """

    params: RaisedCosParams
    frequency_response: Tensor
    impulse_response: Optional[Tensor]

    def __init__(self, params: RaisedCosParams):
        """
        Initialize the Raised Cosine Filter.

        Args:
            params (RaisedCosineParams): Configuration parameters for the filter.
        """
        super().__init__()

        self.params = params
        self.frequency_response = self.calculate_frequency_response()
        self.impulse_response = self.calculate_impulse_response()
        CONSOLE.print(f"Instantiated {self.__class__.__name__} with params: {params}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the filter to the input signal based on the specified parameters.

        Args:
            x (Tensor): Input signal.

        Returns:
            Tensor: Filtered signal.
        """
        if self.params.input_is_time_domain:
            if self.params.conv_in_time_domain:
                return self.apply_in_time_domain(x)
            else:
                x_freq = torch.fft.fft(x)
                filtered_x_freq = self.apply_in_frequency_domain(x_freq)
                return torch.fft.ifft(filtered_x_freq)
        else:
            return self.apply_in_frequency_domain(x)

    def apply_in_time_domain(self, signal: Tensor) -> Tensor:
        """
        Apply the raised cosine filter in the time domain by convolving the input signal with the impulse response.

        Args:
            signal (Tensor['N, complex128']): Input signal.

        Returns:
            Tensor['N, complex128']: Filtered signal.
        """
        filtered_signal = torch.nn.functional.conv1d(
            signal.view(1, 1, -1), self.impulse_response.view(1, 1, -1), padding="same"
        )
        return filtered_signal.view(-1)

    def apply_in_frequency_domain(self, signal: Tensor) -> Tensor:
        """
        Apply the raised cosine filter in the frequency domain by multiplying the Fourier transform of the input signal with the frequency response.

        Args:
            signal (Tensor['N, complex128']): Input signal.

        Returns:
            Tensor['N, complex128']: Filtered signal.
        """
        filtered_signal_fft = signal * self.frequency_response
        return filtered_signal_fft

    def calculate_frequency_response(self) -> Tensor:
        """
        Returns:
            Tensor['N, float64']: Frequency response of the filter.
        """
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

    def calculate_impulse_response(self) -> Tensor:
        """
        Returns:
            Tensor['N, complex64']: Impulse response of the filter.
        """
        assert self.frequency_response is not None
        freq_response_shifted = torch.fft.ifftshift(self.frequency_response)
        h = torch.fft.ifft(freq_response_shifted)
        h_centered = h / len(self.frequency_response)
        return h_centered

    def plot_frequency_response(
        self,
        input_signal: Optional[Tensor] = None,
        filtered_signal: Optional[Tensor] = None,
        x_lim: Optional[float] = None,
        is_norm_xaxis: bool = False,
        title: Optional[str] = None,
    ):
        """
        Args:
            input_signal (Optional[Tensor['N, complex64']]): Optional input signal to plot.
            filtered_signal (Optional[Tensor['N, complex64']]): Optional filtered signal to plot.
            x_lim (Optional[float]): Limit for the x-axis.
            is_norm_xaxis (bool): Whether to normalize the x-axis.
            title (Optional[str]): Title for the plot.
        """
        f = (
            np.linspace(
                -self.params.sampling_freq / 2,
                self.params.sampling_freq / 2,
                self.params.num_samples,
            )
            if not is_norm_xaxis
            else np.linspace(-0.5, 0.5, self.params.num_samples)
        )
        plt.figure(figsize=(10, 5))
        sns.set_style("whitegrid")

        if input_signal is not None and filtered_signal is not None:
            input_signal_fft = torch.fft.fftshift(torch.fft.fft(input_signal))
            filtered_signal_fft = torch.fft.fftshift(torch.fft.fft(filtered_signal))
            magnitude_input = np.abs(input_signal_fft.numpy())
            magnitude_filtered = np.abs(filtered_signal_fft.numpy())
            sns.lineplot(x=f, y=magnitude_input, label="Input Signal")
            sns.lineplot(x=f, y=magnitude_filtered, label="Filtered Signal")
            title = title or "Frequency Response of Input and Filtered Signals"
        else:
            magnitude = torch.abs(self.frequency_response).numpy()
            sns.lineplot(x=f, y=magnitude)
            title = title or "Frequency Response of Raised Cosine Filter"

        plt.title(title)
        plt.xlabel(
            "Normalized Frequency (f / f_s)" if is_norm_xaxis else "Frequency (Hz)"
        )
        plt.ylabel("Magnitude")

        if x_lim:
            plt.xlim(-x_lim, x_lim)

        sns.despine()
        plt.tight_layout()
        plt.show()

    def plot_impulse_response(
        self,
        input_signal: Optional[Tensor] = None,
        filtered_signal: Optional[Tensor] = None,
        x_lim: Optional[float] = None,
        is_norm_xaxis: bool = False,
        title: Optional[str] = None,
    ):
        """
        Plot the impulse response of the Raised Cosine Filter.

        Args:
            input_signal (Optional[Tensor['N, complex64']]): Optional input signal to plot.
            filtered_signal (Optional[Tensor['N, complex64']]): Optional filtered signal to plot.
            x_lim (Optional[float]): Limit for the x-axis.
            is_norm_xaxis (bool): Whether to normalize the x-axis.
            title (Optional[str]): Title for the plot.
        """
        t = (
            np.linspace(
                -4 / self.params.sampling_freq,
                4 / self.params.sampling_freq,
                self.params.num_samples,
            )
            if not is_norm_xaxis
            else np.linspace(
                -self.params.num_samples // 2,
                self.params.num_samples // 2,
                self.params.num_samples,
            )
            / self.params.sps
        )
        plt.figure(figsize=(10, 5))
        sns.set_style("whitegrid")

        if input_signal is not None and filtered_signal is not None:
            input_signal_np = np.abs(input_signal.numpy())
            filtered_signal_np = np.abs(filtered_signal.numpy())
            sns.lineplot(x=t, y=input_signal_np, label="Input Signal")
            sns.lineplot(x=t, y=filtered_signal_np, label="Filtered Signal")
            title = title or "Time Response of Input and Filtered Signals"
        else:
            impulse = np.abs(torch.fft.fftshift(self.impulse_response).numpy())
            sns.lineplot(x=t, y=impulse)
            title = title or "Impulse Response of Raised Cosine Filter"

        plt.title(title)
        plt.xlabel("Normalized Time (t / T)" if is_norm_xaxis else "Time (s)")
        plt.ylabel("Amplitude")

        if x_lim:
            plt.xlim(-x_lim, x_lim)

        sns.despine()
        plt.tight_layout()
        plt.show()

import unittest

import numpy as np
import torch
from torch.autograd import gradcheck

from nosmod import RaisedCosParams
from nosmod.utils import CONSOLE


class TestRaisedCosineFilter(unittest.TestCase):

    def setUp(self):
        self.params = RaisedCosParams(
            center_freq=1e4,
            rolloff_fact=0.5,
            sampling_freq=1e6,
            nos=128,
            sps=32,
            attenuation_db=0,
        )
        self.filter = self.params.setup_target()

    def test_differentiability(self):
        """
        Check if the filter is differentiable.
        """
        signal = torch.randn(
            self.params.num_samples, dtype=torch.complex64, requires_grad=True
        )
        test = gradcheck(self.filter, (signal,), eps=1e-6, atol=1e-4)
        self.assertTrue(test, "RaisedCosineFilter is not differentiable")

    def test_magnitude_at_center_frequency(self):
        """
        Check if the magnitude response at the center frequency is close to the expected value.
        """
        f = torch.linspace(
            -self.params.sampling_freq / 2,
            self.params.sampling_freq / 2,
            self.params.num_samples,
        )
        freq_response = self.filter.calculate_frequency_response()
        center_freq_idx = (torch.abs(f - self.params.center_freq)).argmin()
        expected_magnitude = 1
        self.assertAlmostEqual(
            freq_response[center_freq_idx].item(),
            expected_magnitude,
            places=5,
            msg="Magnitude at center frequency is incorrect",
        )

    def test_magnitude_at_transition_band(self):
        """
        Check if the magnitude response in the transition band is within the expected range.
        """
        f = torch.linspace(
            -self.params.sampling_freq / 2,
            self.params.sampling_freq / 2,
            self.params.num_samples,
        )
        freq_response = self.filter.calculate_frequency_response()
        transition_band_start_idx = (
            torch.abs(
                f
                - (
                    self.params.center_freq
                    + (self.params.sampling_freq * (1 - self.params.rolloff_fact) / 2)
                )
            )
        ).argmin()
        transition_band_end_idx = (
            torch.abs(
                f
                - (
                    self.params.center_freq
                    + (self.params.sampling_freq * (1 + self.params.rolloff_fact) / 2)
                )
            )
        ).argmin()
        transition_magnitude = (
            freq_response[transition_band_start_idx:transition_band_end_idx]
            .mean()
            .item()
        )
        self.assertTrue(
            0 < transition_magnitude < 1,
            "Magnitude in the transition band is incorrect",
        )
        CONSOLE.print(
            f"Transition band start index: {transition_band_start_idx}"
            f"Transition band end index: {transition_band_end_idx}"
            f"Transition band stop frequency: {f[transition_band_end_idx].item()}"
            f"Transition band start frequency: {f[transition_band_start_idx].item()}"
        )
        CONSOLE.print(f"Transition magnitude: {transition_magnitude}")


if __name__ == "__main__":
    unittest.main()

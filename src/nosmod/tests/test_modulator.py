import unittest

import matplotlib.pyplot as plt
import torch

from nosmod.network_components.modulator import (  # ArbitraryModulator,
    AMModulator,
    ModulationType,
    ModulatorParams,
    PSKModulator,
)


class TestModulator(unittest.TestCase):
    def setUp(self):
        self.params_psk = ModulatorParams(
            num_constellation_points=4, target=ModulationType.PSK
        )
        self.params_am = ModulatorParams(
            num_constellation_points=4, target=ModulationType.AM
        )
        self.constpoints = torch.tensor(
            [[[1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]]], dtype=torch.complex64
        )
        # self.params_arbitrary = ModulatorParams(
        #     num_constellation_points=4, target=ModulationType.ARBITRARY
        # )

    def test_psk_modulator(self):
        modulator = PSKModulator(self.params_psk)
        x_x = torch.tensor([0, 1, 2, 3])
        x_y = torch.tensor([0, 1, 2, 3])
        output = modulator(x_x, x_y)

        # Check amplitude
        amplitude = torch.abs(output)
        self.assertTrue(torch.allclose(amplitude, torch.ones_like(amplitude)))

        # Check phase
        expected_phase = torch.exp(
            1j * 2 * torch.pi * x_x / self.params_psk.num_constellation_points
        )
        self.assertTrue(torch.allclose(output[:, 0], expected_phase))

        # Plot constellation
        plt.scatter(output[:, 0].real, output[:, 0].imag, label="PSK Modulator")
        plt.title("PSK Modulator Constellation Diagram")
        plt.xlabel("In-phase")
        plt.ylabel("Quadrature")
        plt.grid(True)
        plt.legend()
        plt.show()

    def test_am_modulator(self):
        modulator = AMModulator(self.params_am)
        x_x = torch.tensor([0, 1, 2, 3])
        x_y = torch.tensor([0, 1, 2, 3])
        output = modulator(x_x, x_y)

        # Check amplitude levels
        levels = torch.linspace(-1, 1, self.params_am.num_constellation_points)
        expected_output = torch.stack(
            (levels[x_x.long()], levels[x_y.long()]), dim=-1
        ).to(torch.complex64)
        self.assertTrue(torch.allclose(output, expected_output))

        # Plot constellation
        plt.scatter(output[:, 0].real, output[:, 0].imag, label="AM Modulator")
        plt.title("AM Modulator Constellation Diagram")
        plt.xlabel("In-phase")
        plt.ylabel("Quadrature")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    unittest.main()

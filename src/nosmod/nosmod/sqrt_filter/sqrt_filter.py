import copy
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from pypho import (
    ModulatedLaser,
    OpticalFilter,
    PyphoSetup,
    SignalSource,
    SymbolGenerator,
)

from ..config import Config, SqrtFilterParams


def generate_sqrt_filter_train_data(
    config: Config, sqrt_filter_params: SqrtFilterParams
) -> Config:
    # Define network elements
    setup = PyphoSetup(
        nos=sqrt_filter_params.nos,
        sps=sqrt_filter_params.sps,
        symbolrate=sqrt_filter_params.symbolrate,
    )
    symbol_generator = SymbolGenerator(glova=setup, nos=setup.nos, pattern="random")
    signal_source = SignalSource(glova=setup, pulseshape="gauss_rz", fwhm=0.33)
    laser = ModulatedLaser(glova=setup, power=0, Df=0, teta=0)
    optical_filter = OpticalFilter(
        glova=setup, Df=0, B=20, filtype="cosrolloff", alpha=0.5
    )
    # modulator = ArbitraryModulator(glova=setup)

    # # Simulation
    # # 4-QAM
    # constellation_points_4qam = [
    #     ([1.0 * np.exp(2.0j * np.pi * x / 4.0) for x in range(0, 4)]),
    #     ([(0), (0)], [(0), (1)], [(1), (1)], [(1), (0)]),
    # ]

    one_bit_sequence = symbol_generator(pattern="ones")
    electrical_signal = signal_source(
        bitsequence=one_bit_sequence
    )  # Generate the electrical signal
    transmitted_signal = laser(
        esig=electrical_signal
    )  # Generate the transmitted signal
    E = copy.deepcopy(transmitted_signal)  # Copy the transmitted signal
    filtered_seq = np.zeros(
        (2, 2, setup.nos * setup.sps)
    )  # Initialize the output array
    input_seq = np.zeros((2, 2, setup.nos * setup.sps))  # Initialize the input array

    df = pd.DataFrame(columns=["E_in", "E_out"])
    data: Dict[str, List] = {
        "E_in": [],  # np.ndarray[float32, (2, 2, setup.nos * setup.sps)] [X/Y, I/Q, # symbols]
        "E_out": [],  # np.ndarray[float32, (2, 2, setup.nos * setup.sps)] [X/Y, I/Q, # symbols]
    }

    for _ in tqdm(range(sqrt_filter_params.num_samples)):
        # Generate random complex numbers for the input signal
        E[0]["E"][0] = np.random.uniform(
            -0.5, 0.5, setup.sps * setup.nos
        ) + 1j * np.random.uniform(-0.5, 0.5, setup.sps * setup.nos)
        E[0]["E"][1] = np.random.uniform(
            -0.5, 0.5, setup.sps * setup.nos
        ) + 1j * np.random.uniform(-0.5, 0.5, setup.sps * setup.nos)

        input_seq = np.stack(
            [
                [np.real(E[0]["E"][0]), np.imag(E[0]["E"][0])],
                [np.real(E[0]["E"][1]), np.imag(E[0]["E"][1])],
            ]
        )

        E = optical_filter(E)

        filtered_seq = np.stack(
            [
                [np.real(E[0]["E"][0]), np.imag(E[0]["E"][0])],
                [np.real(E[0]["E"][1]), np.imag(E[0]["E"][1])],
            ]
        )

        data["E_in"].append(input_seq)
        data["E_out"].append(filtered_seq)

    df = pd.DataFrame(data)
    df.to_pickle(config.paths.sqrt_filter_file)

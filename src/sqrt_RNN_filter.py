import copy
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from pypho import (
    ArbitraryModulator,
    ModulatedLaser,
    OpticalFilter,
    PyphoSetup,
    SignalSource,
    SymbolGenerator,
)


def generate_sqrt_filter_train_data(config: Config) -> Config:
    # Define network elements
    setup = PyphoSetup(
        nos=config.sqrt_filter_params.nos,
        sps=config.sqrt_filter_params.sps,
        symbolrate=config.sqrt_filter_params.symbolrate,
    )
    symbol_generator = SymbolGenerator(glova=setup, nos=setup.nos, pattern="random")
    signal_source = SignalSource(glova=setup, pulseshape="gauss_rz", fwhm=0.33)
    laser = ModulatedLaser(glova=setup, power=0, Df=0, teta=0)
    optical_filter = OpticalFilter(
        glova=setup, Df=0, B=20, filtype="cosrolloff", alpha=0.5
    )
    modulator = ArbitraryModulator(glova=setup)

    # Simulation
    # 4-QAM
    constellation_points_4qam = [
        ([1.0 * np.exp(2.0j * np.pi * x / 4.0) for x in range(0, 4)]),
        ([(0), (0)], [(0), (1)], [(1), (1)], [(1), (0)]),
    ]

    one_bit_sequence = symbol_generator(pattern="ones")
    electrical_signal = signal_source(
        bitsequence=one_bit_sequence
    )  # Generate the electrical signal
    transmitted_signal = laser(
        esig=electrical_signal
    )  # Generate the transmitted signal
    E = copy.deepcopy(transmitted_signal)  # Copy the transmitted signal
    output_temp = np.zeros((setup.nos * setup.sps, 4))  # Initialize the output array
    input_temp = np.zeros((setup.nos * setup.sps, 4))  # Initialize the input array

    df = pd.DataFrame(columns=["E_in", "E_out"])
    data: Dict[str, List] = {
        "E_in": [],
        "E_out": [],
    }

    for trial in tqdm(range(config.sqrt_filter_params.num_trials)):
        # Generate random complex numbers for the input signal
        E[0]["E"][0] = (
            np.random.uniform(0, 1, setup.sps * setup.nos)
            + 1j * np.random.uniform(0, 1, setup.sps * setup.nos)
            - 0.5
            - 0.5j
        )
        E[0]["E"][1] = (
            np.random.uniform(0, 1, setup.sps * setup.nos)
            + 1j * np.random.uniform(0, 1, setup.sps * setup.nos)
            - 0.5
            - 0.5j
        )

        # Reshape the input signal
        input_temp = np.reshape(
            np.array([np.imag(E[0]["E"][0])]), 1 * setup.sps * setup.nos
        )

        # Apply the optical filter to the input ssss
        E = optical_filter(E=copy.deepcopy(E))

        # Reshape the output signal
        output_temp = np.reshape(
            np.array([np.imag(E[0]["E"][0])]), 1 * setup.sps * setup.nos
        )

        data["E_in"].append(input_temp)
        data["E_out"].append(output_temp)

    df = pd.DataFrame(data)
    df.to_pickle(config.paths.sqrt_filter_file)

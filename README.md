# NOSMod Project

## Overview

**NOSMod** (**N**eural **O**ptical **S**ignal **Mod**ulation and Demodulation) is a deep learning project focused on minimizing inter-symbol interference (ISI) and more generally total bit error rate (BER) in optical communication systems through the use of neural networks. This project aims to implement a system consisting of two neural networks (Encoder and Decoder), which are _jointly optimized_ under these objectives.

In **fibre optical communication systems**, the modulation and demodulation processes are crucial for transmitting data with high fidelity. Traditional modulators and demodulators face challenges with ISI when ultra-narrow filtering is applied. Utilizing narrow filters is essential for maximizing the spectral efficiency (channel capacity) of optical communication systems. The **Nyquist ISI Criterion** describes conditions under which the impulse response of a communication channel results in no ISI. This is achieved when the impulse response is zero at all sampling points except one, ensuring that each symbol is isolated from the others during transmission and reception.
Additionally, **Channel Capacity**, defined as the maximum rate at which information can be transmitted over a communication channel with an arbitratily low error.


**NOSMod** seeks to address these challenges by leveraging neural networks to create an advanced modulation and demodulation system. By training the encoder and decoder networks jointly, the system aims to minimize ISI, allowing the use of narrow filters without significant information loss. The aim is to enhance the channel capacity by developing a holistic modulation scheme that can account for the behavior of the transmission line, as well as interactions between the I and Q components as well as both polarizations.

## Objectives
- **Transmission Line**: Start with a simple Cosine Roll-off filter ([Cosine Roll-off Filter](doc/FILTER.md)) and extend to more complex transmission line models that consider real-world channel effects. The transmission line should be differentiable to allow backpropagation.
- **Encoder**: Modulate the digital input data into a complex-valued signal for transmission over the optical channel.
- **Decoder**: Reconstruct the original data from the received signal.
- **Joint Training**: Optimize both networks jointly.

## Training Considerations
- Generate new (random) data for each training epoch/transmission.
- Employ state-of-the-art neural architecture search methods.

## Metrics
To evaluate the performance of the neural modulation and demodulation system, we consider the following metrics:

- **Bit Error Rate (BER)**: The number of bit errors divided by the total number of bits transmitted. Lower BER indicates better performance.
- **Reconstruction Error**: Typically measured as Mean Squared Error (MSE) between the transmitted and received signals.
- **Latent Space Regularity**: Assessed using KL Divergence, particularly for Variational Autoencoders (VAEs).
- **Perceptual Loss**: Measures the perceptual similarity between the original and reconstructed signals, ensuring that the reconstructed signal maintains the qualitative features of the original.
- **Cross-Entropy Loss (CE-Loss)**: Commonly used for classification tasks, could be adapted depending on the nature of the modulation scheme.

## Potential Network Architectures
- **Convolutional Neural Networks (CNNs)**: Effective for spatial data processing and can be adapted for 1D signal processing.
- **Recurrent Neural Networks (RNNs)**: Such as LSTM or GRU, suitable for sequential data and capturing temporal dependencies.
- **Transformers**: Efficient transformers can handle long-range dependencies and are suitable for sequential data processing. Given the expected correlation between subsequent symbols, attention mechanisms which consider only the temporal neighbourhood of a symbol could be beneficial.
- **Variational Autoencoders (VAEs)**: The problem can be formulated as a VAE task, where the transmission line represents the latent space?
- **Kolmogorov-Arnold Networks**: Might be a suitable architecture:
  - [Kolmogorov-Arnold Networks](https://arxiv.org/html/2404.19756v1)
  - [Convolutional Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.13155)


---

### Documentation
- [FILTER.md](doc/FILTER.md)
- [GUIDELINES.md](doc/GUIDELINES.md)
- [METRICS.md](doc/METRICS.md)
- [QUESTIONS.md](doc/QUESTIONS.md)
- [ROADMAP.md](doc/ROADMAP.md)
- [TODO.md](doc/TODO.md)
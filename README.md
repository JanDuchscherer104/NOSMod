# NOSMod Project

## Overview

**NOSMod** (**N**eural **O**ptical **S**ignal **Mod**ulation and Demodulation) is a deep learning project focused on minimizing inter-symbol interference (ISI) and total bit error rate (BER) in optical communication systems through the use of neural networks. This project aims to implement a system using two neural networks (Encoder and Decoder) optimized for these objectives.

## Problem Statement

In optical communication systems, the modulation and demodulation processes are crucial for transmitting data with high fidelity. Traditional modulators and demodulators often face challenges with inter-symbol interference (ISI) and bit error rate (BER), especially when employing ultra-narrow filters. These issues can degrade the quality of the transmitted signal, leading to higher error rates and reduced data integrity.

**NOSMod** seeks to address these challenges by leveraging neural networks to create an advanced modulation and demodulation system. By training the encoder and decoder networks jointly, the system aims to minimize ISI and BER, allowing the use of narrow filters without significant information loss.

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
- **Transformers**: Efficient transformers can handle long-range dependencies and are suitable for sequential data processing.
- **Variational Autoencoders (VAEs)**: Useful for encoding complex distributions and regularizing the latent space.

## Neural Architecture Search (NAS)
To identify the optimal neural network architectures for the encoder and decoder, we propose the following framework for Neural Architecture Search (NAS):

1. **Define the Search Space**:
    - Include a variety of layers (CNN, RNN, Transformer).
    - Parameter ranges for the number of layers, units per layer, kernel sizes, etc.
    - Activation functions and normalization techniques.

2. **Search Strategy**:
    - **Random Search**: Randomly sample architectures from the search space.
    - **Bayesian Optimization**: Use a probabilistic model to select promising architectures.
    - **Evolutionary Algorithms**: Evolve architectures over successive generations.

3. **Evaluation Strategy**:
    - Train sampled architectures on a subset of the data.
    - Evaluate using validation metrics (BER, MSE, etc.).
    - Rank architectures based on their performance.

4. **Selection and Refinement**:
    - Select the top-performing architectures.
    - Fine-tune the selected architectures on the full dataset.

## Project Roadmap

### Phase 1: Initial Research and Setup
- Conduct literature review on neural modulation and demodulation.
- Set up the project repository and initial codebase.
- Define theoretical basis and initial models.

### Phase 2: Development and Implementation
- Implement initial models for encoder and decoder.
- Develop and integrate the Cosine Roll-off filter.
- Implement data generation and preprocessing pipelines.

### Phase 3: Training and Optimization
- Train the initial models and evaluate performance.
- Implement Neural Architecture Search (NAS) framework.
- Optimize models using NAS and fine-tuning.

### Phase 4: Evaluation and Refinement
- Evaluate the final models on test data.
- Compare performance against traditional methods.
- Refine models based on evaluation results.

### Phase 5: Documentation and Deployment
- Document the theoretical basis, model architectures, and training processes.
- Package the final implementation as a PyPi package.
- Prepare a comprehensive project report and publish results.

## Getting Started
### Prerequisites
- Python packages listed in `nosmod/requirements.txt`

### Installation
To install dependencies:
```bash
pip install -r nosmod/requirements.txt

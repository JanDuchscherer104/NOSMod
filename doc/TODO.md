## TODOs

### NOSMod
- [ ] Ensure correctness of the raised cosine filter implementation.
- [ ] Improve model for transmission line, receiver, transmitter.
- [ ] Describe prohect goals from a theoretical basis:
    - [ ] Find 2NNs (Encoder + Decoder, jointly trained).
- [ ] Do some research on the theory in the field:
    - [ ] Find suitable metrics to quantify the quality of the estimator.
- [ ] Define functions and objects that can be used for:
    - [ ] Plotting.
    - [ ] Data handling.
    - [ ] Training.
    - [ ] Evaluation.
    - [ ] Experiment management - cloud would be nice - should be free though.
    - [ ] Parameter search with state-of-the-art methods.
- [ ] Add theoretical MD files, papers, and notes to the repository.
- [ ] Create TODOs as issues/epics in the repository.
- [ ] Implement neural filter, to substitude of raised cosine filter.
- [ ] Get typical values of parameters:
  - typical frequency ranges
  - sampling rates
  - symbol and bit rates.
- [ ] Which Alphabets to use, characteristics of symbols.
- [ ] Define interface classes for:
  - **Data Generationm:** $\text{Params} \rightarrow \mathcal{T}_{\mathcal{A}} \in \mathbb{R}^{\mathcal{B} \times \|\mathcal{A}\|}$, where $\mathcal{A}$ is the alphabet and $\|\mathcal{A}\|$ is the number of symbols in the alphabet, and might be multi-dimensional. $\mathcal{T}_{\mathcal{A}}$ is a tensor in the symbol space of the alphabet.
  - **Modulator:** $\mathcal{T}_{\mathcal{A}} \rightarrow \mathcal{T}_{\mathbb{C}} \in \mathbb{C}^{\mathcal{B} \times 2}$, where $\mathcal{B}$ is the batch size, and $\mathcal{T}_{\mathbb{C}}$ is a tensor in the space of two complex planes (polarizations).
  - **Transmission Line:** $\mathcal{T}_{\mathbb{C}} \rightarrow \mathcal{T}_{\mathbb{C}}$
  - **Demodulator:** $\mathcal{T}_{\mathbb{C}} \rightarrow \mathcal{T}_{\mathcal{A}}$
---
---

### Further Improvements of PyPho
- [ ] Rewrite PyPho in Mojo - `MoPho` with full GPU support for large-scale simulations.
- [ ] Implement the entire and improved PyPho library in Mojo and PyTorch (MoPho: MoPho).
- [ ] Define interface class for network components.
- [ ] Create a MoPho PyPi package.
- [ ] Use `PyTorch` to obtain differentiable components; allows to calculate the gradient of a variable w.r.t. parameters of the employed models or other variables in simulated or measured data.
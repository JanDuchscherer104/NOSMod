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

---
---

### Further Improvements of PyPho
- [ ] Rewrite PyPho in Mojo - `MoPho` with full GPU support for large-scale simulations.
- [ ] Implement the entire and improved PyPho library in Mojo and PyTorch (MoPho: MoPho).
- [ ] Create a MoPho PyPi package.
- [ ] Use `PyTorch` to obtain differentiable components; allows to calculate the gradient of a variable w.r.t. parameters of the employed models or other variables in simulated or measured data.
# NOSMod Project Roadmap

## Phase 1: Initial Research and Setup
- [ ] **Literature Review**: Summarize key papers and research findings in `doc/LITERATURE_REVIEW.md`.
- [x] **Repository Setup**: Create a GitHub repository and organize the project structure.
- [x] **Initial Documentation**: Write an initial `README.md`, some docs that outline the theoretical basis and the project's goals.
- [ ] **Theoretical Basis**: Document the theoretical background and initial models in `doc/THEORY.md`.

## Phase 2: Development and Implementation

- [ ] **>>WIP<< Filter Integration**: Implement the Cosine Roll-off filter and integrate it into the pipeline.
- [ ] **Model Implementation**: Develop and test initial encoder and decoder models.
- [ ] **>>WIP<< PyTorch Lightning Framework**: Implement a top-level PyTorch Lightning framework for training and evaluation.
- [ ] **Operation Space**: Outline the constraints under which the system should operate.
  - filter parameters, freqency ranges
  - modulation schemes, spaces of encoded and decoded symbols
  - noise models
  - improve model for transmission line, receiver, transmitter.
 **Differentiable Components**: Ensure all components are differentiable for backpropagation.

## Phase 3: Training and Optimization
- [ ] **Model Training**: Train the initial models using the data pipeline and evaluate performance using defined metrics.
- [ ] **NAS Framework**: Develop and integrate a NAS framework to explore various network architectures.

## Phase 4: Evaluation and Refinement

- [ ] **Comprehensive Evaluation**: Test the final models using a separate test dataset and evaluate performance using metrics like BER, MSE, SNR, etc.
- [ ] **Comparison Study**: Compare the NN-based approach with traditional methods.
- [ ] **Model Refinement**: Refine the models based on evaluation and comparison results.

## Phase 5: Documentation and Deployment

- [ ] **Documentation**: Write detailed documentation for all aspects of the project, including theory, implementation, and results.
- [ ] **Packaging**: Develop (and publish) a PyPi package for the NOSMod library.
- [ ] **Project Report**: Compile a comprehensive project report summarizing the goals, methodology, and results.


---
---

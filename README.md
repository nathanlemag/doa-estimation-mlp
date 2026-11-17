# Deep Learning for Single-Snapshot DOA Estimation

A PyTorch multi-head MLP that estimates both the number of active sources ($K \in \{1, 2, 3, 4\}$) and their Direction-of-Arrival (DOA) angles from a single snapshot ($T = 1$).

Based on the paper:  
**"Performance Advantages of Deep Neural Networks for Angle of Arrival Estimation"**

## Project Overview

Please refer to the Jupyter notebook (`HW5.ipynb`) for full explanations, visualizations, and results.

## Project structure

```text
doa-estimation-mlp/
├── assets/          # Images
├── data/            # Snapshot generation and PyTorch DataLoaders
├── evaluation/      # Evaluation metrics and plotting utilities
├── models/          # MLP architecture
├── training/        # Loss function and training loop
├── weights/         # Saved model weights
└── HW5.ipynb        # Main project report and experiments

```
---
*ELEC-E5481 — Machine Learning for Signal Processing, Aalto University, Autumn 2026*

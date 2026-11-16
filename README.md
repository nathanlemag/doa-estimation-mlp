# Deep Learning for Single-Snapshot DOA Estimation

PyTorch implementation of a multi-head Multi-Layer Perceptron (MLP) for joint source number detection ($K \in \{1, 2, 3, 4\}$) and Direction-of-Arrival (DOA) estimation from a single array snapshot ($T = 1$).

## Project Structure

- `data/data_generation.py` — Generates synthetic complex array snapshots across multiple SNR levels.
- `data/dataloader.py` — Splits complex snapshots into L2-normalized real/imaginary vectors, pads target angles, and builds PyTorch `DataLoader`s.
- `models/network.py` — 8-layer MLP (`Linear -> BatchNorm1d -> ReLU`) with 5 parallel output heads (1 classification head + 4 DOA regression heads) and PNG architecture export.
- `training/loss.py` — Hybrid loss combining `CrossEntropyLoss` for source counting and masked, permutation-invariant `MSELoss` (via sorting) for DOA regression.
- `training/trainer.py` — Training and validation loop using the Adam optimizer.
- `evaluation/metrics.py` — Evaluates source count accuracy (%) and DOA RMSE (degrees) per SNR on the test set.
- `HW5.ipynb` — Main notebook running data generation, training, and evaluation plots.

## Google colab setup

Link to this repo
[doa-estimation-mlp](https://github.com/nathanlemag/doa-estimation-mlp)

On local terminal after file changes:

```bash
git add . && git commit -m "fix" && git push
```

On on a google collab cell after changes:

```bash
!git pull
```

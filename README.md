# Deep Learning for Single-Snapshot DOA Estimation

A PyTorch multi-head MLP that estimates both the number of active sources ($K \in \{1, 2, 3, 4\}$) and their Direction-of-Arrival (DOA) angles from a single snapshot ($T = 1$).

## Project Structure

- `data/data_generation.py` — Creates synthetic array snapshots across different SNR levels.
- `data/dataloader.py` — Splits complex snapshots into real and imaginary parts, normalizes them, pads target angles to fixed size, and builds PyTorch `DataLoader`s.
- `models/network.py` — Defines the 8-layer MLP (`Linear -> BatchNorm1d -> ReLU`) with 5 parallel heads (1 for counting sources, 4 for predicting angles) and exports the model diagram.
- `training/loss.py` — Combines cross-entropy loss (for source count) and masked RMSE loss (for sorted DOA angles).
- `training/trainer.py` — Runs the training and validation loop using Adam.
- `evaluation/metrics.py` — Computes source count accuracy (%) and DOA RMSE (degrees) at each SNR level and plots the curves.
- `weights/doa_model.pt` — Saved weights of the trained model (used when `SKIP_TRAINING = True`).
- `HW5.ipynb` — Main notebook that ties everything together and displays the results.

## Google Colab Workflow

Repository link: [doa-estimation-mlp](https://github.com/nathanlemag/doa-estimation-mlp)

After editing files locally:

```bash
git add . && git commit -m "update" && git push
```

To pull the latest changes in Google Colab:

```bash
!git pull
```

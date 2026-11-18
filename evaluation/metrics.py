import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def evaluate_by_snr(model, test_loader):
    device = next(model.parameters()).device
    model.eval()

    all_pred_k, all_true_k, all_snr, all_mse = [], [], [], []

    with torch.no_grad():
        for x, true_class, true_doas, snr in test_loader:
            class_logits, doa_preds = model(x.to(device))

            all_pred_k.append(class_logits.argmax(dim=1).cpu())
            all_true_k.append(true_class)
            all_snr.append(snr)

            # Mean squared error (MSE) per sample according to the true number of sources K
            batch_mse = torch.zeros(len(x))
            for k_idx, head_pred in enumerate(doa_preds):
                mask = (true_class == k_idx)
                if mask.any():
                    pred_angles = torch.sort(head_pred[mask], dim=1).values.cpu()
                    true_angles = true_doas[mask, : k_idx + 1]
                    batch_mse[mask] = F.mse_loss(pred_angles, true_angles, reduction="none").mean(dim=1)

            all_mse.append(batch_mse)

    # Concatenation of all batches into 1D NumPy arrays
    pred_k = torch.cat(all_pred_k).numpy()
    true_k = torch.cat(all_true_k).numpy()
    snr_arr = torch.cat(all_snr).numpy()
    mse_arr = torch.cat(all_mse).numpy()

    # Calculation of accuracy and RMSE for each SNR level
    snrs = np.unique(snr_arr)
    acc_per_snr = [100.0 * (pred_k[snr_arr == s] == true_k[snr_arr == s]).mean() for s in snrs]
    rmse_per_snr = [np.sqrt(mse_arr[snr_arr == s].mean()) for s in snrs]

    return snrs, acc_per_snr, rmse_per_snr


def plot_snr_results(snrs, acc_per_snr, rmse_per_snr):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(snrs, acc_per_snr, marker="o")
    ax1.set(title="Source Count Detection Accuracy vs SNR", xlabel="SNR (dB)", ylabel="Accuracy (%)")
    ax1.grid(True)

    ax2.plot(snrs, rmse_per_snr, marker="o", color="orange")
    ax2.set(title="DOA Estimation RMSE vs SNR", xlabel="SNR (dB)", ylabel="RMSE (degrees)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
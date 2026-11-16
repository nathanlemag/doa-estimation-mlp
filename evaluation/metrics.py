import numpy as np
import torch
import matplotlib.pyplot as plt


def evaluate_by_snr(model, test_loader):
    device = next(model.parameters()).device
    model.eval()

    all_pred_k, all_true_k, all_snr, all_mse = [], [], [], []

    with torch.no_grad():
        for x, true_class, true_doas, snr in test_loader:
            x = x.to(device)
            class_logits, doa_preds = model(x)

            # 1. Predicted number of sources (0..3)
            all_pred_k.append(torch.argmax(class_logits, dim=1).cpu())
            all_true_k.append(true_class)
            all_snr.append(snr)

            # 2. Vectorized angle MSE per sample (grouped by true K = 1..4)
            batch_mse = torch.zeros(len(x))
            for k_idx in range(4):
                mask = (true_class == k_idx)
                if mask.any():
                    pred_angles = torch.sort(doa_preds[k_idx][mask], dim=1).values.cpu()
                    true_angles = true_doas[mask, : k_idx + 1]
                    batch_mse[mask] = torch.mean((pred_angles - true_angles) ** 2, dim=1)

            all_mse.append(batch_mse)

    # Concatenate all batches into 1D NumPy arrays
    pred_k = torch.cat(all_pred_k).numpy()
    true_k = torch.cat(all_true_k).numpy()
    snr_arr = torch.cat(all_snr).numpy()
    mse_arr = torch.cat(all_mse).numpy()

    # Compute Accuracy (%) and RMSE (degrees) for each SNR level
    snrs = np.sort(np.unique(snr_arr))
    acc_per_snr = [100.0 * np.mean(pred_k[snr_arr == s] == true_k[snr_arr == s]) for s in snrs]
    rmse_per_snr = [np.sqrt(np.mean(mse_arr[snr_arr == s])) for s in snrs]

    return snrs, acc_per_snr, rmse_per_snr


def plot_snr_results(snrs, acc_per_snr, rmse_per_snr):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(snrs, acc_per_snr, marker="o")
    ax1.set_title("Source Count Detection Accuracy vs SNR")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True)

    ax2.plot(snrs, rmse_per_snr, marker="o", color="orange")
    ax2.set_title("DOA Estimation RMSE vs SNR")
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("RMSE (degrees)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
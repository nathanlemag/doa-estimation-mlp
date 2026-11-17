import numpy as np
import torch
import matplotlib.pyplot as plt


def evaluate_by_snr(model, test_loader):
    # Find out if the model is on GPU or CPU and switch it to evaluation mode
    device = next(model.parameters()).device
    model.eval()

    # Lists to store predictions and ground truth across all test batches
    all_pred_k, all_true_k, all_snr, all_mse = [], [], [], []

    # Disable gradient tracking to save memory and speed up testing
    with torch.no_grad():
        for x, true_class, true_doas, snr in test_loader:
            x = x.to(device)
            class_logits, doa_preds = model(x)  # Run the model on the test batch

            
            all_pred_k.append(torch.argmax(class_logits, dim=1).cpu()) # Source counting: pick the class index (0..3) with the highest score
            all_true_k.append(true_class)
            all_snr.append(snr)

            # Angle error (MSE): same logic as in DOALoss, grouped by true K (1..4)
            batch_mse = torch.zeros(len(x))
            for k_idx in range(4):
                mask = (true_class == k_idx)  # Select only samples that truly have k_idx + 1 sources
                if mask.any():
                    # Sort predicted angles from smallest to largest so they match the sorted true angles
                    pred_angles = torch.sort(doa_preds[k_idx][mask], dim=1).values.cpu()
                    # Keep only the real angles (ignore the zero-padding on the right)
                    true_angles = true_doas[mask, : k_idx + 1]
                    # Compute the average squared angle error for each sample in this group
                    batch_mse[mask] = torch.mean((pred_angles - true_angles) ** 2, dim=1)

            all_mse.append(batch_mse)

    # Merge all batches into single 1D NumPy arrays for easy filtering
    pred_k = torch.cat(all_pred_k).numpy()
    true_k = torch.cat(all_true_k).numpy()
    snr_arr = torch.cat(all_snr).numpy()
    mse_arr = torch.cat(all_mse).numpy()

    # Get every unique SNR level in ascending order (e.g., 0, 1, ..., 9 dB)
    snrs = np.sort(np.unique(snr_arr))

    # For each SNR level, compute the percentage of exact K predictions
    acc_per_snr = [100.0 * np.mean(pred_k[snr_arr == s] == true_k[snr_arr == s]) for s in snrs]
    # For each SNR level, take the square root of the mean MSE to get RMSE in degrees
    rmse_per_snr = [np.sqrt(np.mean(mse_arr[snr_arr == s])) for s in snrs]

    return snrs, acc_per_snr, rmse_per_snr


def plot_snr_results(snrs, acc_per_snr, rmse_per_snr):
    # Create a side-by-side figure with two plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left plot: Source count accuracy (%) vs. SNR
    ax1.plot(snrs, acc_per_snr, marker="o")
    ax1.set_title("Source Count Detection Accuracy vs SNR")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True)

    # Right plot: Angle estimation error (RMSE in degrees) vs. SNR
    ax2.plot(snrs, rmse_per_snr, marker="o", color="orange")
    ax2.set_title("DOA Estimation RMSE vs SNR")
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("RMSE (degrees)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
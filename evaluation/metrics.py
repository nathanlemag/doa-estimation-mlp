import numpy as np
import torch


def evaluate_by_snr(model, test_loader):
    device = next(model.parameters()).device
    model.eval()

    all_pred_k, all_true_k, all_snr, all_sq_errors = [], [], [], []

    with torch.no_grad():
        for x, true_class, true_doas, snr in test_loader:
            x = x.to(device)
            class_logits, doa_preds = model(x)

            # Predicted K (0..3)
            pred_class = torch.argmax(class_logits, dim=1).cpu()

            all_pred_k.append(pred_class)
            all_true_k.append(true_class)
            all_snr.append(snr)

            # Compute angle MSE per sample using the true K head
            batch_errors = torch.zeros(len(x))
            for i in range(len(x)):
                k_idx = true_class[i].item()
                k = k_idx + 1
                pred_angles = torch.sort(doa_preds[k_idx][i]).values.cpu()
                true_angles = true_doas[i, :k]
                batch_errors[i] = torch.mean((pred_angles - true_angles) ** 2)

            all_sq_errors.append(batch_errors)

    pred_k = torch.cat(all_pred_k).numpy()
    true_k = torch.cat(all_true_k).numpy()
    snr_arr = torch.cat(all_snr).numpy()
    sq_errors = torch.cat(all_sq_errors).numpy()

    # Automatically get all unique SNR values present in the test set
    snrs = np.sort(np.unique(snr_arr))
    acc_per_snr = []
    rmse_per_snr = []

    for s in snrs:
        mask = (snr_arr == s)
        acc = 100.0 * np.mean(pred_k[mask] == true_k[mask])
        rmse = np.sqrt(np.mean(sq_errors[mask]))
        acc_per_snr.append(acc)
        rmse_per_snr.append(rmse)

    return snrs, acc_per_snr, rmse_per_snr
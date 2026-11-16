import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def prepare_dataloaders(snapshots, doas, snrs, samples_per_snr, batch_size=256):
    # Temporary lists to collect data for each K before merging them all at once
    X_list, class_list, doa_list, snr_list = [], [], [], []

    for k_str in snapshots:
        k = int(k_str)

        # 1. Prepare input X: complex (10000, 20, 1) -> real (10000, 40)
        snaps = np.array(snapshots[k_str])      # convert list to array (10000, 20, 1)
        snaps = snaps.squeeze(-1)               # remove last dimension -> (10000, 20)
        x = np.hstack([snaps.real, snaps.imag]) # type: ignore # stack all 20 real parts then all 20 imag parts -> (10000, 40)
        X_list.append(x)

        # 2. Prepare target DOAs: sort each row and pad with zeros at the end to fixed size (10000, 4)
        angles = np.sort(np.array(doas[k_str]), axis=1) # axis=1 sorts each sample's angles from low to high
        doa_pad = np.pad(angles, ((0, 0), (0, 4 - k)))  # (0,0) = no row padding; (0, 4-k) = pad 0s on the right
        doa_list.append(doa_pad)

        # 3. Prepare classification labels (0 to 3) and SNR values -> 1D arrays of shape (10000,)
        class_idx = np.full(len(x), k - 1)              # PyTorch CrossEntropyLoss expects class indices in [0, 1, 2, 3]
        class_list.append(class_idx)

        snr_vals = np.repeat(snrs, samples_per_snr)
        snr_list.append(snr_vals)
        

    # Merge all 4 classes into single PyTorch tensors (40000 samples total)
    dataset = TensorDataset(
        torch.tensor(np.vstack(X_list), dtype=torch.float32),       # vstack 2D arrays -> (40000, 40)
        torch.tensor(np.concatenate(class_list), dtype=torch.long), # concatenate 1D arrays -> (40000,), long required for CrossEntropyLoss
        torch.tensor(np.vstack(doa_list), dtype=torch.float32),     # vstack 2D arrays -> (40000, 4)
        torch.tensor(np.concatenate(snr_list), dtype=torch.float32),# concatenate 1D arrays -> (40000,)
    )

    # 80 / 10 / 10 train / val / test split with fixed seed for reproducibility
    train_set, val_set, test_set = random_split(
        dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
    )

    # Wrap datasets into DataLoaders to feed batches of 256 samples to the network
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )
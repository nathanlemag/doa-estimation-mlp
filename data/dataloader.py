import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def DOADataloader(snapshots, doas, snrs, samples_per_snr, batch_size=256):
    X_list, class_list, doa_list, snr_list = [], [], [], []
    max_sources = len(snapshots)

    for k_str, snap_list in snapshots.items():
        k = int(k_str)

        # Complex (N, M, 1) -> Real (N, 2*M)
        snaps = np.asarray(snap_list).squeeze(-1)
        X_list.append(np.hstack([snaps.real, snaps.imag])) # type: ignore

        # Sorting angles and padding on the right up to max_sources
        angles = np.sort(doas[k_str], axis=1)
        doa_list.append(np.pad(angles, ((0, 0), (0, max_sources - k))))

        # Labels (0 to 3) and associated SNR
        class_list.append(np.full(len(snaps), k - 1))
        snr_list.append(np.repeat(snrs, samples_per_snr))

    dataset = TensorDataset(
        torch.tensor(np.vstack(X_list), dtype=torch.float32),
        torch.tensor(np.concatenate(class_list), dtype=torch.long),
        torch.tensor(np.vstack(doa_list), dtype=torch.float32),
        torch.tensor(np.concatenate(snr_list), dtype=torch.float32),
    )

    train_set, val_set, test_set = random_split(
        dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
    )

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )
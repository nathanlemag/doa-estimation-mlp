import torch
import torch.nn as nn
import torch.nn.functional as F


class DOALoss(nn.Module):
    def forward(self, class_logits, doa_preds, true_class, true_doas):
        doa_1, doa_2, doa_3, doa_4 = doa_preds

        # 1. Classification loss: compares predicted logits (B, 4) with true class indices (B,) in {0, 1, 2, 3}
        loss_class = F.cross_entropy(class_logits, true_class)

        # 2. Regression loss:
        # Each sample in the batch has a different true K, so we filter (mask) the batch by class:
        #   - mask_k is a True/False list of length B (e.g. [False, True, False, True])
        #   - doa_k[mask_k] keeps ONLY the rows where true K matches this head
        #   - true_doas[mask_k, :k] keeps those same rows AND slices the first k columns (removing the 0-padding)
        #   - torch.sort(..., dim=1).values sorts predicted angles from low to high so they match true_doas order

        # K = 1 (class 0)
        mask_1 = (true_class == 0)
        loss_1 = F.mse_loss(doa_1[mask_1], true_doas[mask_1, :1])

        # K = 2 (class 1)
        mask_2 = (true_class == 1)
        pred_2 = torch.sort(doa_2[mask_2], dim=1).values
        loss_2 = F.mse_loss(pred_2, true_doas[mask_2, :2])

        # K = 3 (class 2)
        mask_3 = (true_class == 2)
        pred_3 = torch.sort(doa_3[mask_3], dim=1).values
        loss_3 = F.mse_loss(pred_3, true_doas[mask_3, :3])

        # K = 4 (class 3)
        mask_4 = (true_class == 3)
        pred_4 = torch.sort(doa_4[mask_4], dim=1).values
        loss_4 = F.mse_loss(pred_4, true_doas[mask_4, :4])

        # Average MSE across the 4 heads
        loss_reg = (loss_1 + loss_2 + loss_3 + loss_4) / 4.0

        return loss_class + loss_reg, loss_class, loss_reg
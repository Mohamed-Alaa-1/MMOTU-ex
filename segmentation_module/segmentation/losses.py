"""
segmentation/losses.py

Dice + BCEWithLogits loss, matching the original AuraViT training recipe
(see segmentation_extension_plan.md Section 1, point 6). Operates on raw
logits directly (seg_output has no sigmoid applied inside the model), so
BCEWithLogits is used rather than plain BCE, and Dice is computed on
sigmoid(logits) internally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0, bce_weight: float = 0.5):
        super().__init__()
        if not 0.0 <= bce_weight <= 1.0:
            raise ValueError(f"bce_weight must be in [0, 1], got {bce_weight}")
        self.smooth = smooth
        self.bce_weight = bce_weight

    def forward(self, pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        if pred_logits.shape != target_mask.shape:
            raise ValueError(
                f"pred_logits shape {tuple(pred_logits.shape)} does not match "
                f"target_mask shape {tuple(target_mask.shape)}"
            )

        pred_probs = torch.sigmoid(pred_logits)
        pred_flat = pred_probs.reshape(pred_probs.size(0), -1)
        target_flat = target_mask.reshape(target_mask.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice_score.mean()

        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_mask)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss

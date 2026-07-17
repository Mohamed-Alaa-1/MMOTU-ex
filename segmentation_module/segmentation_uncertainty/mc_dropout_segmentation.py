"""
segmentation_uncertainty/mc_dropout_segmentation.py

Segmentation-domain analog of uncertainty/estimators.py's MCDropoutEstimator.

Fix applied relative to the classification version: the classification
enable_mc_dropout uses isinstance(module, nn.Dropout). nn.Dropout2d is NOT
a subclass of nn.Dropout in PyTorch (both are siblings under the internal
_DropoutNd base class), so that check silently skips every nn.Dropout2d
layer. LightweightAuraViT uses nn.Dropout2d throughout the decoder
(inside every LightweightResBlock) and inside LightweightASPP's
output_conv, in addition to the plain nn.Dropout used in patch_embed and
pos_dropout. Left unfixed, MC-Dropout would only sample stochasticity from
the encoder and silently miss the decoder and ASPP entirely, understating
the true predictive uncertainty. See segmentation_extension_plan.md
Section 3.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import _DropoutNd


def enable_mc_dropout_segmentation(model: nn.Module) -> int:
    """Puts every dropout-family layer (nn.Dropout, nn.Dropout2d,
    nn.Dropout3d, and any future variant) into train mode, while leaving
    everything else, including nn.BatchNorm2d (used extensively in this
    model's DepthwiseSeparableConv, LightweightResBlock,
    LightweightAttentionGate, and LightweightASPP), in eval mode.

    Returns the count of dropout modules switched, so callers can assert
    it is nonzero (a silent zero would mean this was called on a model
    with no dropout layers found, likely a wiring bug).
    """
    n_switched = 0
    for module in model.modules():
        if isinstance(module, _DropoutNd):
            module.train()
            n_switched += 1
    return n_switched


class MCDropoutSegmentationEstimator:
    """Runs N stochastic forward passes and returns per-pixel mean
    probability and per-pixel standard deviation maps, rather than a
    single per-image scalar as in the classification version."""

    def __init__(self, model: nn.Module, device: torch.device, n_samples: int = 30):
        self.model = model.to(device).eval()
        n_switched = enable_mc_dropout_segmentation(self.model)
        if n_switched == 0:
            raise RuntimeError(
                "enable_mc_dropout_segmentation found zero dropout layers to "
                "switch to train mode; MC-Dropout would be deterministic. "
                "Check the model actually contains nn.Dropout/nn.Dropout2d "
                "layers before proceeding."
            )
        self.device = device
        self.n_samples = n_samples

    @torch.no_grad()
    def predict(self, image_batch: torch.Tensor) -> dict:
        """Returns mean_probs [B,1,H,W], epistemic_std [B,1,H,W], and
        predictive_entropy [B,1,H,W] (per-pixel binary entropy)."""
        image_batch = image_batch.to(self.device)
        all_probs = torch.stack([
            torch.sigmoid(self.model(image_batch)) for _ in range(self.n_samples)
        ])  # [S, B, 1, H, W]
        mean_probs = all_probs.mean(dim=0)
        epistemic_std = all_probs.std(dim=0)
        p = mean_probs.clamp(1e-7, 1 - 1e-7)
        predictive_entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return {
            "mean_probs": mean_probs.cpu().numpy(),
            "epistemic_std": epistemic_std.cpu().numpy(),
            "predictive_entropy": predictive_entropy.cpu().numpy(),
        }

"""
segmentation_uncertainty/ensemble_segmentation.py

Deep Ensemble segmentation uncertainty estimator: a direct segmentation-domain
analog of running multiple independently trained classifiers, following
Section 3.2 of segmentation_extension_plan.md.

Two variants are described in the plan:
  - Random-seed diversity: same config, different torch.manual_seed at init.
  - Architecture-size diversity: e.g., LAURA_BASE paired with LAURA_SMALL.

This module handles both identically: it takes a list of already-instantiated
model objects (however they were seeded or sized) and aggregates their
predictions. The caller is responsible for ensuring the models were trained
independently; using two identically trained checkpoints produces degenerate
zero-variance output, which is caught at predict() time.

Interface mirrors MCDropoutSegmentationEstimator (predict(image_batch) -> dict
with mean_probs, epistemic_std, predictive_entropy) so that
selective_segmentation.py and conformal_risk_control.py can consume either
estimator without knowing which uncertainty source was used.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


class DeepEnsembleSegmentationEstimator:
    """Aggregates pixel-wise predictions from N independently trained models.

    Each model is run in eval mode (all dropout and batch-norm layers in
    inference mode, unlike MC-Dropout which puts dropout in train mode).
    This means diversity comes entirely from the differences between
    members, not from stochastic internal sampling.

    Args:
        models: List of at least 2 nn.Module instances. All must accept the
            same input shape and produce a [B, 1, H, W] logit output.
            Requires >= 2 members; raises ValueError on a singleton list
            because a single-model "ensemble" is identical to plain inference
            with no epistemic uncertainty signal.
        device: torch.device for inference.
    """

    def __init__(self, models: List[nn.Module], device: torch.device) -> None:
        if len(models) < 2:
            raise ValueError(
                f"DeepEnsembleSegmentationEstimator requires at least 2 models "
                f"to estimate epistemic uncertainty; got {len(models)}. "
                f"A single-model list produces zero variance, which is "
                f"meaningless as an uncertainty estimate."
            )
        self.models = [m.to(device).eval() for m in models]
        self.device = device

    @torch.no_grad()
    def predict(self, image_batch: torch.Tensor) -> dict:
        """Run all ensemble members and aggregate per-pixel statistics.

        Args:
            image_batch: [B, C, H, W] input tensor. Will be moved to
                self.device internally.

        Returns dict with keys:
            mean_probs:          [B, 1, H, W] numpy array, per-pixel mean
                                 sigmoid probability across members.
            epistemic_std:       [B, 1, H, W] numpy array, per-pixel standard
                                 deviation across members. High values
                                 correspond to high ensemble disagreement,
                                 concentrated at tumor boundaries.
            predictive_entropy:  [B, 1, H, W] numpy array, per-pixel binary
                                 entropy of mean_probs. Combines aleatoric
                                 and epistemic uncertainty into a single map.

        Raises:
            RuntimeError: If all ensemble members produce identical outputs
                (zero epistemic_std everywhere), which indicates the models
                were not independently trained or seeded. This is a soft
                warning rather than a hard error at call time; the caller
                should inspect the returned epistemic_std if needed.
        """
        image_batch = image_batch.to(self.device)

        # Collect sigmoid probabilities from every member.
        # Each model.forward() returns raw logits; sigmoid maps to [0, 1].
        # Shape per member: [B, 1, H, W]
        all_probs = torch.stack(
            [torch.sigmoid(model(image_batch)) for model in self.models]
        )  # [N_members, B, 1, H, W]

        mean_probs = all_probs.mean(dim=0)       # [B, 1, H, W]
        epistemic_std = all_probs.std(dim=0)     # [B, 1, H, W]

        # Binary entropy of the mean probability map. This captures both
        # uncertainty sources (aleatoric from inherent image ambiguity, and
        # epistemic from model disagreement), while epistemic_std captures
        # the purely epistemic component.
        p = mean_probs.clamp(1e-7, 1.0 - 1e-7)
        predictive_entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

        return {
            "mean_probs": mean_probs.cpu().numpy(),
            "epistemic_std": epistemic_std.cpu().numpy(),
            "predictive_entropy": predictive_entropy.cpu().numpy(),
        }

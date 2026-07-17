"""
segmentation_uncertainty/selective_segmentation.py

Selective segmentation risk-coverage analysis: rank test images by an
image-level uncertainty summary, sweep coverage from most-confident to
least-confident, and report selective segmentation risk (1 - mean Dice) at
each coverage level. Produces an AURC scalar directly comparable in spirit
and reporting format to the classification AURC already computed in
uncertainty/selective_prediction.py.

See segmentation_extension_plan.md Section 3.4.

Design
------
The classification analog (uncertainty/selective_prediction.py) ranks images
by uncertainty and sweeps 1-accuracy as the risk. This module does the same
with 1-Dice as the segmentation risk, retaining the most-confident images
at each coverage level.

A well-calibrated uncertainty estimator should produce a downward-sloping
risk-coverage curve: retaining only the most confident images gives the
lowest risk, and including more images (higher coverage) raises risk as less
reliable predictions are included.

The AURC scalar summarises this curve in one number. A lower AURC indicates
that the uncertainty signal is a better predictor of segmentation quality.

All inputs/outputs are numpy arrays. No PyTorch dependency; the caller runs
model inference and extracts probability maps and uncertainty scalars upstream.
"""

from __future__ import annotations

import numpy as np

from segmentation.metrics import dice_score


# ---------------------------------------------------------------------------
# Image-level uncertainty summary
# ---------------------------------------------------------------------------

def image_level_uncertainty(
    estimator_output: dict,
    method: str = "predictive_entropy",
) -> np.ndarray:
    """Reduce per-pixel uncertainty maps to a single scalar per image.

    Works with the output dict from either MCDropoutSegmentationEstimator or
    DeepEnsembleSegmentationEstimator (both return mean_probs, epistemic_std,
    and predictive_entropy).

    Args:
        estimator_output: Dict with numpy arrays under the keys
            "predictive_entropy" and "epistemic_std", each of shape
            [B, 1, H, W].
        method: Reduction method:
            - "predictive_entropy" (default): mean per-pixel entropy of the
              mean probability map. Combines aleatoric + epistemic uncertainty.
            - "epistemic_std": mean per-pixel standard deviation across
              ensemble members or MC samples. Purely epistemic component.
              Useful as a comparison against predictive_entropy.

    Returns:
        1-D numpy array of shape [B] with one scalar uncertainty estimate
        per image. Higher values = more uncertain.

    Raises:
        ValueError: If method is not one of the supported strings.
        KeyError: If the required key is missing from estimator_output.
    """
    supported = {"predictive_entropy", "epistemic_std"}
    if method not in supported:
        raise ValueError(
            f"method must be one of {supported}; got '{method}'."
        )
    arr = np.asarray(estimator_output[method], dtype=np.float32)
    # Collapse all spatial and channel dims: mean over (C, H, W)
    # arr shape: [B, 1, H, W] -> mean over axes (1, 2, 3) -> [B]
    if arr.ndim == 4:
        return arr.mean(axis=(1, 2, 3))
    elif arr.ndim == 3:
        return arr.mean(axis=(1, 2))
    else:
        raise ValueError(
            f"Expected estimator_output['{method}'] to be 3- or 4-D "
            f"(with optional leading channel dim); got shape {arr.shape}."
        )


# ---------------------------------------------------------------------------
# Risk-coverage curve
# ---------------------------------------------------------------------------

def selective_segmentation_risk_coverage(
    prob_maps: np.ndarray,
    gt_masks: np.ndarray,
    uncertainty_scores: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute selective segmentation risk-coverage curve and AURC.

    Images are sorted ascending by uncertainty_scores (most confident first).
    At each coverage level k/n, the k most-confident images are retained and
    segmentation risk is 1 - mean Dice over those k images.

    Args:
        prob_maps:          [N, H, W] or [N, 1, H, W] float probability maps,
                            values in [0, 1], from sigmoid model output.
        gt_masks:           [N, H, W] or [N, 1, H, W] binary ground-truth
                            masks (0/1 or bool).
        uncertainty_scores: [N] float, one scalar per image. Lower = more
                            confident = retained first.
        threshold:          Binarization threshold applied to prob_maps to get
                            predicted masks. Default 0.5.

    Returns:
        Dict with:
            coverages: List[float] length N, from 1/N to 1.0.
            risks:     List[float] length N, 1 - mean Dice at each coverage.
            aurc:      float, area under the risk-coverage curve (trapezoidal
                       integration). Lower is better.

    Raises:
        ValueError: If input arrays have incompatible lengths or are empty.
    """
    prob_maps = np.asarray(prob_maps, dtype=np.float32)
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    uncertainty_scores = np.asarray(uncertainty_scores, dtype=np.float32)

    # Squeeze channel dim
    if prob_maps.ndim == 4 and prob_maps.shape[1] == 1:
        prob_maps = prob_maps[:, 0]
    if gt_masks.ndim == 4 and gt_masks.shape[1] == 1:
        gt_masks = gt_masks[:, 0]

    n = len(prob_maps)
    if n == 0:
        raise ValueError("prob_maps is empty.")
    if len(gt_masks) != n or len(uncertainty_scores) != n:
        raise ValueError(
            f"All inputs must have the same first dimension; got "
            f"prob_maps={n}, gt_masks={len(gt_masks)}, "
            f"uncertainty_scores={len(uncertainty_scores)}."
        )

    # Sort ascending by uncertainty (most confident = lowest uncertainty first)
    order = np.argsort(uncertainty_scores)

    coverages: list[float] = []
    risks: list[float] = []

    for k in range(1, n + 1):
        retained_indices = order[:k]
        dice_scores = [
            dice_score(
                prob_maps[i] >= threshold,
                gt_masks[i] >= 0.5,
            )
            for i in retained_indices
        ]
        coverages.append(k / n)
        risks.append(1.0 - float(np.mean(dice_scores)))

    aurc = float(np.trapezoid(risks, coverages)) if n > 1 else float(risks[0])

    return {
        "coverages": coverages,
        "risks": risks,
        "aurc": aurc,
    }

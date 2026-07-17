"""
segmentation_uncertainty/conformal_risk_control.py

Conformal Risk Control (CRC) for pixel-level tumor segmentation, following
Angelopoulos et al. "Conformal Risk Control" (2022/2024) and its recent
medical segmentation applications (Mossina et al., CVPR Workshops 2024;
Bereska et al., 2025).

See segmentation_extension_plan.md Section 3.3.

Calibration task
----------------
Given a held-out calibration set of (prob_map, gt_mask) pairs, find the
largest pixel-probability threshold lambda* in [0, 1] such that the expected
per-image False Negative Rate (FNR) is guaranteed to be at most alpha on
unseen test images.

FNR definition
--------------
    FNR_i(lambda) = (# true positive pixels with predicted prob < lambda)
                    / (# total true positive pixels in image i)

FNR(lambda) is monotone non-increasing in lambda: as the threshold drops,
more pixels are predicted positive, so fewer true positives are missed.
The calibration therefore finds:

    lambda* = max { lambda in Lambda : mean_i [ FNR_i(lambda) ] <= alpha }

where Lambda is a fixed discrete grid [0.0, 0.01, ..., 1.0] (100 steps).
Images with no foreground pixels (empty masks) contribute FNR = 0 at all
thresholds and are included in the count.

Statistical guarantee (informal)
---------------------------------
By the CRC theorem, if the calibration set is exchangeable with the test set
and |calibration set| = n, the expected FNR on a fresh test image is at most
alpha + (1 / (n + 1)). For n >= 100, this slack is <= 1%. The caller should
ensure the calibration set is large enough relative to the target alpha.

All inputs and outputs are numpy arrays (not tensors). No PyTorch dependency
in this module so it can be tested independently of a GPU or a trained model.
"""

from __future__ import annotations

import numpy as np


# Discrete grid of threshold candidates. 101 points from 0.0 to 1.0
# inclusive gives 0.01 resolution, enough for clinical risk control at
# alpha = 0.05 or 0.10.
_LAMBDA_GRID: np.ndarray = np.linspace(0.0, 1.0, 101)


def _image_fnr(prob_map: np.ndarray, gt_mask: np.ndarray,
               threshold: float) -> float:
    """FNR for a single image at a given threshold.

    Args:
        prob_map:  2-D or 1-D float array of predicted pixel probabilities.
        gt_mask:   Same shape as prob_map, binary (0 or 1, bool ok).
        threshold: Scalar in [0, 1].

    Returns:
        FNR in [0, 1]. Returns 0.0 if gt_mask has no foreground pixels
        (empty mask means no false negatives are possible, so FNR = 0 by
        convention, consistent with treating this as a risk-free sample).
    """
    gt_flat = gt_mask.flatten().astype(bool)
    n_positive = int(gt_flat.sum())
    if n_positive == 0:
        return 0.0
    pred_positive = (prob_map.flatten() >= threshold)
    n_true_positive = int((pred_positive & gt_flat).sum())
    return 1.0 - n_true_positive / n_positive


class SegmentationConformalRiskController:
    """Calibrates and applies a Conformal Risk Control threshold for
    pixel-level segmentation, targeting a guaranteed FNR bound.

    Attributes:
        alpha: Target FNR bound (e.g., 0.10 for at most 10% of true tumor
               pixels missed in expectation on unseen images).
        lambda_star: Calibrated threshold (set after calibrate()).
        calibration_fnr: Empirical FNR at lambda_star on the calibration set
               (will be <= alpha by construction).
        n_calibration: Number of calibration images used.
        n_empty_masks: Number of calibration images with no foreground pixels
               (these contribute FNR=0 and are counted in n_calibration).
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(
                f"alpha must be in (0, 1); got {alpha}. "
                f"Typical values are 0.05 (5% FNR guarantee) or "
                f"0.10 (10% FNR guarantee)."
            )
        self.alpha = alpha
        self.lambda_star: float | None = None
        self.calibration_fnr: float | None = None
        self.n_calibration: int = 0
        self.n_empty_masks: int = 0

    def calibrate(self, prob_maps: np.ndarray,
                  gt_masks: np.ndarray) -> "SegmentationConformalRiskController":
        """Fit lambda_star from calibration data.

        Args:
            prob_maps: Array of predicted probability maps. Accepted shapes:
                [N, H, W] or [N, 1, H, W]. Values in [0, 1].
            gt_masks:  Corresponding ground-truth masks. Same shape as
                prob_maps, binary (0/1 or bool).

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If prob_maps is empty or shapes do not match.
        """
        prob_maps = np.asarray(prob_maps, dtype=np.float32)
        gt_masks = np.asarray(gt_masks, dtype=np.float32)

        # Squeeze channel dim if [N, 1, H, W]
        if prob_maps.ndim == 4 and prob_maps.shape[1] == 1:
            prob_maps = prob_maps[:, 0]
        if gt_masks.ndim == 4 and gt_masks.shape[1] == 1:
            gt_masks = gt_masks[:, 0]

        if len(prob_maps) == 0:
            raise ValueError("calibrate() received an empty prob_maps array.")
        if prob_maps.shape != gt_masks.shape:
            raise ValueError(
                f"prob_maps shape {prob_maps.shape} does not match "
                f"gt_masks shape {gt_masks.shape}."
            )

        self.n_calibration = len(prob_maps)
        self.n_empty_masks = int(sum(
            1 for gt in gt_masks if gt.astype(bool).sum() == 0
        ))

        # For each lambda candidate, compute mean FNR across calibration set.
        # We sweep from low to high lambda (most aggressive to least
        # aggressive) and find the highest lambda that still satisfies the
        # mean FNR <= alpha constraint.
        mean_fnrs = np.array([
            np.mean([_image_fnr(p, g, lam) for p, g in zip(prob_maps, gt_masks)])
            for lam in _LAMBDA_GRID
        ])  # shape: [101]

        # Valid lambdas: those where mean FNR <= alpha
        valid_mask = mean_fnrs <= self.alpha
        if not valid_mask.any():
            # Even threshold=0.0 (predict everything positive) exceeds alpha.
            # This should not happen in practice since FNR(lambda=0) = 0,
            # but guard against it (e.g., all-empty calibration masks corner
            # case handled above, but this is an extra safety net).
            self.lambda_star = 0.0
            self.calibration_fnr = float(mean_fnrs[0])
        else:
            # Take the largest valid lambda (least aggressive threshold that
            # still meets the FNR guarantee). Higher threshold = more
            # conservative prediction = fewer false positives at the cost of
            # slightly more false negatives, while still within the budget.
            best_idx = int(np.where(valid_mask)[0].max())
            self.lambda_star = float(_LAMBDA_GRID[best_idx])
            self.calibration_fnr = float(mean_fnrs[best_idx])

        return self

    def predict(self, prob_map: np.ndarray) -> np.ndarray:
        """Apply the calibrated threshold to produce a binary mask.

        Args:
            prob_map: [H, W] or [1, H, W] or [B, 1, H, W] float probability
                map from a segmentation model (after sigmoid).

        Returns:
            Binary mask of same spatial shape, dtype uint8, with 1 = tumor,
            0 = background, under the guaranteed FNR <= alpha threshold.

        Raises:
            RuntimeError: If calibrate() has not been called yet.
        """
        if self.lambda_star is None:
            raise RuntimeError(
                "lambda_star is None: call calibrate() before predict()."
            )
        prob_map = np.asarray(prob_map, dtype=np.float32)
        return (prob_map >= self.lambda_star).astype(np.uint8)

    @property
    def coverage_guarantee(self) -> float | None:
        """Empirical FNR on calibration set (== 1 - sensitivity at lambda*).
        Returns None before calibrate() is called."""
        return self.calibration_fnr

    def summary(self) -> dict:
        """Return a dict of calibration metadata suitable for logging."""
        return {
            "alpha": self.alpha,
            "lambda_star": self.lambda_star,
            "calibration_fnr": self.calibration_fnr,
            "n_calibration": self.n_calibration,
            "n_empty_masks": self.n_empty_masks,
        }

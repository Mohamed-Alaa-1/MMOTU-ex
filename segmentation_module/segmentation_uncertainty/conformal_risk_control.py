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

    lambda* = max { lambda in Lambda : mean_FNR(lambda) <= alpha_corrected }

where Lambda is a fixed discrete grid [0.0, 0.01, ..., 1.0] (101 steps).
Images with no foreground pixels (empty masks) contribute FNR = 0 at all
thresholds and are included in the count.

Finite-sample correction (CRITICAL)
-------------------------------------
Per the CRC theorem (Angelopoulos & Bates 2022, Theorem 1), the corrected
risk level is:

    alpha_corrected = (n / (n+1)) * alpha

This ensures E[FNR on a fresh test image] <= alpha (formal guarantee).
For n >= 100, the slack is < 1%.  For small n, the guarantee becomes
more conservative.

CLARIFICATION on formal vs empirical guarantee
-----------------------------------------------
If the calibration set is too small (n < 30), the finite-sample
correction makes alpha_corrected significantly smaller than alpha, and
the calibrated lambda_star may be very conservative.  In that case,
avoid claiming a strict FNR bound; report as empirical calibration.

All inputs and outputs are numpy arrays (not tensors). No PyTorch dependency.
"""

from __future__ import annotations

import numpy as np


_LAMBDA_GRID: np.ndarray = np.linspace(0.0, 1.0, 101)
_MIN_N_FOR_FORMAL_GUARANTEE: int = 30


def _image_fnr(prob_map: np.ndarray, gt_mask: np.ndarray,
               threshold: float) -> float:
    """FNR for a single image at a given threshold."""
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

    Key distinction
    ---------------
    *Formal CRC guarantee*: uses the finite-sample corrected risk level
        alpha_corrected = (n / (n+1)) * alpha.
    The expected test FNR is then <= alpha (Angelopoulos & Bates 2022, Thm 1).

    *Empirical threshold calibration*: simply finds the largest lambda
    such that mean calibration FNR <= alpha (no formal guarantee).
    This is what many papers implicitly do, and should be clearly labeled.

    This implementation uses alpha_corrected, providing the formal guarantee
    when n is large enough.  The `formal_guarantee_valid` flag in summary()
    tells the caller whether to report as formal CRC or empirical calibration.

    Attributes:
        alpha: Target FNR bound (e.g., 0.10 for at most 10% FNR).
        alpha_corrected: Finite-sample corrected level = (n/(n+1))*alpha.
        lambda_star: CRC-calibrated threshold.
        lambda_star_standard: Standard threshold (always 0.5).
        calibration_fnr: Empirical FNR at lambda_star on calibration set.
        calibration_fnr_at_05: Empirical FNR at threshold=0.5 on calibration.
        n_calibration: Number of calibration images used.
        n_empty_masks: Calibration images with no foreground pixels.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(
                f"alpha must be in (0, 1); got {alpha}."
            )
        self.alpha = alpha
        self.alpha_corrected = None
        self.lambda_star = None
        self.lambda_star_standard: float = 0.5
        self.calibration_fnr = None
        self.calibration_fnr_at_05 = None
        self.n_calibration: int = 0
        self.n_empty_masks: int = 0

    def calibrate(
        self,
        prob_maps: np.ndarray,
        gt_masks: np.ndarray,
    ) -> "SegmentationConformalRiskController":
        """Fit lambda_star using finite-sample corrected CRC.

        Uses alpha_corrected = (n/(n+1)) * alpha, which provides the
        formal guarantee: E[test FNR] <= alpha (Angelopoulos & Bates 2022).

        Args:
            prob_maps: [N, H, W] or [N, 1, H, W] predicted probability maps.
            gt_masks:  Corresponding ground-truth masks, same shape, binary.

        Returns:
            self.
        """
        prob_maps = np.asarray(prob_maps, dtype=np.float32)
        gt_masks  = np.asarray(gt_masks,  dtype=np.float32)

        if prob_maps.ndim == 4 and prob_maps.shape[1] == 1:
            prob_maps = prob_maps[:, 0]
        if gt_masks.ndim == 4 and gt_masks.shape[1] == 1:
            gt_masks = gt_masks[:, 0]

        if len(prob_maps) == 0:
            raise ValueError("calibrate() received an empty prob_maps array.")
        if prob_maps.shape != gt_masks.shape:
            raise ValueError(
                f"Shape mismatch: prob_maps {prob_maps.shape} vs "
                f"gt_masks {gt_masks.shape}."
            )

        self.n_calibration = len(prob_maps)
        self.n_empty_masks = int(sum(
            1 for gt in gt_masks if gt.astype(bool).sum() == 0
        ))

        # Finite-sample corrected risk level
        n = self.n_calibration
        self.alpha_corrected = float((n / (n + 1)) * self.alpha)

        # Compute mean FNR for every threshold on the grid
        mean_fnrs = np.array([
            np.mean([
                _image_fnr(p, g, lam)
                for p, g in zip(prob_maps, gt_masks)
            ])
            for lam in _LAMBDA_GRID
        ])

        # FNR at standard 0.5 threshold (for reporting)
        idx_05 = int(np.argmin(np.abs(_LAMBDA_GRID - 0.5)))
        self.calibration_fnr_at_05 = float(mean_fnrs[idx_05])

        # CRC threshold: largest lambda where mean_FNR <= alpha_corrected
        valid_mask = mean_fnrs <= self.alpha_corrected
        if not valid_mask.any():
            self.lambda_star = 0.0
            self.calibration_fnr = float(mean_fnrs[0])
        else:
            best_idx = int(np.where(valid_mask)[0].max())
            self.lambda_star = float(_LAMBDA_GRID[best_idx])
            self.calibration_fnr = float(mean_fnrs[best_idx])

        return self

    def predict(self, prob_map: np.ndarray) -> np.ndarray:
        """Apply the CRC-calibrated threshold to produce a binary mask."""
        if self.lambda_star is None:
            raise RuntimeError(
                "lambda_star is None: call calibrate() before predict()."
            )
        prob_map = np.asarray(prob_map, dtype=np.float32)
        return (prob_map >= self.lambda_star).astype(np.uint8)

    @property
    def coverage_guarantee(self):
        """Empirical FNR on calibration set at lambda_star."""
        return self.calibration_fnr

    @property
    def formal_guarantee_valid(self) -> bool:
        """True if n >= 30, making the formal CRC bound meaningful."""
        return self.n_calibration >= _MIN_N_FOR_FORMAL_GUARANTEE

    def summary(self) -> dict:
        """Return calibration metadata for logging and reporting.

        Check `formal_guarantee_valid` to decide whether to report
        this as formal CRC or empirical threshold calibration.
        When False, do NOT claim a strict FNR bound of alpha.
        """
        return {
            "alpha": self.alpha,
            "alpha_corrected": self.alpha_corrected,
            "lambda_star_crc": self.lambda_star,
            "lambda_star_standard": self.lambda_star_standard,
            "calibration_fnr_at_crc_threshold": self.calibration_fnr,
            "calibration_fnr_at_05_threshold": self.calibration_fnr_at_05,
            "n_calibration": self.n_calibration,
            "n_empty_masks": self.n_empty_masks,
            "formal_guarantee_valid": self.formal_guarantee_valid,
            "note": (
                "Formal CRC guarantee (Angelopoulos & Bates 2022)."
                if self.formal_guarantee_valid
                else "Empirical threshold calibration only (n < 30); "
                     "do not claim a strict FNR bound."
            ),
        }

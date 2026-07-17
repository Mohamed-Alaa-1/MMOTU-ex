"""
segmentation/metrics.py

Standard segmentation evaluation metrics. All functions operate on
already-binarized numpy masks (bool or 0/1), not raw logits or
probabilities; threshold upstream (default 0.5 on sigmoid output).
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray, smooth: float = 1e-7) -> float:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + true.sum() + smooth))


def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray, smooth: float = 1e-7) -> float:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return float((intersection + smooth) / (union + smooth))


def _get_boundary(mask: np.ndarray) -> np.ndarray:
    """Binary boundary pixels via erosion difference."""
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    eroded = ndimage.binary_erosion(mask)
    return mask & ~eroded


def boundary_f1_score(pred_mask: np.ndarray, true_mask: np.ndarray, tolerance_px: int = 2) -> float:
    """Boundary F1 with a pixel tolerance: a predicted boundary pixel counts
    as a match if a true boundary pixel exists within tolerance_px, and
    vice versa (standard BF-score formulation)."""
    pred_boundary = _get_boundary(pred_mask)
    true_boundary = _get_boundary(true_mask)

    if pred_boundary.sum() == 0 and true_boundary.sum() == 0:
        return 1.0
    if pred_boundary.sum() == 0 or true_boundary.sum() == 0:
        return 0.0

    true_dist = ndimage.distance_transform_edt(~true_boundary)
    pred_dist = ndimage.distance_transform_edt(~pred_boundary)

    precision = float((true_dist[pred_boundary] <= tolerance_px).mean())
    recall = float((pred_dist[true_boundary] <= tolerance_px).mean())

    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def hausdorff_distance_95(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """95th percentile symmetric Hausdorff distance between boundary point
    sets, in pixels. Returns np.nan if either mask is empty (undefined)."""
    pred_boundary = _get_boundary(pred_mask)
    true_boundary = _get_boundary(true_mask)

    pred_pts = np.argwhere(pred_boundary)
    true_pts = np.argwhere(true_boundary)

    if len(pred_pts) == 0 or len(true_pts) == 0:
        return float("nan")

    # Symmetric HD95 via per-point nearest-neighbor distance distributions,
    # not scipy's directed_hausdorff (which returns only the single max, not
    # the 95th percentile). Build distance matrices via cKDTree for speed.
    from scipy.spatial import cKDTree

    tree_true = cKDTree(true_pts)
    tree_pred = cKDTree(pred_pts)

    d_pred_to_true, _ = tree_true.query(pred_pts)
    d_true_to_pred, _ = tree_pred.query(true_pts)

    all_dists = np.concatenate([d_pred_to_true, d_true_to_pred])
    return float(np.percentile(all_dists, 95))


def compute_all_segmentation_metrics(pred_mask: np.ndarray, true_mask: np.ndarray,
                                      boundary_tolerance_px: int = 2) -> dict:
    return {
        "dice": dice_score(pred_mask, true_mask),
        "iou": iou_score(pred_mask, true_mask),
        "boundary_f1": boundary_f1_score(pred_mask, true_mask, boundary_tolerance_px),
        "hd95": hausdorff_distance_95(pred_mask, true_mask),
    }

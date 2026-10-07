import math
from typing import List
import numpy as np

# Backwards compatible alias at bottom for existing imports.


def score_matrix(probs: np.ndarray, kind: str, k_reg: int = 5, lambda_reg: float = 0.01) -> np.ndarray:
    """Nonconformity score of EVERY candidate class, shape (N, K)."""
    if kind == "lac":
        return 1.0 - probs
    if kind in ("aps", "raps"):
        order = np.argsort(-probs, axis=1, kind="stable")
        cum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
        if kind == "raps":
            # Add regularization penalty for ranks > k_reg
            # rank is 1-indexed; penalty starts when rank > k_reg
            ranks = np.arange(1, probs.shape[1] + 1)[None, :]
            penalty = np.maximum(0, ranks - k_reg) * lambda_reg
            cum += penalty
        scores = np.empty_like(cum)
        np.put_along_axis(scores, order, cum, axis=1)
        return scores
    raise ValueError(f"Unknown score '{kind}'")


def conformal_rank(n: int, alpha: float) -> int:
    return math.ceil(round((n + 1) * (1.0 - alpha), 10))


def _qhat(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = conformal_rank(n, alpha)
    return float("inf") if (n == 0 or k > n) else float(np.sort(scores)[k - 1])


class MondrianConformalPredictor:
    """Class conditional split conformal prediction.
    Set(x) = {y : s(x, y) <= qhat_y}. Every class is tested independently, so the true class is
    retained with probability >= 1 - alpha conditional on Y = y whenever class y is not on fallback.
    """

    def __init__(self, alpha: float = 0.10, min_calibration_per_class: int = 10,
                 score: str = "lac", rare_class_policy: str = "global"):
        assert score in ("lac", "aps", "raps") and rare_class_policy in ("global", "inf")
        self.alpha, self.min_cal, self.score, self.policy = alpha, min_calibration_per_class, score, rare_class_policy
        self.class_qhat = None
        self.class_n = None
        self.global_qhat = float("inf")
        self.fallback_classes: List[int] = []

    def calibrate(self, cal_probs: np.ndarray, cal_labels: np.ndarray) -> None:
        K = cal_probs.shape[1]
        cal_labels = np.asarray(cal_labels, dtype=int)
        true_scores = score_matrix(cal_probs, self.score)[np.arange(len(cal_labels)), cal_labels]
        self.global_qhat = _qhat(true_scores, self.alpha)
        self.class_qhat = np.full(K, np.inf)
        self.class_n = np.zeros(K, dtype=int)
        self.fallback_classes = []
        for c in range(K):
            s = true_scores[cal_labels == c]
            self.class_n[c] = len(s)
            if len(s) < self.min_cal or conformal_rank(len(s), self.alpha) > len(s):
                self.fallback_classes.append(c)
                self.class_qhat[c] = self.global_qhat if self.policy == "global" else np.inf
            else:
                self.class_qhat[c] = _qhat(s, self.alpha)

    def predict_mask(self, probs: np.ndarray) -> np.ndarray:
        mask = score_matrix(probs, self.score) <= self.class_qhat[None, :]
        empty = ~mask.any(axis=1)
        if empty.any():                                   # union with top 1 never reduces coverage
            mask[empty, probs[empty].argmax(1)] = True
        return mask

    def predict_sets(self, probs: np.ndarray) -> List[List[int]]:
        return [np.flatnonzero(r).tolist() for r in self.predict_mask(probs)]


MondrianAPSConformalPredictor = MondrianConformalPredictor


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def evaluate_conformal_sets(pred_sets: List[List[int]], true_labels: np.ndarray, num_classes: int) -> dict:
    y = np.asarray(true_labels, dtype=int)
    covered = np.array([int(y[i]) in pred_sets[i] for i in range(len(y))])
    sizes = np.array([len(s) for s in pred_sets])
    pc_cov, pc_size, pc_n, pc_ci = {}, {}, {}, {}
    for c in range(num_classes):
        m = y == c
        if not m.any():
            continue
        pc_cov[c], pc_size[c], pc_n[c] = float(covered[m].mean()), float(sizes[m].mean()), int(m.sum())
        pc_ci[c] = wilson_interval(int(covered[m].sum()), int(m.sum()))
    return {"marginal_coverage": float(covered.mean()), "avg_set_size": float(sizes.mean()),
            "singleton_rate": float((sizes == 1).mean()), "per_class_coverage": pc_cov,
            "per_class_set_size": pc_size, "per_class_n": pc_n, "per_class_coverage_ci": pc_ci}

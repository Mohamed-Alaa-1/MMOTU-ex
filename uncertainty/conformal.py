"""
uncertainty/conformal.py

MondrianAPS and RAPS conformal predictors for multi-class classification.

Theory background
-----------------
Adaptive Prediction Sets (APS) — Angelopoulos et al. 2020.
  For each calibration sample (p, y):
      score(p, y) = cumulative probability when classes are sorted
                    descending by probability, up to and INCLUDING
                    the true class.
  At test time, include all classes up to (and including) the first
  class whose cumulative probability exceeds the class-specific
  threshold q_hat(y).

Mondrian / Class-Conditional split:
  For each class c separately, find the (1-alpha) quantile of the
  APS scores among calibration samples with true label = c.
  Use the global quantile as fallback if a class has fewer than
  min_calibration_per_class samples.

Finite-sample correction:
  For n calibration samples in class c, the quantile level is
      q_level = ceil((n+1)*(1-alpha)) / n,  clipped to 1.0.
  This guarantees (1-alpha) marginal coverage even for small n
  (Vovk et al., Tibshirani et al.).

RAPS — Regularized APS (Angelopoulos et al. 2021):
  Adds a penalty term  lambda * max(0, rank - k_reg)
  to each APS score, discouraging large prediction sets.
  lambda >= 0 (regularisation strength), k_reg (grace rank).

Step-by-step inclusion logic (predict_sets)
-------------------------------------------
Given test softmax probabilities p[0..K-1]:
  1. Sort classes in descending order of p -> order[].
  2. Compute cumulative sum of sorted probabilities -> L[rank].
  3. For each rank r (0, 1, 2, ...):
       class c   = order[r]
       threshold = class_thresholds.get(c, global_threshold)
       if L[r] <= threshold:   include class c
       else:                   stop
  4. If the resulting set is empty, add the top-1 class.

This ensures every set contains at least 1 class.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# APS score helper
# ---------------------------------------------------------------------------

def _aps_score(probs: np.ndarray, label: int) -> float:
    """Compute the APS non-conformity score for a single sample.

    The score is the cumulative probability mass up to and including
    the true class when classes are sorted by descending probability.

    Args:
        probs: 1-D array of softmax probabilities (must sum to ~1).
        label: Integer true class index.

    Returns:
        Scalar in (0, 1].
    """
    order = np.argsort(-probs)            # descending rank order
    sorted_probs = probs[order]           # probabilities in rank order
    cumulative = np.cumsum(sorted_probs)  # cumulative sums
    label_position = int(np.where(order == label)[0][0])
    return float(cumulative[label_position])


def _raps_score(probs: np.ndarray, label: int,
                lam: float = 0.01, k_reg: int = 5) -> float:
    """Compute the RAPS non-conformity score.

    RAPS adds a regularisation penalty to APS to discourage large
    prediction sets:  score = APS_score + lambda * max(0, rank - k_reg)
    where rank is the 0-indexed position of the true class (0 = top-1).

    Args:
        probs:  1-D softmax probabilities.
        label:  Integer true class index.
        lam:    Regularisation strength (>= 0).
        k_reg:  Grace rank. Classes ranked <= k_reg incur no penalty.

    Returns:
        Scalar >= 0.
    """
    order = np.argsort(-probs)
    label_position = int(np.where(order == label)[0][0])
    aps = _aps_score(probs, label)
    penalty = lam * max(0, label_position - k_reg)
    return float(aps + penalty)


# ---------------------------------------------------------------------------
# Mondrian APS conformal predictor
# ---------------------------------------------------------------------------

class MondrianAPSConformalPredictor:
    """Class-conditional (Mondrian) conformal predictor using APS scores.

    Per-class calibration is performed independently for each class,
    providing class-conditional coverage guarantees (one per class).
    Classes with fewer than `min_calibration_per_class` samples fall back
    to a global threshold computed over all calibration samples.

    Args:
        alpha: Miscoverage level; target coverage is (1 - alpha).
        min_calibration_per_class: Minimum samples required to compute a
            per-class threshold (default 10).
    """

    def __init__(self, alpha: float = 0.1,
                 min_calibration_per_class: int = 10) -> None:
        self.alpha = alpha
        self.min_calibration_per_class = min_calibration_per_class
        self.class_thresholds: Dict[int, float] = {}
        self.global_threshold: Optional[float] = None
        self.calibration_counts: Dict[int, int] = {}   # n per class
        self.fallback_classes: List[int] = []           # classes using global

    # ------------------------------------------------------------------ #
    @staticmethod
    def _aps_score(probs_row: np.ndarray, true_label: int) -> float:
        return _aps_score(probs_row, true_label)

    # ------------------------------------------------------------------ #
    def calibrate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> "MondrianAPSConformalPredictor":
        """Fit per-class thresholds from calibration data.

        Args:
            probs:  [N, K] float array of softmax probabilities.
            labels: [N] integer array of true class indices.

        Returns:
            self  (for method chaining).
        """
        probs  = np.asarray(probs,  dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        # --- Collect per-class APS scores ---
        scores_by_class: Dict[int, List[float]] = {}
        for probs_row, label in zip(probs, labels):
            scores_by_class.setdefault(int(label), []).append(
                self._aps_score(probs_row, int(label))
            )

        # --- Global fallback threshold (over ALL calibration samples) ---
        all_scores: List[float] = []
        for scores in scores_by_class.values():
            all_scores.extend(scores)

        n_all = len(all_scores)
        q_level_all = min(
            np.ceil((n_all + 1) * (1 - self.alpha)) / n_all, 1.0
        )
        self.global_threshold = float(
            np.quantile(all_scores, q_level_all, method="higher")
        )

        # --- Per-class thresholds with finite-sample correction ---
        self.calibration_counts = {}
        self.fallback_classes   = []
        self.class_thresholds   = {}

        for class_label, scores in sorted(scores_by_class.items()):
            n = len(scores)
            self.calibration_counts[class_label] = n

            if n < self.min_calibration_per_class:
                self.fallback_classes.append(class_label)
                logger.warning(
                    "Class %d has only %d calibration samples "
                    "(minimum=%d); using global threshold %.4f.",
                    class_label, n,
                    self.min_calibration_per_class,
                    self.global_threshold,
                )
                continue

            # Finite-sample corrected quantile level
            q_level = min(
                np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0
            )
            self.class_thresholds[class_label] = float(
                np.quantile(scores, q_level, method="higher")
            )

        logger.info(
            "Conformal calibration complete. "
            "alpha=%.3f  global_threshold=%.4f  "
            "per-class thresholds computed for %d/%d classes  "
            "fallback classes=%s",
            self.alpha,
            self.global_threshold,
            len(self.class_thresholds),
            len(scores_by_class),
            self.fallback_classes,
        )

        return self

    # ------------------------------------------------------------------ #
    def predict_sets(self, probs: np.ndarray) -> List[List[int]]:
        """Produce conformal prediction sets for a batch of samples.

        Step-by-step inclusion logic
        ----------------------------
        For each sample with softmax probabilities p[0..K-1]:
          1. Sort classes in descending order of p   ->  order[]
          2. Compute cumulative sums of sorted p      ->  L[rank]
          3. Walk ranks r = 0, 1, 2, ...:
               class c   = order[r]
               q_hat     = class_thresholds.get(c, global_threshold)
               if L[r] <= q_hat: include class c  (continue to next rank)
               else:             stop  (do NOT include c)
          4. Guarantee non-empty set: if set is empty, return [argmax(p)].

        Args:
            probs: [N, K] softmax probabilities.

        Returns:
            List of N lists, each containing the predicted class indices.
        """
        probs = np.asarray(probs, dtype=np.float64)
        prediction_sets: List[List[int]] = []

        for sample_probs in probs:
            order      = np.argsort(-sample_probs)   # descending
            sorted_p   = sample_probs[order]
            cumsum     = np.cumsum(sorted_p)
            sample_set: List[int] = []

            for rank, class_idx in enumerate(order):
                q_hat = self.class_thresholds.get(
                    int(class_idx), self.global_threshold
                )
                if cumsum[rank] <= q_hat:
                    sample_set.append(int(class_idx))
                else:
                    break

            if not sample_set:
                sample_set = [int(np.argmax(sample_probs))]

            prediction_sets.append(sample_set)

        return prediction_sets


# ---------------------------------------------------------------------------
# RAPS conformal predictor
# ---------------------------------------------------------------------------

class RAPSConformalPredictor:
    """Regularized Adaptive Prediction Sets (RAPS) conformal predictor.

    RAPS adds a regularisation penalty to APS scores to produce smaller
    prediction sets without sacrificing coverage.  Particularly useful
    for clinical applications where compact prediction sets improve utility.

    Reference: Angelopoulos et al. (2021) 'Uncertainty Sets for Image
    Classifiers using Conformal Prediction'.

    Args:
        alpha:  Miscoverage level.
        lam:    Regularisation strength (lambda >= 0). Larger values
                produce smaller sets. Typical starting value: 0.01.
        k_reg:  Grace rank. Classes ranked <= k_reg (0-indexed) incur
                no penalty. Typical values: 1..5.
    """

    def __init__(self, alpha: float = 0.1,
                 lam: float = 0.01, k_reg: int = 5) -> None:
        self.alpha  = alpha
        self.lam    = lam
        self.k_reg  = k_reg
        self.q_hat: Optional[float] = None

    def calibrate(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> "RAPSConformalPredictor":
        """Compute the global RAPS threshold from calibration data.

        Args:
            probs:  [N, K] softmax probabilities.
            labels: [N] true class indices.

        Returns:
            self.
        """
        probs  = np.asarray(probs,  dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        scores = np.array([
            _raps_score(p, int(y), lam=self.lam, k_reg=self.k_reg)
            for p, y in zip(probs, labels)
        ])

        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.q_hat = float(np.quantile(scores, q_level, method="higher"))

        logger.info(
            "RAPS calibration complete. alpha=%.3f  lam=%.4f  "
            "k_reg=%d  q_hat=%.4f",
            self.alpha, self.lam, self.k_reg, self.q_hat,
        )
        return self

    def predict_sets(self, probs: np.ndarray) -> List[List[int]]:
        """Produce RAPS conformal prediction sets.

        Inclusion rule: include class c at rank r if
            L[r] + lambda * max(0, r - k_reg) <= q_hat
        where L[r] is the cumulative probability up to rank r.

        Args:
            probs: [N, K] softmax probabilities.

        Returns:
            List of N lists of predicted class indices.
        """
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict_sets().")

        probs = np.asarray(probs, dtype=np.float64)
        prediction_sets: List[List[int]] = []

        for sample_probs in probs:
            order    = np.argsort(-sample_probs)
            sorted_p = sample_probs[order]
            cumsum   = np.cumsum(sorted_p)
            sample_set: List[int] = []

            for rank, class_idx in enumerate(order):
                regularised_score = cumsum[rank] + self.lam * max(
                    0, rank - self.k_reg
                )
                if regularised_score <= self.q_hat:
                    sample_set.append(int(class_idx))
                else:
                    break

            if not sample_set:
                sample_set = [int(np.argmax(sample_probs))]

            prediction_sets.append(sample_set)

        return prediction_sets


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> tuple:
    """Bootstrap 95% CI for the mean of `values`."""
    rng   = np.random.default_rng(seed=0)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lo = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(means, 100 * (1 - (1 - ci) / 2)))
    return lo, hi


def evaluate_conformal_sets(
    pred_sets,
    labels: np.ndarray,
    num_classes: int,
    n_boot: int = 2000,
) -> dict:
    """Compute marginal and per-class coverage with 95% bootstrap CIs.

    IMPORTANT — Calibration-set vs. test-set coverage
    --------------------------------------------------
    When called with the **calibration** data (same data used in `calibrate()`),
    APS/RAPS construction guarantees coverage >= (1 - alpha) by design. This
    means marginal_coverage will often be exactly 1.0 on calibration data,
    and marginal_coverage_ci = (1.0, 1.0). This is mathematically expected and
    correct — it is NOT evidence of perfect generalisation.

    For a valid, unbiased estimate of test-set coverage, always call this
    function with **held-out test** predictions (data NOT used in calibrate()).

    Args:
        pred_sets:   List of N prediction sets (each a list of class indices).
        labels:      [N] true class indices.
        num_classes: Total number of classes K.
        n_boot:      Bootstrap resamples for confidence intervals.

    Returns:
        dict with keys:
          marginal_coverage           float
          marginal_coverage_ci        (lo, hi) tuple
          avg_set_size                float
          singleton_rate              float
          per_class_coverage          {class: float}
          per_class_coverage_ci       {class: (lo, hi)}
          on_calibration_data_warning str or None
    """
    labels = np.asarray(labels)
    covered = np.array([labels[i] in pred_sets[i] for i in range(len(labels))],
                       dtype=float)

    marginal_coverage = float(covered.mean())
    marginal_ci       = _bootstrap_ci(covered, n_boot=n_boot)
    avg_set_size      = float(np.mean([len(s) for s in pred_sets])) if pred_sets else 0.0
    singleton_rate    = float(np.mean([len(s) == 1 for s in pred_sets])) if pred_sets else 0.0

    # Warn when coverage = 1.0 — likely called on calibration data
    calibration_warning: str | None = None
    if marginal_coverage >= 1.0:
        calibration_warning = (
            "marginal_coverage=1.0: This is expected when called on the "
            "calibration set (APS guarantees >=1-alpha coverage by construction). "
            "Call evaluate_conformal_sets() on held-out TEST data for an unbiased "
            "coverage estimate."
        )
        logger.warning(calibration_warning)

    per_class_coverage = {}
    per_class_coverage_ci = {}

    for class_label in range(num_classes):
        class_indices = np.where(labels == class_label)[0]
        if len(class_indices) == 0:
            per_class_coverage[class_label]    = float("nan")
            per_class_coverage_ci[class_label] = (float("nan"), float("nan"))
            continue

        class_covered = covered[class_indices]
        per_class_coverage[class_label]    = float(class_covered.mean())
        per_class_coverage_ci[class_label] = _bootstrap_ci(
            class_covered, n_boot=n_boot
        )

    return {
        "marginal_coverage":           marginal_coverage,
        "marginal_coverage_ci":        marginal_ci,
        "avg_set_size":                avg_set_size,
        "singleton_rate":              singleton_rate,
        "per_class_coverage":          per_class_coverage,
        "per_class_coverage_ci":       per_class_coverage_ci,
        "on_calibration_data_warning": calibration_warning,
    }

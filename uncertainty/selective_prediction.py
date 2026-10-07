import numpy as np


def coverage_at_risk(coverages: np.ndarray, risks: np.ndarray, target: float) -> float:
    ok = np.flatnonzero(risks <= target)
    return float(coverages[ok.max()]) if ok.size else 0.0


def risk_coverage_curve(probs: np.ndarray, labels: np.ndarray, uncertainty_scores: np.ndarray) -> dict:
    """Exact risk coverage curve over all n operating points; AURC = mean selective risk."""
    n = len(labels)
    errors = (probs.argmax(1) != labels).astype(np.float64)
    order = np.argsort(uncertainty_scores, kind="stable")          # most confident first
    k = np.arange(1, n + 1)
    risks = np.cumsum(errors[order]) / k
    coverages = k / n
    return {"coverages": coverages.tolist(), "risks": risks.tolist(), "aurc": float(risks.mean()),
            "coverage_at_5pct_risk": coverage_at_risk(coverages, risks, 0.05),
            "coverage_at_10pct_risk": coverage_at_risk(coverages, risks, 0.10)}

import numpy as np
import pytest
from uncertainty.conformal import MondrianConformalPredictor, evaluate_conformal_sets, conformal_rank


def _synthetic(n, K=8, seed=0):
    rng = np.random.RandomState(seed)
    z = rng.randn(n, K) * 2.0
    z[:, 0] += 1.5                                           # imbalanced classes
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    y = (p.cumsum(1) > rng.rand(n, 1)).argmax(1)             # labels drawn from p: perfectly calibrated
    return p, y


@pytest.mark.parametrize("score", ["lac", "aps", "raps"])
def test_marginal_and_class_conditional_coverage(score):
    alpha = 0.10
    cp_, cy = _synthetic(6000, seed=1)
    tp, ty = _synthetic(6000, seed=2)
    cp = MondrianConformalPredictor(alpha, 10, score)
    cp.calibrate(cp_, cy)
    ev = evaluate_conformal_sets(cp.predict_sets(tp), ty, 8)
    assert ev["marginal_coverage"] >= 1 - alpha - 0.02
    for c, n in ev["per_class_n"].items():
        if n >= 150:
            assert ev["per_class_coverage"][c] >= 1 - alpha - 0.06


def test_rank_and_rare_class_fallback():
    assert conformal_rank(9, 0.10) == 9 and conformal_rank(8, 0.10) == 9    # n=8 cannot reach 90%
    p, y = _synthetic(2000, seed=3)
    keep = np.concatenate([np.flatnonzero(y != 7), np.flatnonzero(y == 7)[:8]])
    cp = MondrianConformalPredictor(0.10, 10)
    cp.calibrate(p[keep], y[keep])
    assert 7 in cp.fallback_classes and cp.class_qhat[7] == cp.global_qhat

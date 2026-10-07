import numpy as np
import pandas as pd
import pytest
from data.splits import (create_patient_level_splits, create_kfold_splits, load_splits,
                         SPLIT_NAMES, DEFAULT_RATIOS)


def _meta(n_groups=700, seed=0):
    rng = np.random.RandomState(seed)
    p = np.array([.23, .15, .23, .06, .045, .18, .07, .035]); p /= p.sum()
    rows = []
    for g in range(n_groups):
        c = rng.choice(8, p=p)
        for j in range(rng.randint(1, 4)):
            rows.append({"patient_id": g, "image_path": f"{g}_{j}.jpg", "mask_path": None, "class_label": c})
    return pd.DataFrame(rows)


def test_no_group_leakage_and_all_splits_present():
    df = create_patient_level_splits(_meta(), strict=False)
    assert set(df["split"]) == set(SPLIT_NAMES)
    assert (df.groupby("patient_id")["split"].nunique() == 1).all()


def test_ratios_close_to_target():
    df = create_patient_level_splits(_meta(), strict=False)
    frac = df["split"].value_counts(normalize=True)
    for s, r in DEFAULT_RATIOS.items():
        assert abs(frac[s] - r) < 0.02


def test_deterministic():
    a = create_patient_level_splits(_meta(), random_state=1, strict=False)
    b = create_patient_level_splits(_meta(), random_state=1, strict=False)
    assert (a["split"].values == b["split"].values).all()


def test_kfold_tests_are_disjoint_and_complete():
    folds = create_kfold_splits(_meta(), n_splits=5, strict=False)
    tests = [set(f.loc[f.split == "test", "patient_id"]) for f in folds]
    assert sum(len(t) for t in tests) == _meta()["patient_id"].nunique()
    assert all(tests[i].isdisjoint(tests[j]) for i in range(5) for j in range(i + 1, 5))


def test_load_splits_rejects_legacy(tmp_path):
    df = create_patient_level_splits(_meta(), strict=False)
    df["split"] = df["split"].replace({"calib": "val"})
    p = tmp_path / "legacy.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_splits(str(p))

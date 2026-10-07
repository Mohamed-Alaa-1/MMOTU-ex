import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

SPLIT_NAMES: Tuple[str, ...] = ("train", "val", "calib", "test")
DEFAULT_RATIOS: Dict[str, float] = {"train": 0.60, "val": 0.10, "calib": 0.15, "test": 0.15}


def stratified_group_split(df: pd.DataFrame, ratios: Dict[str, float],
                           group_col: str = "patient_id", label_col: str = "class_label",
                           seed: int = 42) -> pd.DataFrame:
    """Greedy stratified group split with arbitrary ratios.

    Whole groups are assigned to the split with the largest remaining per class
    deficit, processing large groups first. Groups never straddle splits.
    """
    names = list(ratios)
    r = np.array([ratios[n] for n in names], dtype=float)
    if not np.isclose(r.sum(), 1.0):
        raise ValueError(f"Ratios must sum to 1, got {r.sum():.4f}")

    n_classes = int(df[label_col].max()) + 1
    groups, counts = [], []
    for gid, labels in df.groupby(group_col, sort=True)[label_col]:
        groups.append(gid)
        counts.append(np.bincount(labels.to_numpy(dtype=int), minlength=n_classes))
    counts = np.stack(counts).astype(float)                      # (G, C)

    class_totals = counts.sum(0)                                  # (C,)
    target = r[:, None] * class_totals[None, :]                   # (S, C)
    weight = 1.0 / np.maximum(class_totals, 1.0) ** 2             # rare classes count equally

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(groups))
    order = order[np.argsort(-counts[order].sum(1), kind="stable")]

    current = np.zeros_like(target)
    assignment = {}
    for i in order:
        delta = (((current + counts[i] - target) ** 2 - (current - target) ** 2) * weight).sum(1)
        s = rng.choice(np.flatnonzero(np.isclose(delta, delta.min())))
        current[s] += counts[i]
        assignment[groups[i]] = names[s]

    out = df.copy()
    out["split"] = out[group_col].map(assignment)
    return out


def verify_splits(df: pd.DataFrame, group_col: str = "patient_id", label_col: str = "class_label",
                  min_calib_per_class: int = 10, strict: bool = True) -> pd.DataFrame:
    problems: List[str] = []
    missing = set(SPLIT_NAMES) - set(df["split"].unique())
    if missing:
        problems.append(f"missing splits: {sorted(missing)}")

    leaked = df.groupby(group_col)["split"].nunique()
    if (leaked > 1).any():
        problems.append(f"{int((leaked > 1).sum())} groups appear in more than one split")

    ct = pd.crosstab(df["split"], df[label_col])
    for s in set(SPLIT_NAMES) & set(ct.index):
        absent = ct.columns[ct.loc[s] == 0].tolist()
        if absent:
            problems.append(f"split '{s}' has no samples for classes {absent}")

    if df[group_col].nunique() == len(df):
        logger.warning("Every image is its own group: this split is effectively IMAGE level. "
                       "Supply real patient identifiers before claiming patient level separation.")

    if "calib" in ct.index:
        low = ct.loc["calib"][ct.loc["calib"] < min_calib_per_class]
        if len(low):
            logger.warning(f"Calib classes below {min_calib_per_class} samples "
                           f"(will use global quantile fallback): {low.to_dict()}")

    for p in problems:
        (logger.error if strict else logger.warning)(p)
    if problems and strict:
        raise ValueError("Split verification failed: " + "; ".join(problems))
    return ct


def create_patient_level_splits(metadata_df: pd.DataFrame, ratios: Optional[Dict[str, float]] = None,
                                random_state: int = 42, group_col: str = "patient_id",
                                label_col: str = "class_label", strict: bool = True) -> pd.DataFrame:
    ratios = dict(ratios or DEFAULT_RATIOS)
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"ratios must define exactly {SPLIT_NAMES}")
    df = metadata_df.reset_index(drop=True)
    out = stratified_group_split(df, ratios, group_col, label_col, random_state)
    ct = verify_splits(out, group_col, label_col, strict=strict)
    logger.info("Class distribution across splits:\n%s", ct.to_string())
    return out


def create_kfold_splits(metadata_df: pd.DataFrame, n_splits: int = 5,
                        ratios: Optional[Dict[str, float]] = None, random_state: int = 42,
                        group_col: str = "patient_id", label_col: str = "class_label",
                        strict: bool = True) -> List[pd.DataFrame]:
    """Outer fold = test. Remainder is split train/val/calib with the configured relative ratios."""
    ratios = dict(ratios or DEFAULT_RATIOS)
    inner = {k: ratios[k] for k in ("train", "val", "calib")}
    z = sum(inner.values())
    inner = {k: v / z for k, v in inner.items()}

    df = metadata_df.reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for fold, (rest_idx, test_idx) in enumerate(sgkf.split(df, df[label_col], df[group_col])):
        rest = stratified_group_split(df.iloc[rest_idx], inner, group_col, label_col, random_state + fold)
        out = df.copy()
        out["fold"] = fold
        out["split"] = "test"
        out.loc[rest.index, "split"] = rest["split"]
        verify_splits(out, group_col, label_col, strict=strict)
        folds.append(out)
    return folds


def load_splits(splits_csv: str):
    """Returns (train_df, val_df, calib_df, test_df). Rejects legacy 3 way split files."""
    df = pd.read_csv(splits_csv)
    present = set(df["split"].unique())
    if present != set(SPLIT_NAMES):
        raise ValueError(f"{splits_csv} contains splits {sorted(present)}, expected {list(SPLIT_NAMES)}. "
                         f"Regenerate with Stage 1 (legacy 3 way file must be archived).")
    return tuple(df[df["split"] == s].reset_index(drop=True) for s in SPLIT_NAMES)


def save_split_manifest(df: pd.DataFrame, path: Path, seed: int, ratios: Dict[str, float],
                        group_col: str = "patient_id", label_col: str = "class_label") -> dict:
    h = hashlib.sha256()
    for row in df.sort_values("image_path")[["image_path", "split"]].itertuples(index=False):
        h.update(f"{row.image_path}|{row.split}\n".encode())
    ct = pd.crosstab(df["split"], df[label_col])
    manifest = {
        "sha256": h.hexdigest(), "seed": seed, "ratios": dict(ratios),
        "n_images": int(len(df)), "n_groups": int(df[group_col].nunique()),
        "counts": {s: int((df["split"] == s).sum()) for s in SPLIT_NAMES},
        "class_counts": {s: {str(c): int(v) for c, v in row.items()} for s, row in ct.iterrows()},
    }
    Path(path).write_text(json.dumps(manifest, indent=2))
    return manifest

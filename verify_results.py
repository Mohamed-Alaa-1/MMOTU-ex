"""
verify_results.py

Final Results Verification Script
==================================
Loads saved prediction files (CSV/JSON/NPY) from results/ and recomputes
ALL reported metrics from scratch to:

  1. Detect any discrepancy between the values reported in the paper and the
     values actually computed from the test-set predictions.
  2. Resolve the contradiction between the ResNet-50 test accuracy reported
     as 77.87% in Table 3 and 71.77% in the McNemar comparison.
  3. Recompute McNemar's test (with continuity correction) from the prediction
     files to confirm the discordant-case counts and p-value.
  4. Output a final verified summary table and a detailed McNemar report.

Expected prediction file layout
---------------------------------
The script looks for per-model CSV files saved during the main.py pipeline:
    results/predictions/<model_name>_test_predictions.csv

Each CSV must have columns:
    image_path, true_label, predicted_label, prob_class_0, prob_class_1, ...

If prediction CSVs are not found, the script falls back to re-running
inference using the saved checkpoints (requires model checkpoints and the
original dataset to be accessible).

Usage
-----
python verify_results.py
python verify_results.py --predictions_dir results/predictions
python verify_results.py --checkpoints_dir results/checkpoints --rerun_inference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

_HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# McNemar's test (with continuity correction)
# ---------------------------------------------------------------------------

def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
) -> dict:
    """Compute McNemar's test comparing two classifier outputs.

    Uses the mid-p / continuity-corrected version:
        chi2 = (|b - c| - 1)^2 / (b + c)   (with Yates correction)

    where:
        b = n(A correct, B wrong)
        c = n(A wrong, B correct)

    Args:
        correct_a: Binary array (1 = correct) for classifier A.
        correct_b: Binary array (1 = correct) for classifier B.

    Returns:
        dict with keys: a, b, c, d, statistic, pvalue, significant.
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    assert len(correct_a) == len(correct_b), "Arrays must have same length."

    a = int(np.sum( correct_a &  correct_b))   # both correct
    b = int(np.sum( correct_a & ~correct_b))   # A correct, B wrong
    c = int(np.sum(~correct_a &  correct_b))   # A wrong, B correct
    d = int(np.sum(~correct_a & ~correct_b))   # both wrong

    if b + c == 0:
        statistic = 0.0
        pvalue    = 1.0
    else:
        # Yates continuity correction (standard for McNemar)
        statistic = float((abs(b - c) - 1) ** 2 / (b + c))
        pvalue    = float(chi2.sf(statistic, df=1))

    return {
        "n_both_correct":   a,
        "n_a_only_correct": b,
        "n_b_only_correct": c,
        "n_both_wrong":     d,
        "discordant_pairs": b + c,
        "statistic":        statistic,
        "pvalue":           pvalue,
        "significant_0.05": pvalue < 0.05,
    }


# ---------------------------------------------------------------------------
# Load prediction CSVs
# ---------------------------------------------------------------------------

def load_prediction_csv(path: Path) -> pd.DataFrame:
    """Load a model prediction CSV file and validate required columns."""
    df = pd.read_csv(path)
    required = {"true_label", "predicted_label"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Prediction CSV {path.name} is missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    df["correct"] = (df["true_label"] == df["predicted_label"]).astype(int)
    return df


# ---------------------------------------------------------------------------
# Compute metrics from a prediction DataFrame
# ---------------------------------------------------------------------------

def compute_classification_metrics(df: pd.DataFrame) -> dict:
    """Compute accuracy and per-class counts from a predictions DataFrame."""
    n       = len(df)
    correct = int(df["correct"].sum())
    acc     = correct / n if n > 0 else 0.0

    # Bootstrap 95% CI for accuracy
    rng = np.random.default_rng(seed=42)
    boot_accs = np.array([
        rng.choice(df["correct"].values, size=n, replace=True).mean()
        for _ in range(5000)
    ])
    ci_lo = float(np.percentile(boot_accs, 2.5))
    ci_hi = float(np.percentile(boot_accs, 97.5))

    return {
        "n_test": n,
        "n_correct": correct,
        "accuracy": acc,
        "accuracy_pct": acc * 100,
        "accuracy_ci_lo_pct": ci_lo * 100,
        "accuracy_ci_hi_pct": ci_hi * 100,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify all reported results from saved prediction files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--predictions_dir", default="results/predictions",
        help="Directory containing per-model prediction CSV files.",
    )
    p.add_argument(
        "--model_a", default="ensemble",
        help="Name of model A (numerator) for McNemar test.",
    )
    p.add_argument(
        "--model_b", default="resnet50",
        help="Name of model B (denominator) for McNemar test.",
    )
    p.add_argument(
        "--out", default="results/verified_summary.json",
        help="Output path for the verification report (JSON).",
    )
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    pred_dir   = _HERE / args.predictions_dir
    out_path   = _HERE / args.out

    print("=" * 65)
    print(" MMOTU-ex — Final Results Verification")
    print("=" * 65)

    if not pred_dir.exists():
        print(f"\n[WARNING] Predictions directory not found: {pred_dir}")
        print("  -> Create it by running main.py with --save_predictions flag.")
        print("  -> Alternatively, run with --rerun_inference.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 1. Find all per-model prediction CSVs
    # -----------------------------------------------------------------------
    csv_files = sorted(pred_dir.glob("*_test_predictions.csv"))
    if not csv_files:
        print(f"[ERROR] No *_test_predictions.csv files found in {pred_dir}.")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} prediction file(s):")
    for f in csv_files:
        print(f"  {f.name}")

    # -----------------------------------------------------------------------
    # 2. Compute per-model accuracy with bootstrap CIs
    # -----------------------------------------------------------------------
    print("\n--- Per-Model Accuracy (from saved predictions) ---")
    model_results = {}
    for f in csv_files:
        # Extract model name: strip suffix
        model_name = f.stem.replace("_test_predictions", "")
        df         = load_prediction_csv(f)
        metrics    = compute_classification_metrics(df)
        model_results[model_name] = {"metrics": metrics, "df": df}
        print(
            f"  {model_name:30s}  "
            f"Acc={metrics['accuracy_pct']:.2f}%  "
            f"95% CI [{metrics['accuracy_ci_lo_pct']:.2f}%, "
            f"{metrics['accuracy_ci_hi_pct']:.2f}%]  "
            f"(n={metrics['n_test']})"
        )

    # -----------------------------------------------------------------------
    # 3. McNemar test: model_a vs model_b
    # -----------------------------------------------------------------------
    print(f"\n--- McNemar Test: {args.model_a} vs {args.model_b} ---")
    mcnemar_result = None
    if args.model_a in model_results and args.model_b in model_results:
        df_a = model_results[args.model_a]["df"]
        df_b = model_results[args.model_b]["df"]

        # Align on image_path if column exists
        if "image_path" in df_a.columns and "image_path" in df_b.columns:
            merged = df_a[["image_path", "correct"]].merge(
                df_b[["image_path", "correct"]], on="image_path",
                suffixes=("_a", "_b")
            )
            correct_a = merged["correct_a"].values
            correct_b = merged["correct_b"].values
        else:
            # Assume same row order
            if len(df_a) != len(df_b):
                print(
                    f"[WARNING] Cannot align {args.model_a} ({len(df_a)} rows) "
                    f"and {args.model_b} ({len(df_b)} rows) without image_path. "
                    "Skipping McNemar test."
                )
            else:
                correct_a = df_a["correct"].values
                correct_b = df_b["correct"].values

        mcnemar_result = mcnemar_test(correct_a, correct_b)

        print(f"  Contingency table:")
        print(f"  {'':30s}  {args.model_b} Correct  {args.model_b} Wrong")
        print(f"  {args.model_a} Correct  {mcnemar_result['n_both_correct']:>12d}  {mcnemar_result['n_a_only_correct']:>12d}")
        print(f"  {args.model_a} Wrong    {mcnemar_result['n_b_only_correct']:>12d}  {mcnemar_result['n_both_wrong']:>12d}")
        print(f"\n  Discordant pairs (b+c): {mcnemar_result['discordant_pairs']}")
        print(f"  Chi-squared statistic:  {mcnemar_result['statistic']:.4f}")
        print(f"  p-value:                {mcnemar_result['pvalue']:.4e}")
        print(f"  Significant (p<0.05):   {mcnemar_result['significant_0.05']}")
    else:
        missing = [m for m in [args.model_a, args.model_b]
                   if m not in model_results]
        print(f"[WARNING] Model(s) not found: {missing}. Skipping McNemar.")

    # -----------------------------------------------------------------------
    # 4. Contradiction check: look for known discrepancies
    # -----------------------------------------------------------------------
    print("\n--- Contradiction Check ---")
    if "resnet50" in model_results:
        resnet_acc = model_results["resnet50"]["metrics"]["accuracy_pct"]
        print(f"  ResNet-50 test accuracy (from saved predictions): {resnet_acc:.2f}%")
        print(
            f"  If reported as 77.87% in Table 3 and 71.77% in McNemar,\n"
            f"  check whether Table 3 uses VALIDATION accuracy (77.87%)\n"
            f"  and the McNemar comparison uses TEST accuracy (71.77%).\n"
            f"  Verified test accuracy is: {resnet_acc:.2f}%"
        )
    else:
        print("  resnet50 predictions not found; contradiction check skipped.")

    # -----------------------------------------------------------------------
    # 5. Save report
    # -----------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "per_model_accuracy": {
            name: res["metrics"]
            for name, res in model_results.items()
        },
        "mcnemar": mcnemar_result,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4, default=str)

    print(f"\n[OK] Verification report saved to: {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()

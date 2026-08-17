"""
evaluation/published_baselines_comparison.py

Assemble the final segmentation comparison table:
  - Our trained models (LAURA ensemble, U-Net, Attention U-Net, DeepLabV3+, U-Net++)
    loaded from their evaluation_report.json files.
  - Published MMOTU baselines from Zhao et al. 2022 (arXiv:2207.06799, Table 2).

Usage
-----
# Minimum (LAURA report only — fills other cells as "Not run"):
python evaluation/published_baselines_comparison.py

# Full comparison after all baselines have been evaluated:
python evaluation/published_baselines_comparison.py ^
    --laura   results/evaluation/segmentation/evaluation_report.json ^
    --unet    results/evaluation/unet/evaluation_report.json ^
    --attunet results/evaluation/attunet/evaluation_report.json ^
    --deeplab results/evaluation/deeplab/evaluation_report.json ^
    --unetpp  results/evaluation/unetpp/evaluation_report.json ^
    --out     results/segmentation_comparison_table.csv

Output
------
Prints a formatted table and saves a CSV at --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Published results — Zhao et al. 2022 (arXiv:2207.06799) Table 2, OTU_2D
# "-" means not reported in the original paper.
# NOTE: These are on the OTU_2D subset. Direct comparison is valid only when
# your patient-level split matches theirs. If splits differ, note:
# "approximate comparison with published results."
# ---------------------------------------------------------------------------
PUBLISHED = [
    {
        "model":     "U-Net (Zhao et al. 2022)",
        "source":    "Zhao et al. 2022, Table 2",
        "dice":      "0.7231",
        "dice_ci":   "—",
        "iou":       "0.6015",
        "iou_ci":    "—",
        "recall":    "—",
        "precision": "—",
        "params_M":  "31.04",
        "fps":       "—",
        "seeds":     "1",
    },
    {
        "model":     "Att-UNet (Zhao et al. 2022)",
        "source":    "Zhao et al. 2022, Table 2",
        "dice":      "0.7389",
        "dice_ci":   "—",
        "iou":       "0.6198",
        "iou_ci":    "—",
        "recall":    "—",
        "precision": "—",
        "params_M":  "34.88",
        "fps":       "—",
        "seeds":     "1",
    },
    {
        "model":     "TransUNet (Zhao et al. 2022)",
        "source":    "Zhao et al. 2022, Table 2",
        "dice":      "0.7102",
        "dice_ci":   "—",
        "iou":       "0.5934",
        "iou_ci":    "—",
        "recall":    "—",
        "precision": "—",
        "params_M":  "105.28",
        "fps":       "—",
        "seeds":     "1",
    },
    {
        "model":     "SwinUNet (Zhao et al. 2022)",
        "source":    "Zhao et al. 2022, Table 2",
        "dice":      "0.7156",
        "dice_ci":   "—",
        "iou":       "0.5987",
        "iou_ci":    "—",
        "recall":    "—",
        "precision": "—",
        "params_M":  "27.14",
        "fps":       "—",
        "seeds":     "1",
    },
    {
        "model":     "DeepLabV3+ (Zhao et al. 2022)",
        "source":    "Zhao et al. 2022, Table 2",
        "dice":      "0.7198",
        "dice_ci":   "—",
        "iou":       "0.5965",
        "iou_ci":    "—",
        "recall":    "—",
        "precision": "—",
        "params_M":  "39.76",
        "fps":       "—",
        "seeds":     "1",
    },
]

COLUMNS = [
    "model", "source", "dice", "dice_ci", "iou", "iou_ci",
    "recall", "precision", "params_M", "fps", "seeds",
]

COL_WIDTHS = {
    "model":     30,
    "source":    26,
    "dice":       6,
    "dice_ci":   18,
    "iou":        6,
    "iou_ci":    18,
    "recall":     7,
    "precision":  9,
    "params_M":   8,
    "fps":        6,
    "seeds":      5,
}

COL_HEADERS = {
    "model":     "Model",
    "source":    "Source",
    "dice":      "Dice",
    "dice_ci":   "Dice 95% CI",
    "iou":       "IoU",
    "iou_ci":    "IoU 95% CI",
    "recall":    "Recall",
    "precision": "Precision",
    "params_M":  "Params M",
    "fps":       "FPS",
    "seeds":     "Seeds",
}


def _load_report(path: str | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    if not p.exists():
        print(f"  [WARNING] Report not found: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def _fmt_ci(ci: list | None) -> str:
    if ci is None or len(ci) < 2:
        return "—"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _fmt_val(v, decimals: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def report_to_row(report: dict, label: str, source: str, seeds: int) -> dict:
    """Convert an evaluation_report.json dict to a comparison table row."""
    dice     = report.get("test_mean_standard_dice")
    dice_ci  = report.get("test_mean_standard_dice_ci")
    iou      = report.get("test_mean_standard_iou")
    iou_ci   = report.get("test_mean_standard_iou_ci")
    recall   = report.get("test_mean_standard_recall")
    prec     = report.get("test_mean_standard_precision")
    fps      = report.get("inference_fps")

    # Parameter count — sum across all model_X entries if ensemble
    params_info = report.get("model_params", {})
    if params_info:
        total_m = sum(v.get("total_params_M", 0) for v in params_info.values())
        params_str = f"{total_m:.2f}"
    else:
        params_str = "—"

    return {
        "model":     label,
        "source":    source,
        "dice":      _fmt_val(dice),
        "dice_ci":   _fmt_ci(dice_ci),
        "iou":       _fmt_val(iou),
        "iou_ci":    _fmt_ci(iou_ci),
        "recall":    _fmt_val(recall),
        "precision": _fmt_val(prec),
        "params_M":  params_str,
        "fps":       _fmt_val(fps, decimals=1),
        "seeds":     str(seeds),
    }


def _not_run(label: str) -> dict:
    return {
        "model":     label,
        "source":    "This work",
        "dice":      "not run",
        "dice_ci":   "—",
        "iou":       "—",
        "iou_ci":    "—",
        "recall":    "—",
        "precision": "—",
        "params_M":  "—",
        "fps":       "—",
        "seeds":     "—",
    }


def _print_table(rows: list[dict]) -> None:
    # Header
    header = "  ".join(
        COL_HEADERS[c].ljust(COL_WIDTHS[c]) for c in COLUMNS
    )
    sep = "  ".join("-" * COL_WIDTHS[c] for c in COLUMNS)
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(
            str(row.get(c, "—")).ljust(COL_WIDTHS[c]) for c in COLUMNS
        )
        print(line)


def _save_csv(rows: list[dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print segmentation comparison table (published + ours).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--laura",   default=None,
                   help="evaluation_report.json for LAURA ensemble.")
    p.add_argument("--laura2",  default=None,
                   help="evaluation_report.json for LAURA seed-2 (if run).")
    p.add_argument("--unet",    default=None,
                   help="evaluation_report.json for U-Net.")
    p.add_argument("--unet2",   default=None,
                   help="evaluation_report.json for U-Net seed-2 (if run).")
    p.add_argument("--attunet", default=None,
                   help="evaluation_report.json for Attention U-Net.")
    p.add_argument("--deeplab", default=None,
                   help="evaluation_report.json for DeepLabV3+.")
    p.add_argument("--unetpp",  default=None,
                   help="evaluation_report.json for U-Net++.")
    p.add_argument("--out",     default="results/segmentation_comparison_table.csv",
                   help="Output CSV path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 100)
    print("  Segmentation Comparison — LAURA vs Baselines vs Published MMOTU Results")
    print("=" * 100)
    print()
    print("  NOTE: Published values are from Zhao et al. 2022 (arXiv:2207.06799), Table 2 (OTU_2D subset).")
    print("  Direct comparison is valid only when the patient-level split matches.")
    print("  If splits differ, report as 'approximate comparison with published results'.")
    print()

    rows: list[dict] = []

    # ── Published baselines (always shown) ──────────────────────────────────
    for b in PUBLISHED:
        rows.append(b)

    # ── Separator row ────────────────────────────────────────────────────────
    rows.append({c: "---" for c in COLUMNS})

    # ── Our models ──────────────────────────────────────────────────────────
    # LAURA ensemble
    laura_report = _load_report(args.laura)
    if laura_report is not None:
        rows.append(report_to_row(laura_report,
                                  label="LAURA (Ensemble, seed 42)",
                                  source="This work", seeds=1))
    else:
        rows.append(_not_run("LAURA (Ensemble)"))

    # LAURA seed 2
    if args.laura2:
        r = _load_report(args.laura2)
        if r:
            rows.append(report_to_row(r,
                                      label="LAURA (Ensemble, seed 2)",
                                      source="This work", seeds=2))

    # U-Net
    unet_report = _load_report(args.unet)
    if unet_report is not None:
        rows.append(report_to_row(unet_report,
                                  label="U-Net (seed 42)",
                                  source="This work", seeds=1))
    else:
        rows.append(_not_run("U-Net (seed 42)"))

    if args.unet2:
        r = _load_report(args.unet2)
        if r:
            rows.append(report_to_row(r,
                                      label="U-Net (seed 2)",
                                      source="This work", seeds=2))

    # Attention U-Net
    attunet_report = _load_report(args.attunet)
    if attunet_report is not None:
        rows.append(report_to_row(attunet_report,
                                  label="Attention U-Net (seed 42)",
                                  source="This work", seeds=1))
    else:
        rows.append(_not_run("Attention U-Net (seed 42)"))

    # DeepLabV3+
    deeplab_report = _load_report(args.deeplab)
    if deeplab_report is not None:
        rows.append(report_to_row(deeplab_report,
                                  label="DeepLabV3+ (seed 42)",
                                  source="This work", seeds=1))
    else:
        rows.append(_not_run("DeepLabV3+ (seed 42)"))

    # U-Net++
    unetpp_report = _load_report(args.unetpp)
    if unetpp_report is not None:
        rows.append(report_to_row(unetpp_report,
                                  label="U-Net++ (seed 42)",
                                  source="This work", seeds=1))
    else:
        rows.append(_not_run("U-Net++ (seed 42)"))

    _print_table(rows)

    out_path = _PROJECT_ROOT / args.out
    _save_csv(rows, out_path)
    print()
    print(f"  Table saved -> {out_path}")


if __name__ == "__main__":
    main()

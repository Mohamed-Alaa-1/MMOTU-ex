"""
run_overlay_scan.py

Standalone script: scan the entire MMOTU training corpus for images likely
to contain burned-in caliper crosshairs, measurement text, or other UI
overlays, using detect_overlay_risk() from segmentation/dataset.py.

This is the highest-priority open item from segmentation_extension_plan.md
Section 7:

    "Run scan_dataset_for_overlay_risk() against the actual results/splits.csv
    image corpus. This is now the single highest-priority open item: if
    burned-in calipers turn out to be widespread across the real MMOTU
    training corpus (not just this one illustrative sample), that is a finding
    relevant to the existing classification and XAI work too, not only this
    segmentation extension, and would need addressing before any further
    training on this data."

Run from the project root (f:\\GitHub repos\\MMOTU-ex):

    python segmentation_module/run_overlay_scan.py

Or with a custom splits CSV or output path:

    python segmentation_module/run_overlay_scan.py \\
        --splits results/splits.csv \\
        --output results/overlay_risk_report.csv \\
        --top_n 20

The script resolves image paths from splits.csv relative to the project root,
since that is how paths are stored in splits.csv (e.g., data/raw/OTU_2d/...).

CPU-only, no GPU or model weights required. Processes ~1,469 JPEG images;
expected runtime 2-5 minutes on a CPU-only development machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make segmentation_module's packages importable when run as a script from
# the project root.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pandas as pd

from segmentation.dataset import detect_overlay_risk

PROJECT_ROOT = _HERE.parent


def resolve_path(p: str) -> Path:
    """Resolve a path that may be relative to the project root."""
    candidate = Path(p)
    if candidate.is_absolute():
        return candidate
    # Try relative to cwd first, then relative to project root
    if candidate.exists():
        return candidate.resolve()
    proj_candidate = PROJECT_ROOT / p
    if proj_candidate.exists():
        return proj_candidate
    return candidate.resolve()  # return even if not found; error will surface later


def run_scan(splits_csv: str, output_csv: str, top_n: int = 20) -> pd.DataFrame:
    splits_path = resolve_path(splits_csv)
    if not splits_path.exists():
        print(f"ERROR: splits CSV not found at {splits_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(splits_path)
    print(f"Loaded {len(df)} image records from {splits_path}")
    print(f"Columns: {df.columns.tolist()}")

    if "image_path" not in df.columns:
        print("ERROR: splits CSV must have an 'image_path' column.", file=sys.stderr)
        sys.exit(1)

    results = []
    n_errors = 0

    for idx, row in df.iterrows():
        raw_path = row["image_path"]
        img_path = resolve_path(raw_path)

        # Progress indicator every 100 images
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} images ...")

        try:
            result = detect_overlay_risk(str(img_path))
            result["split"] = row.get("split", None)
            result["class_label"] = row.get("class_label", None)
            results.append(result)
        except Exception as e:
            n_errors += 1
            results.append({
                "image_path": str(img_path),
                "flagged_pixel_fraction": float("nan"),
                "likely_overlay": False,
                "max_channel_divergence": -1,
                "mean_channel_divergence": float("nan"),
                "split": row.get("split", None),
                "class_label": row.get("class_label", None),
                "error": str(e),
            })

    report_df = pd.DataFrame(results).sort_values(
        "flagged_pixel_fraction", ascending=False
    ).reset_index(drop=True)

    # Save the full report
    output_path = resolve_path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)

    # Print summary
    n_flagged = int(report_df["likely_overlay"].sum())
    n_total = len(report_df) - n_errors
    print(f"\n{'='*60}")
    print(f"OVERLAY RISK SCAN SUMMARY")
    print(f"{'='*60}")
    print(f"Total images scanned : {n_total}")
    print(f"Errors (file missing) : {n_errors}")
    print(f"Flagged as likely overlay : {n_flagged} ({100.0 * n_flagged / max(1, n_total):.1f}%)")

    # Per-split breakdown
    if "split" in report_df.columns and report_df["split"].notna().any():
        print("\nPer-split breakdown:")
        for split_name, group in report_df.groupby("split", dropna=True):
            n_split_flagged = int(group["likely_overlay"].sum())
            print(f"  {split_name:8s}: {n_split_flagged}/{len(group)} flagged "
                  f"({100.0 * n_split_flagged / len(group):.1f}%)")

    print(f"\nFull report saved to: {output_path}")

    if n_flagged > 0:
        print(f"\nTop {min(top_n, n_flagged)} most suspicious images:")
        top_flagged = report_df[report_df["likely_overlay"]].head(top_n)
        for _, r in top_flagged.iterrows():
            print(
                f"  {Path(r['image_path']).name:15s}  "
                f"flagged={r['flagged_pixel_fraction']:.4f}  "
                f"max_divergence={int(r['max_channel_divergence'])}  "
                f"split={r.get('split', 'N/A')}"
            )
    else:
        print("\nNo images flagged: dataset appears to have no systematic "
              "burned-in overlays at the calibrated threshold (40/255 max "
              "channel divergence, >0.05% of pixels).")

    return report_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan MMOTU dataset for burned-in overlay risk."
    )
    parser.add_argument(
        "--splits",
        default="results/splits.csv",
        help="Path to splits CSV (default: results/splits.csv, relative to project root).",
    )
    parser.add_argument(
        "--output",
        default="results/overlay_risk_report.csv",
        help="Output CSV path for the full report.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top-suspicious images to print in the summary.",
    )
    args = parser.parse_args()
    run_scan(args.splits, args.output, args.top_n)


if __name__ == "__main__":
    main()

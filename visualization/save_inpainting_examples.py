"""
visualization/save_inpainting_examples.py

Save representative before/after caliper inpainting image pairs.

Purpose
-------
Provides visual proof that the inpainting step removes caliper overlays
while preserving the underlying tumour boundary and relevant anatomical
tissue, supporting the validity of the ablation study.

Generated outputs (all in --out_dir):
  inpainting_examples_grid.pdf   — NxM grid of (raw | inpainted | diff) trios
  inpainting_examples_grid.png   — PNG copy for embedding in documents
  inpainting_examples.csv        — Per-image pixel change statistics:
      image_path, changed_pixels, changed_fraction, max_abs_diff

Usage
-----
    python visualization/save_inpainting_examples.py \\
        --splits results/splits.csv \\
        --n_examples 12 \\
        --out_dir results/figures/inpainting_examples

Optional: target specific images
    python visualization/save_inpainting_examples.py \\
        --splits results/splits.csv \\
        --indices 5 14 27 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

_HERE         = Path(__file__).resolve().parent   # visualization/
_PROJECT_ROOT = _HERE.parent                       # project root

for _p in [str(_HERE), str(_PROJECT_ROOT),
           str(_PROJECT_ROOT / "segmentation_module")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from segmentation.dataset import MMOTUSegmentationDataset

# ── colour scheme ──────────────────────────────────────────────────────────
BLUE_DARK  = "#003366"
BLUE_MED   = "#3366CC"
BLUE_LIGHT = "#6699FF"
BLUE_PALE  = "#99CCFF"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_sample(ds_raw, ds_clean, idx: int):
    """Return (raw_np, clean_np) grayscale float32 images in [0, 1]."""
    img_raw,   _ = ds_raw[idx]
    img_clean, _ = ds_clean[idx]
    # Tensors are [C, H, W]; take first channel
    raw   = img_raw[0].numpy().astype(np.float32)
    clean = img_clean[0].numpy().astype(np.float32)
    return raw, clean


def _changed_pixels_stats(raw: np.ndarray, clean: np.ndarray) -> dict:
    diff  = np.abs(raw - clean)
    mask  = diff > 1e-3   # pixels changed beyond floating-point noise
    return {
        "changed_pixels":   int(mask.sum()),
        "changed_fraction": float(mask.mean()),
        "max_abs_diff":     float(diff.max()),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_grid(
    pairs: list,
    out_path: Path,
    indices: list,
) -> None:
    """Save a 3-column grid: Raw | Inpainted | Difference."""
    n = len(pairs)
    fig = plt.figure(figsize=(3 * 3.2, n * 3.0 + 0.6))
    fig.suptitle(
        "Caliper Inpainting: Before | After | |Difference|\n"
        "(verify that tumour boundary is preserved)",
        fontsize=10, color=BLUE_DARK, fontweight="bold",
        y=1.002,
    )
    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.35, wspace=0.08)

    col_titles = ["Raw (calipers visible)", "Inpainted (clean)", "|Difference|"]
    for j, ct in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(ct, color=BLUE_DARK, pad=3)

    for row, (raw, clean, idx) in enumerate(pairs):
        diff  = np.abs(raw - clean)
        # Scale diff for visibility
        diff_vis = diff / (diff.max() + 1e-8)
        stats = _changed_pixels_stats(raw, clean)

        for col, (img, cmap, vmin, vmax) in enumerate([
            (raw,      "gray",    0, 1),
            (clean,    "gray",    0, 1),
            (diff_vis, BLUE_PALE, 0, 1),
        ]):
            ax = fig.add_subplot(gs[row, col])
            if col < 2:
                ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                ax.imshow(img, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(
                    f"Image #{idx}\n"
                    f"Δ={stats['changed_fraction']*100:.1f}%",
                    fontsize=7, color=BLUE_DARK,
                )

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grid saved → {out_path.with_suffix('.pdf').name}")
    print(f"  Grid saved → {out_path.with_suffix('.png').name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Save before/after caliper inpainting examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--splits", default="results/splits.csv")
    p.add_argument("--split",  default="test",
                   choices=["train", "val", "test"],
                   help="Which split to draw examples from.")
    p.add_argument("--n_examples", type=int, default=12,
                   help="Number of example images (ignored if --indices set).")
    p.add_argument("--indices", type=int, nargs="*", default=None,
                   help="Specific dataset indices to visualise.")
    p.add_argument("--out_dir", default="results/figures/inpainting_examples")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_df = pd.read_csv(_PROJECT_ROOT / args.splits)
    split_df  = splits_df[splits_df["split"] == args.split].reset_index(drop=True)

    print(f"Loaded {len(split_df)} images from '{args.split}' split.")

    # Build two datasets: raw (calipers visible) and clean (inpainted)
    ds_raw   = MMOTUSegmentationDataset(
        split_df, image_size=256, apply_inpainting=False, augment=False
    )
    ds_clean = MMOTUSegmentationDataset(
        split_df, image_size=256, apply_inpainting=True,  augment=False
    )

    # Choose indices
    if args.indices:
        indices = [i for i in args.indices if 0 <= i < len(ds_raw)]
    else:
        n  = min(args.n_examples, len(ds_raw))
        # Spread evenly across dataset to get diverse examples
        indices = [int(i) for i in np.linspace(0, len(ds_raw) - 1, n)]

    print(f"Generating {len(indices)} example pairs…")

    # Collect pairs and stats
    pairs      = []
    stats_rows = []
    for idx in indices:
        raw, clean = _load_sample(ds_raw, ds_clean, idx)
        stats = _changed_pixels_stats(raw, clean)
        stats["image_index"] = idx
        if "image_path" in split_df.columns:
            stats["image_path"] = split_df.iloc[idx]["image_path"]
        pairs.append((raw, clean, idx))
        stats_rows.append(stats)

    # Save grid
    save_grid(pairs, out_dir / "inpainting_examples_grid", indices)

    # Save stats CSV
    stats_df = pd.DataFrame(stats_rows)
    csv_path = out_dir / "inpainting_examples.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"  Stats CSV saved → {csv_path.name}")

    # Summary
    print("\n--- Inpainting Change Statistics ---")
    print(f"  Mean changed fraction : {stats_df['changed_fraction'].mean()*100:.2f}%")
    print(f"  Max changed fraction  : {stats_df['changed_fraction'].max()*100:.2f}%")
    print(f"  Mean max |diff|       : {stats_df['max_abs_diff'].mean():.4f}")
    print(
        "\n  [CHECK] If changed_fraction is small (< 5%) for most images,\n"
        "  the inpainting is local (calipers only) and does not corrupt\n"
        "  the tumour region. Inspect the PDF grid to verify visually."
    )
    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()

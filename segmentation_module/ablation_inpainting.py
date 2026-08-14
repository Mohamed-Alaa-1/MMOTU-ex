"""
segmentation_module/ablation_inpainting.py

Caliper Shortcut Ablation Study
================================
Evaluates two models against two test sets to expose the caliper shortcut:

  Model A — "Inpainted"  : trained with   inpainting (apply_inpainting=True)
  Model B — "Shortcut"   : trained without inpainting (apply_inpainting=False)

  Test set "clean" : images passed through inpaint_overlay_pixels at inference
  Test set "raw"   : images served as-is (calipers visible)

The 2×2 matrix that appears in the paper:

              | Test on RAW (calipers) | Test on CLEAN (inpainted)
  Model A     |  moderate Dice          |  best Dice (real learning)
  Model B     |  inflated Dice          |  ★ FAIL    (shortcut exposed)

The script also generates:
  • ablation_2x2_table.csv         — raw numbers
  • ablation_bar_chart.pdf         — side-by-side bar chart (blue palette)
  • ablation_qualitative_grid.pdf  — per-image prediction panels for each cell

Usage
-----
python segmentation_module/ablation_inpainting.py \\
    --model_a results/checkpoints/segmentation/laura_small_best.pt \\
    --model_b results/checkpoints/segmentation/laura_small_shortcut_best.pt \\
    --splits  results/splits.csv

Optional:
    --n_qual  4        # images per cell in the qualitative grid (default 4)
    --out_dir results/evaluation/ablation
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent   # segmentation_module/
_PROJECT_ROOT = _HERE.parent

for _p in [str(_HERE), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from segmentation.models.auravit_config import LAURA_BASE, LAURA_SMALL, LAURA_TINY
from segmentation.models.lightweight_auravit import LightweightAuraViT
from segmentation.dataset import MMOTUSegmentationDataset
from segmentation.metrics import dice_score, iou_score

# ── palette ──────────────────────────────────────────────────────────────────
BLUE_DARK   = "#003366"
BLUE_MED    = "#3366CC"
BLUE_LIGHT  = "#6699FF"
BLUE_PALE   = "#99CCFF"

_BLUE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "mmotu_blue", ["white", BLUE_PALE, BLUE_LIGHT, BLUE_MED, BLUE_DARK]
)

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = LightweightAuraViT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {path.name}  ({n:.2f}M params)")
    return model


@torch.no_grad()
def _evaluate(model, dataset, device, desc="Evaluating"):
    """Run model over the dataset and return per-image dice and iou arrays."""
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    dices, ious = [], []
    for imgs, masks in tqdm(loader, desc=desc, leave=False):
        logits = model(imgs.to(device))
        probs  = torch.sigmoid(logits).cpu().numpy()   # [B,1,H,W]
        masks_np = masks.numpy()                        # [B,1,H,W]
        for i in range(len(imgs)):
            pred = probs[i, 0] >= 0.5
            gt   = masks_np[i, 0] >= 0.5
            dices.append(dice_score(pred, gt))
            ious.append(iou_score(pred, gt))
    return np.array(dices), np.array(ious)


def _get_sample_predictions(model, dataset, device, n=4, indices=None):
    """
    Returns lists of (image_np, gt_np, pred_np) tuples for n images.
    indices: if given, use those specific dataset indices; else use first n.
    """
    if indices is None:
        indices = list(range(min(n, len(dataset))))
    results = []
    for idx in indices:
        img_t, mask_t = dataset[idx]
        with torch.no_grad():
            logit = model(img_t.unsqueeze(0).to(device))
            prob  = torch.sigmoid(logit)[0, 0].cpu().numpy()
        img_np  = (img_t[0].numpy() * 255).clip(0, 255).astype(np.uint8)
        gt_np   = (mask_t[0].numpy() >= 0.5)
        pred_np = (prob >= 0.5)
        results.append((img_np, gt_np, pred_np))
    return results


# ── bar chart ────────────────────────────────────────────────────────────────

def _plot_bar_chart(matrix, out_path):
    """
    matrix: {
        ("Model A Inpainted", "Clean (inpainted)"): dice_value,
        ("Model A Inpainted", "Raw (calipers)"):    dice_value,
        ("Model B Shortcut",  "Clean (inpainted)"): dice_value,
        ("Model B Shortcut",  "Raw (calipers)"):    dice_value,
    }
    """
    models   = ["Model A\n(Inpainted)", "Model B\n(Shortcut)"]
    test_sets = ["Clean (inpainted)", "Raw (calipers)"]
    colors    = [BLUE_MED, BLUE_LIGHT]    # test-set colours
    linestyles = ["-", "--"]              # printed as hatches
    hatches    = ["", "///"]

    x      = np.arange(len(models))
    width  = 0.32
    offset = [-width / 2, width / 2]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for j, (ts, color, hatch) in enumerate(zip(test_sets, colors, hatches)):
        vals = [matrix[("Model A Inpainted", ts)],
                matrix[("Model B Shortcut",  ts)]]
        bars = ax.bar(x + offset[j], vals, width,
                      label=ts, color=color,
                      hatch=hatch, edgecolor=BLUE_DARK, linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{v:.3f}",
                    ha="center", va="bottom",
                    fontsize=8.5, color=BLUE_DARK, fontweight="bold")

    # Annotate the "★ FAIL" cell
    fail_val = matrix[("Model B Shortcut", "Clean (inpainted)")]
    fail_x   = x[1] + offset[0]
    ax.annotate(
        "★ Shortcut\n  exposed",
        xy=(fail_x, fail_val),
        xytext=(fail_x - 0.18, fail_val + 0.07),
        fontsize=8, color="#CC0000", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#CC0000", lw=1.3),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Mean Dice Score", color=BLUE_DARK)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Caliper Shortcut Ablation Study\n"
        "Inpainted vs Shortcut Model on Clean and Raw Test Sets",
        color=BLUE_DARK,
    )
    ax.legend(title="Test Set", loc="upper right", framealpha=0.8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart saved → {out_path.name}")


# ── qualitative grid ─────────────────────────────────────────────────────────

def _plot_qualitative_grid(cells, out_path):
    """
    cells: list of dicts with keys:
        label, samples: list of (img_np, gt_np, pred_np)
    Produces an N-column grid where each row is one cell.
    """
    n_cols = max(len(c["samples"]) for c in cells)
    n_rows = len(cells)          # one row per (model × test-set) combination
    # each cell has 3 sub-rows: image, gt overlay, pred overlay
    fig, axes = plt.subplots(
        n_rows * 3, n_cols,
        figsize=(n_cols * 2.6, n_rows * 7.2),
        squeeze=False,
    )
    fig.suptitle(
        "Qualitative Ablation: Prediction Quality per Cell",
        fontsize=12, color=BLUE_DARK, fontweight="bold",
    )

    for row_i, cell in enumerate(cells):
        rb = row_i * 3
        for col_i, (img_np, gt_np, pred_np) in enumerate(cell["samples"]):
            for r in range(3):
                axes[rb + r][col_i].axis("off")

            # sub-row 0: original US
            axes[rb + 0][col_i].imshow(img_np, cmap="gray", vmin=0, vmax=255)
            if col_i == 0:
                axes[rb + 0][col_i].set_ylabel(
                    cell["label"], fontsize=8, color=BLUE_DARK,
                    rotation=90, labelpad=4,
                )
            if row_i == 0:
                axes[rb + 0][col_i].set_title(f"Sample {col_i+1}", fontsize=8)

            # sub-row 1: GT contour
            axes[rb + 1][col_i].imshow(img_np, cmap="gray", vmin=0, vmax=255)
            if gt_np.sum() > 0:
                axes[rb + 1][col_i].contour(
                    gt_np.astype(float), levels=[0.5],
                    colors=[BLUE_MED], linestyles=["solid"], linewidths=[1.5],
                )

            # sub-row 2: Prediction contour
            axes[rb + 2][col_i].imshow(img_np, cmap="gray", vmin=0, vmax=255)
            if pred_np.sum() > 0:
                axes[rb + 2][col_i].contour(
                    pred_np.astype(float), levels=[0.5],
                    colors=[BLUE_DARK], linestyles=["dashed"], linewidths=[1.5],
                )

        # Label unused columns
        for col_i in range(len(cell["samples"]), n_cols):
            for r in range(3):
                axes[rb + r][col_i].axis("off")

    # Row labels on the left margin
    row_labels = [c["label"] for c in cells]
    for row_i, lbl in enumerate(row_labels):
        rb = row_i * 3 + 1
        axes[rb][0].set_ylabel(lbl, fontsize=8, color=BLUE_DARK,
                               rotation=90, labelpad=6)

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=BLUE_MED,  linewidth=1.5, linestyle="solid",  label="Ground Truth"),
        Line2D([0], [0], color=BLUE_DARK, linewidth=1.5, linestyle="dashed", label="Prediction"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, fontsize=9, framealpha=0.8,
               bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Qualitative grid saved → {out_path.name}")


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Caliper Shortcut Ablation Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_a", required=True,
                   help="Checkpoint for Model A (trained WITH inpainting).")
    p.add_argument("--model_b", required=True,
                   help="Checkpoint for Model B (trained WITHOUT inpainting).")
    p.add_argument("--splits", default="results/splits.csv")
    p.add_argument("--out_dir", default="results/evaluation/ablation")
    p.add_argument("--n_qual", type=int, default=4,
                   help="Images per cell in the qualitative grid.")
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" Caliper Shortcut Ablation Study ")
    print("=" * 60)

    # 1. Load both models ------------------------------------------------
    print("\nLoading models…")
    model_a = _load_model(args.model_a, device)   # inpainted
    model_b = _load_model(args.model_b, device)   # shortcut

    # 2. Build four datasets (2 models × 2 test sets) --------------------
    splits_df = pd.read_csv(_PROJECT_ROOT / args.splits)
    test_df   = splits_df[splits_df["split"] == "test"].reset_index(drop=True)

    print(f"\nBuilding test datasets ({len(test_df)} images each)…")
    ds_clean = MMOTUSegmentationDataset(
        test_df, image_size=256, apply_inpainting=True,  augment=False
    )
    ds_raw = MMOTUSegmentationDataset(
        test_df, image_size=256, apply_inpainting=False, augment=False
    )

    # 3. Evaluate all four cells -----------------------------------------
    print("\nRunning 2×2 ablation matrix…")

    print("  [1/4] Model A  on  Clean (inpainted)…")
    dice_a_clean, iou_a_clean = _evaluate(model_a, ds_clean, device,
                                           "A × Clean")

    print("  [2/4] Model A  on  Raw   (calipers)…")
    dice_a_raw,   iou_a_raw   = _evaluate(model_a, ds_raw,   device,
                                           "A × Raw")

    print("  [3/4] Model B  on  Clean (inpainted)…")
    dice_b_clean, iou_b_clean = _evaluate(model_b, ds_clean, device,
                                           "B × Clean")

    print("  [4/4] Model B  on  Raw   (calipers)…")
    dice_b_raw,   iou_b_raw   = _evaluate(model_b, ds_raw,   device,
                                           "B × Raw")

    # 4. Assemble results ------------------------------------------------
    matrix = {
        ("Model A Inpainted", "Clean (inpainted)"): float(dice_a_clean.mean()),
        ("Model A Inpainted", "Raw (calipers)"):    float(dice_a_raw.mean()),
        ("Model B Shortcut",  "Clean (inpainted)"): float(dice_b_clean.mean()),
        ("Model B Shortcut",  "Raw (calipers)"):    float(dice_b_raw.mean()),
    }
    iou_matrix = {
        ("Model A Inpainted", "Clean (inpainted)"): float(iou_a_clean.mean()),
        ("Model A Inpainted", "Raw (calipers)"):    float(iou_a_raw.mean()),
        ("Model B Shortcut",  "Clean (inpainted)"): float(iou_b_clean.mean()),
        ("Model B Shortcut",  "Raw (calipers)"):    float(iou_b_raw.mean()),
    }

    shortcut_inflation = (
        matrix[("Model B Shortcut",  "Raw (calipers)")]
      - matrix[("Model B Shortcut",  "Clean (inpainted)")]
    )
    inpainting_gain = (
        matrix[("Model A Inpainted", "Clean (inpainted)")]
      - matrix[("Model B Shortcut",  "Clean (inpainted)")]
    )

    # 5. Print summary ---------------------------------------------------
    print("\n" + "=" * 60)
    print(" RESULTS: 2×2 Ablation Matrix (Mean Dice) ")
    print("=" * 60)
    print(f"{'':30s} {'Clean':>12s}  {'Raw':>12s}")
    print("-" * 58)
    print(f"{'Model A — Inpainted':30s} "
          f"{matrix[('Model A Inpainted','Clean (inpainted)')]:>12.4f}  "
          f"{matrix[('Model A Inpainted','Raw (calipers)')]:>12.4f}")
    print(f"{'Model B — Shortcut (no inpainting)':30s} "
          f"{matrix[('Model B Shortcut','Clean (inpainted)')]:>12.4f}  "
          f"{matrix[('Model B Shortcut','Raw (calipers)')]:>12.4f}")
    print("-" * 58)
    print(f"\n★  Shortcut inflation  (B_raw − B_clean) : {shortcut_inflation:+.4f}")
    print(f"★  Inpainting gain     (A_clean − B_clean): {inpainting_gain:+.4f}")

    # 6. Save CSV --------------------------------------------------------
    rows = []
    for (model_lbl, test_lbl), dice_val in matrix.items():
        rows.append({
            "model":    model_lbl,
            "test_set": test_lbl,
            "mean_dice": dice_val,
            "mean_iou":  iou_matrix[(model_lbl, test_lbl)],
        })
    csv_path = out_dir / "ablation_2x2_table.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  CSV saved → {csv_path.name}")

    # 7. Bar chart -------------------------------------------------------
    _plot_bar_chart(matrix, out_dir / "ablation_bar_chart.pdf")

    # 8. Qualitative grid ------------------------------------------------
    print(f"\nSampling {args.n_qual} images per cell for qualitative grid…")
    # Use the same fixed indices across all cells for a fair visual comparison
    n_q     = min(args.n_qual, len(ds_clean))
    indices = list(range(n_q))

    cells = [
        {
            "label":   "Model A (Inpainted)\n× Clean Test Set",
            "samples": _get_sample_predictions(model_a, ds_clean, device,
                                               n=n_q, indices=indices),
        },
        {
            "label":   "Model A (Inpainted)\n× Raw Test Set",
            "samples": _get_sample_predictions(model_a, ds_raw,   device,
                                               n=n_q, indices=indices),
        },
        {
            "label":   "Model B (Shortcut)\n× Clean Test Set  ← FAIL",
            "samples": _get_sample_predictions(model_b, ds_clean, device,
                                               n=n_q, indices=indices),
        },
        {
            "label":   "Model B (Shortcut)\n× Raw Test Set",
            "samples": _get_sample_predictions(model_b, ds_raw,   device,
                                               n=n_q, indices=indices),
        },
    ]
    _plot_qualitative_grid(cells, out_dir / "ablation_qualitative_grid.pdf")

    print("\n" + "=" * 60)
    print(f" Done. All outputs in: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

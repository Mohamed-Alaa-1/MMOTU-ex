"""
segmentation_module/visualize_uncertainty.py

Generates publication-ready PDF figures visualising the Deep Ensemble's
epistemic uncertainty on the MMOTU ovarian tumour test set.

For each image in the sample a 1×4 panel is produced:
  Col 1 – Original ultrasound (grayscale, after inpainting)
  Col 2 – Ground truth mask  (solid #3366CC contour overlay)
  Col 3 – Ensemble prediction (dashed #003366 contour overlay)
  Col 4 – Epistemic uncertainty heat-map (custom blue gradient colourmap)

All outputs are saved as PDF to results/figures/segmentation/uncertainty/.

Usage
-----
# Visualise the 9 most uncertain test images (default):
python segmentation_module/visualize_uncertainty.py \\
    --checkpoints results/checkpoints/segmentation/laura_small_best.pt \\
                  results/checkpoints/segmentation/laura_base_model2_best.pt

# Pick a specific number of images or select the most confident instead:
python segmentation_module/visualize_uncertainty.py \\
    --checkpoints ... \\
    --n_images 6 --select most_uncertain

python segmentation_module/visualize_uncertainty.py \\
    --checkpoints ... \\
    --n_images 6 --select most_confident
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── project paths ────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent   # segmentation_module/
_PROJECT_ROOT = _HERE.parent

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from segmentation.models.auravit_config import LAURA_BASE, LAURA_SMALL, LAURA_TINY
from segmentation.models.lightweight_auravit import LightweightAuraViT
from segmentation.dataset import MMOTUSegmentationDataset
from segmentation_uncertainty.ensemble_segmentation import DeepEnsembleSegmentationEstimator
from segmentation_uncertainty.selective_segmentation import image_level_uncertainty

# ── palette (monochromatic blue, project-wide convention) ───────────────────
BLUE_DARK   = "#003366"
BLUE_MED    = "#3366CC"
BLUE_LIGHT  = "#6699FF"
BLUE_PALE   = "#99CCFF"

# Custom blue colourmap: white → pale → light → medium → dark
_BLUE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "mmotu_blue",
    ["white", BLUE_PALE, BLUE_LIGHT, BLUE_MED, BLUE_DARK],
)

# Matplotlib global style
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       9,
    "axes.titlesize":  10,
    "axes.titleweight":"bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":      150,
})


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_models(checkpoint_paths, device):
    """Reconstruct LightweightAuraViT models from their .pt checkpoints."""
    models = []
    arch_map = {"LAURA_BASE": LAURA_BASE, "LAURA_SMALL": LAURA_SMALL,
                "LAURA_TINY": LAURA_TINY}
    for p in checkpoint_paths:
        path = Path(p)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg  = ckpt["config"]          # stored as model.cf by trainer
        model = LightweightAuraViT(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        models.append(model)
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Loaded {path.name}  ({n:.2f}M params)")
    return models


def _contour_overlay(ax, image_gray, mask_binary, color, linestyle, label):
    """Draw a single-class contour over a grayscale background on `ax`."""
    ax.imshow(image_gray, cmap="gray", vmin=0, vmax=255)
    # contour expects a 2-D array; binary values 0/1 → one level at 0.5
    if mask_binary.sum() > 0:
        ax.contour(
            mask_binary.astype(float),
            levels=[0.5],
            colors=[color],
            linestyles=[linestyle],
            linewidths=[1.5],
        )
    ax.set_title(label)
    ax.axis("off")


def _build_figure(img_np, gt_np, pred_np, unc_np, title, conformal_lambda):
    """Create and return a 1×4 figure for one test image."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    fig.suptitle(title, fontsize=10, color=BLUE_DARK, fontweight="bold", y=1.01)

    # --- Col 1: original image ---
    axes[0].imshow(img_np, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Ultrasound (inpainted)")
    axes[0].axis("off")

    # --- Col 2: ground truth ---
    _contour_overlay(axes[1], img_np, gt_np, color=BLUE_MED,
                     linestyle="solid", label="Ground Truth")

    # --- Col 3: ensemble prediction (at conformal threshold) ---
    _contour_overlay(axes[2], img_np, pred_np, color=BLUE_DARK,
                     linestyle="dashed",
                     label=f"Prediction (λ={conformal_lambda:.2f})")

    # --- Col 4: epistemic uncertainty heat-map ---
    im = axes[3].imshow(unc_np, cmap=_BLUE_CMAP, vmin=0)
    axes[3].set_title("Epistemic Uncertainty")
    axes[3].axis("off")
    cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label("Ensemble Std", fontsize=8, color=BLUE_DARK)
    cbar.ax.yaxis.set_tick_params(color=BLUE_DARK, labelsize=7)

    fig.tight_layout()
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise Deep-Ensemble epistemic uncertainty on MMOTU test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Paths to two (or more) segmentation model checkpoints.")
    p.add_argument("--splits", default="results/splits.csv")
    p.add_argument("--out_dir", default="results/figures/segmentation/uncertainty")
    p.add_argument("--n_images", type=int, default=9,
                   help="Number of sample images to visualise.")
    p.add_argument("--select", choices=["most_uncertain", "most_confident", "random"],
                   default="most_uncertain",
                   help="Which images to pick from the test set.")
    p.add_argument("--conformal_lambda", type=float, default=0.44,
                   help="Conformal threshold used for prediction contours.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" MMOTU – Epistemic Uncertainty Visualisation ")
    print("=" * 60)

    # 1. Load ensemble ---------------------------------------------------
    print("\nLoading models…")
    models   = _load_models(args.checkpoints, device)
    ensemble = DeepEnsembleSegmentationEstimator(models, device)

    # 2. Load test set ---------------------------------------------------
    splits_df = pd.read_csv(_PROJECT_ROOT / args.splits)
    test_df   = splits_df[splits_df["split"] == "test"].reset_index(drop=True)
    dataset   = MMOTUSegmentationDataset(
        test_df, image_size=256, apply_inpainting=True, augment=False
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    print(f"\nRunning ensemble inference on {len(dataset)} test images…")
    all_imgs, all_masks, all_probs, all_std = [], [], [], []

    for imgs, masks in tqdm(loader):
        est = ensemble.predict(imgs.to(device))
        all_imgs.append(imgs.cpu().numpy())
        all_masks.append(masks.cpu().numpy())
        all_probs.append(est["mean_probs"])     # already numpy
        all_std.append(est["epistemic_std"])

    all_imgs  = np.concatenate(all_imgs,  axis=0)   # [N,1,256,256]
    all_masks = np.concatenate(all_masks, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_std   = np.concatenate(all_std,   axis=0)

    # Scalar uncertainty per image (mean epistemic_std over spatial dims)
    img_uncertainty = all_std.mean(axis=(1, 2, 3))  # [N]

    # 3. Select images ---------------------------------------------------
    np.random.seed(args.seed)
    n = min(args.n_images, len(dataset))
    if args.select == "most_uncertain":
        indices = np.argsort(img_uncertainty)[::-1][:n]
        subset_label = "most uncertain"
    elif args.select == "most_confident":
        indices = np.argsort(img_uncertainty)[:n]
        subset_label = "most confident"
    else:
        indices = np.random.choice(len(dataset), size=n, replace=False)
        subset_label = "random"
    print(f"\nSelected {n} {subset_label} images.")

    # 4. Generate one PDF per image -------------------------------------
    saved = []
    for rank, idx in enumerate(indices, start=1):
        img_np  = (all_imgs[idx, 0] * 255).clip(0, 255).astype(np.uint8)
        gt_np   = (all_masks[idx, 0] >= 0.5)
        pred_np = (all_probs[idx, 0] >= args.conformal_lambda)
        unc_np  = all_std[idx, 0]

        title = (
            f"Sample {rank}/{n}  |  "
            f"Unc={img_uncertainty[idx]:.4f}  |  "
            f"GT pixels={gt_np.sum()}  |  "
            f"Pred pixels={pred_np.sum()}"
        )
        fig = _build_figure(img_np, gt_np, pred_np, unc_np,
                            title, args.conformal_lambda)

        out_path = out_dir / f"uncertainty_{rank:02d}_idx{idx}.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    # 5. Also save a combined summary grid (all N panels, PDF) ----------
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig_summary, axes_grid = plt.subplots(
        rows * 4, cols,
        figsize=(cols * 3.8, rows * 3.8),
        squeeze=False,
    )
    fig_summary.suptitle(
        "MMOTU Ensemble Uncertainty — Test Set Summary",
        fontsize=12, color=BLUE_DARK, fontweight="bold",
    )

    for panel_i, idx in enumerate(indices):
        col = panel_i % cols
        row_base = (panel_i // cols) * 4

        img_np  = (all_imgs[idx, 0] * 255).clip(0, 255).astype(np.uint8)
        gt_np   = (all_masks[idx, 0] >= 0.5)
        pred_np = (all_probs[idx, 0] >= args.conformal_lambda)
        unc_np  = all_std[idx, 0]

        for r in range(4):
            axes_grid[row_base + r][col].axis("off")

        axes_grid[row_base + 0][col].imshow(img_np, cmap="gray", vmin=0, vmax=255)
        axes_grid[row_base + 0][col].set_title("US", fontsize=7)
        _contour_overlay(axes_grid[row_base + 1][col], img_np, gt_np,
                         color=BLUE_MED, linestyle="solid", label="GT")
        _contour_overlay(axes_grid[row_base + 2][col], img_np, pred_np,
                         color=BLUE_DARK, linestyle="dashed", label="Pred")
        axes_grid[row_base + 3][col].imshow(unc_np, cmap=_BLUE_CMAP)
        axes_grid[row_base + 3][col].set_title(
            f"Unc={img_uncertainty[idx]:.3f}", fontsize=7
        )

    fig_summary.tight_layout()
    summary_path = out_dir / "uncertainty_summary_grid.pdf"
    fig_summary.savefig(summary_path, format="pdf", bbox_inches="tight")
    plt.close(fig_summary)
    saved.append(summary_path)

    print(f"\n✓  {len(saved)} PDF(s) saved to  {out_dir}")
    for p in saved:
        print(f"    {p.name}")


if __name__ == "__main__":
    main()

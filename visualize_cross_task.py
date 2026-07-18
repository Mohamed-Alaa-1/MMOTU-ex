"""
visualize_cross_task.py

Bridges the classification Grad-CAM XAI pipeline and the segmentation
Deep-Ensemble uncertainty pipeline for a publication-quality comparison figure.

For each selected image a 1×3 panel is produced:
  Col 1 – Original ultrasound (as stored on disk, RGB or grayscale)
  Col 2 – Classification Grad-CAM heatmap (where does the classifier look?)
  Col 3 – Segmentation Epistemic Uncertainty (where is the ensemble unsure?)

Both heatmaps are resized back to the original image dimensions so they can
be compared pixel-for-pixel.

All outputs are saved as PDF to  results/figures/cross_task/
using the project-wide monochromatic blue colour palette.

Usage
-----
# Requires a trained classification checkpoint and two segmentation checkpoints:
python visualize_cross_task.py \\
    --cls_checkpoint  results/checkpoints/densenet121_best.pt \\
    --cls_model_name  densenet121 \\
    --seg_checkpoints results/checkpoints/segmentation/laura_small_best.pt \\
                      results/checkpoints/segmentation/laura_base_model2_best.pt \\
    --splits          results/splits.csv \\
    --n_images        6

Note: if no classification checkpoint is available yet (e.g. training has not
been run on the GPU machine), pass --cls_checkpoint none and the script will
skip the Grad-CAM column and produce a 1×2 panel instead.
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── project root ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent       # MMOTU-ex/
_SEG  = _ROOT / "segmentation_module"

for _p in [str(_ROOT), str(_SEG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── palette (monochromatic blue, project-wide convention) ───────────────────
BLUE_DARK  = "#003366"
BLUE_MED   = "#3366CC"
BLUE_LIGHT = "#6699FF"
BLUE_PALE  = "#99CCFF"

_BLUE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "mmotu_blue",
    ["white", BLUE_PALE, BLUE_LIGHT, BLUE_MED, BLUE_DARK],
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

# ── segmentation imports (always available) ──────────────────────────────────
from segmentation.models.auravit_config import LAURA_BASE, LAURA_SMALL, LAURA_TINY
from segmentation.models.lightweight_auravit import LightweightAuraViT
from segmentation.dataset import (
    MMOTUSegmentationDataset,
    inpaint_overlay_pixels,
    detect_overlay_risk,
)
from segmentation_uncertainty.ensemble_segmentation import DeepEnsembleSegmentationEstimator


# ── classification imports (optional – skipped if checkpoint == "none") ──────
def _try_import_cam():
    try:
        from models.factory import get_model
        from xai.cam_methods import CAMExplainer
        return get_model, CAMExplainer
    except ImportError:
        return None, None


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_seg_models(checkpoint_paths, device):
    models = []
    for p in checkpoint_paths:
        path = Path(p) if Path(p).is_absolute() else _ROOT / p
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = LightweightAuraViT(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        models.append(model)
        print(f"  [seg]  Loaded {path.name}")
    return models


def _load_cls_model(checkpoint_path, model_name, num_classes, device):
    get_model, _ = _try_import_cam()
    if get_model is None:
        raise RuntimeError("models.factory not importable from project root.")
    model, _ = get_model(model_name, num_classes=num_classes, pretrained=False)
    path = Path(checkpoint_path) if Path(checkpoint_path).is_absolute() \
        else _ROOT / checkpoint_path
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    print(f"  [cls]  Loaded {path.name}")
    return model


def _preprocess_seg(pil_image):
    """Inpaint + grayscale, return (1,1,256,256) float32 tensor in [0,1]."""
    from PIL import Image as PILImage
    # inpaint overlays
    img_np = np.array(pil_image.convert("RGB"))
    img_np = inpaint_overlay_pixels(img_np)
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray   = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)
    t      = torch.from_numpy(gray.astype(np.float32) / 255.0)
    return t.unsqueeze(0).unsqueeze(0)          # [1,1,256,256]


def _preprocess_cls(pil_image, device):
    """Standard ImageNet pre-processing for the classification backbone."""
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf(pil_image.convert("RGB")).unsqueeze(0).to(device)


def _cam_heatmap(cls_model, model_name, image_tensor, class_idx, device):
    """Compute Grad-CAM for a single image; returns [H, W] float32 [0,1]."""
    _, CAMExplainer = _try_import_cam()
    explainer = CAMExplainer(cls_model, model_name, device)
    cam = explainer.compute_cam(image_tensor, class_idx=class_idx,
                                method="gradcam")
    return cam.astype(np.float32)   # [224, 224] normalised to [0,1]


def _unc_heatmap(ensemble, seg_tensor, device):
    """Return (pred_mask, epistemic_std) both [256,256] numpy arrays."""
    est = ensemble.predict(seg_tensor.to(device))
    return (
        est["mean_probs"][0, 0],    # [256,256]
        est["epistemic_std"][0, 0],
    )


def _resize_to(arr2d, h, w):
    """Bilinear resize a 2-D numpy array to (h, w)."""
    return cv2.resize(arr2d, (w, h), interpolation=cv2.INTER_LINEAR)


def _build_panel(orig_rgb_np, gt_np_hw,
                 cam_hw, unc_hw, pred_hw,
                 title, has_cam):
    """Assemble and return the matplotlib figure for one image."""
    n_cols = 4 if has_cam else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3.8, 4.2))
    fig.suptitle(title, fontsize=9, color=BLUE_DARK, fontweight="bold", y=1.01)

    col = 0

    # --- Col 0: original image ---
    axes[col].imshow(orig_rgb_np)
    axes[col].set_title("Original Ultrasound")
    axes[col].axis("off")
    col += 1

    # --- Col 1: Classification Grad-CAM (optional) ---
    if has_cam and cam_hw is not None:
        axes[col].imshow(orig_rgb_np)
        axes[col].imshow(cam_hw, cmap=_BLUE_CMAP, alpha=0.55,
                         vmin=0, vmax=1)
        axes[col].set_title("Classification Grad-CAM")
        axes[col].axis("off")
        col += 1

    # --- Col 2: Segmentation ground truth + prediction overlay ---
    gray_np = cv2.cvtColor(orig_rgb_np, cv2.COLOR_RGB2GRAY)
    axes[col].imshow(gray_np, cmap="gray", vmin=0, vmax=255)
    if gt_np_hw is not None and gt_np_hw.sum() > 0:
        axes[col].contour(gt_np_hw.astype(float), levels=[0.5],
                          colors=[BLUE_MED], linestyles=["solid"],
                          linewidths=[1.5])
    if pred_hw.sum() > 0:
        axes[col].contour(pred_hw.astype(float), levels=[0.5],
                          colors=[BLUE_DARK], linestyles=["dashed"],
                          linewidths=[1.5])
    # Legend proxies
    from matplotlib.lines import Line2D
    axes[col].legend(
        handles=[
            Line2D([0], [0], color=BLUE_MED,  linewidth=1.5, linestyle="solid",
                   label="Ground Truth"),
            Line2D([0], [0], color=BLUE_DARK, linewidth=1.5, linestyle="dashed",
                   label="Prediction"),
        ],
        fontsize=7, loc="lower right", framealpha=0.6,
    )
    axes[col].set_title("Segmentation Prediction")
    axes[col].axis("off")
    col += 1

    # --- Col 3: Epistemic uncertainty heat-map ---
    im = axes[col].imshow(unc_hw, cmap=_BLUE_CMAP, vmin=0)
    axes[col].set_title("Epistemic Uncertainty")
    axes[col].axis("off")
    cbar = fig.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)
    cbar.set_label("Ensemble Std", fontsize=8, color=BLUE_DARK)
    cbar.ax.yaxis.set_tick_params(color=BLUE_DARK, labelsize=7)

    fig.tight_layout()
    return fig


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-task comparison: Grad-CAM vs Segmentation Uncertainty.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cls_checkpoint", default="none",
                   help="Classification model checkpoint path, or 'none' to skip.")
    p.add_argument("--cls_model_name", default="densenet121",
                   help="Architecture name (densenet121, resnet50, …).")
    p.add_argument("--cls_num_classes", type=int, default=8)
    p.add_argument("--seg_checkpoints", nargs="+", required=True,
                   help="Two (or more) segmentation model checkpoint paths.")
    p.add_argument("--splits", default="results/splits.csv")
    p.add_argument("--out_dir", default="results/figures/cross_task")
    p.add_argument("--n_images", type=int, default=6,
                   help="Number of images to visualise.")
    p.add_argument("--select", choices=["most_uncertain", "most_confident", "random"],
                   default="most_uncertain")
    p.add_argument("--conformal_lambda", type=float, default=0.44,
                   help="Conformal threshold used for prediction contours.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 60)
    print(" MMOTU – Cross-Task XAI vs Uncertainty Comparison ")
    print("=" * 60)

    # 1. Load segmentation ensemble -----------------------------------
    print("\nLoading segmentation models…")
    seg_models = _load_seg_models(args.seg_checkpoints, device)
    ensemble   = DeepEnsembleSegmentationEstimator(seg_models, device)

    # 2. Load classification model (optional) -------------------------
    has_cam   = args.cls_checkpoint.lower() != "none"
    cls_model = None
    if has_cam:
        print("\nLoading classification model…")
        try:
            cls_model = _load_cls_model(
                args.cls_checkpoint,
                args.cls_model_name,
                args.cls_num_classes,
                device,
            )
        except Exception as exc:
            print(f"  ⚠  Could not load classification model: {exc}")
            print("     Proceeding without Grad-CAM column.")
            has_cam   = False
            cls_model = None

    # 3. Load test dataset -------------------------------------------
    splits_df = pd.read_csv(_ROOT / args.splits)
    test_df   = splits_df[splits_df["split"] == "test"].reset_index(drop=True)

    # We need the raw file paths to reload images at native resolution
    seg_dataset = MMOTUSegmentationDataset(
        test_df, image_size=256, apply_inpainting=True, augment=False
    )

    # 4. Quick uncertainty pass to rank images -----------------------
    print(f"\nComputing uncertainty for {len(test_df)} test images…")
    img_unc_scores = []
    from torch.utils.data import DataLoader
    rank_loader = DataLoader(seg_dataset, batch_size=16, shuffle=False)
    for imgs, _ in tqdm(rank_loader, desc="Uncertainty pass"):
        est = ensemble.predict(imgs.to(device))
        img_unc_scores.extend(est["epistemic_std"].mean(axis=(1, 2, 3)).tolist())

    img_unc_scores = np.array(img_unc_scores)
    n = min(args.n_images, len(test_df))

    if args.select == "most_uncertain":
        indices = np.argsort(img_unc_scores)[::-1][:n]
    elif args.select == "most_confident":
        indices = np.argsort(img_unc_scores)[:n]
    else:
        indices = np.random.choice(len(test_df), size=n, replace=False)

    print(f"  Selected {n} {args.select} images.")

    # 5. Generate one PDF per image -----------------------------------
    from PIL import Image as PILImage

    saved = []
    for rank, idx in enumerate(indices, start=1):
        row   = test_df.iloc[idx]
        # --- load raw image at native resolution ---
        img_path  = Path(row["image_path"])
        pil_orig  = PILImage.open(img_path).convert("RGB")
        orig_hw   = (pil_orig.height, pil_orig.width)
        orig_rgb  = np.array(pil_orig)

        # --- load ground-truth mask ---
        gt_mask_np = None
        if pd.notna(row.get("mask_path", None)):
            mask_pil = PILImage.open(row["mask_path"]).convert("L")
            mask_np  = np.array(mask_pil.resize((orig_hw[1], orig_hw[0]),
                                                 PILImage.NEAREST))
            gt_mask_np = (mask_np > 10).astype(np.uint8)

        # --- segmentation inference at 256×256 ---
        seg_tensor = _preprocess_seg(pil_orig)
        pred_256, unc_256 = _unc_heatmap(ensemble, seg_tensor, device)
        pred_hw = _resize_to((pred_256 >= args.conformal_lambda).astype(np.uint8),
                              orig_hw[0], orig_hw[1])
        unc_hw  = _resize_to(unc_256, orig_hw[0], orig_hw[1])

        # --- classification Grad-CAM (optional) ---
        cam_hw = None
        if has_cam and cls_model is not None:
            cls_tensor = _preprocess_cls(pil_orig, device)
            with torch.no_grad():
                logits    = cls_model(cls_tensor)
                class_idx = int(logits.argmax(dim=1).item())
            cam_224 = _cam_heatmap(cls_model, args.cls_model_name,
                                   cls_tensor, class_idx, device)
            cam_hw  = _resize_to(cam_224, orig_hw[0], orig_hw[1])

        # --- compose figure ---
        unc_scalar = img_unc_scores[idx]
        title = (
            f"Image {rank}/{n}  |  "
            f"Epistemic Unc = {unc_scalar:.4f}  |  "
            f"Class = {row.get('label', '?')}"
        )
        fig = _build_panel(orig_rgb, gt_mask_np, cam_hw, unc_hw, pred_hw,
                           title, has_cam)

        fname = out_dir / f"cross_task_{rank:02d}_idx{idx}.pdf"
        fig.savefig(fname, format="pdf", bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)
        print(f"  [{rank}/{n}] saved {fname.name}")

    # 6. Combined summary grid ----------------------------------------
    cols     = min(3, n)
    rows     = int(np.ceil(n / cols))
    n_inner  = 4 if has_cam else 3    # panels per image

    fig_grid, gax = plt.subplots(
        rows * n_inner, cols,
        figsize=(cols * 4.0, rows * 4.0),
        squeeze=False,
    )
    fig_grid.suptitle(
        "Cross-Task XAI vs Epistemic Uncertainty — MMOTU Test Set",
        fontsize=12, color=BLUE_DARK, fontweight="bold",
    )

    for panel_i, idx in enumerate(indices):
        col_i = panel_i % cols
        rb    = (panel_i // cols) * n_inner

        row   = test_df.iloc[idx]
        pil   = PILImage.open(row["image_path"]).convert("RGB")
        oh, ow = pil.height, pil.width

        rgb_np = np.array(pil)
        seg_t  = _preprocess_seg(pil)
        pm256, us256 = _unc_heatmap(ensemble, seg_t, device)
        p_hw   = _resize_to((pm256 >= args.conformal_lambda).astype(np.uint8), oh, ow)
        u_hw   = _resize_to(us256, oh, ow)

        for r in range(n_inner):
            gax[rb + r][col_i].axis("off")

        c = 0
        gax[rb + c][col_i].imshow(rgb_np)
        gax[rb + c][col_i].set_title("US", fontsize=7)
        c += 1

        if has_cam:
            cls_t = _preprocess_cls(pil, device)
            with torch.no_grad():
                logits = cls_model(cls_t)
                ci = int(logits.argmax(1).item())
            cam224 = _cam_heatmap(cls_model, args.cls_model_name, cls_t, ci, device)
            c_hw = _resize_to(cam224, oh, ow)
            gax[rb + c][col_i].imshow(rgb_np)
            gax[rb + c][col_i].imshow(c_hw, cmap=_BLUE_CMAP, alpha=0.55)
            gax[rb + c][col_i].set_title("Grad-CAM", fontsize=7)
            c += 1

        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        gax[rb + c][col_i].imshow(gray, cmap="gray", vmin=0, vmax=255)
        if p_hw.sum() > 0:
            gax[rb + c][col_i].contour(p_hw.astype(float), levels=[0.5],
                                       colors=[BLUE_DARK], linestyles=["dashed"],
                                       linewidths=[1.2])
        gax[rb + c][col_i].set_title("Seg Pred", fontsize=7)
        c += 1

        gax[rb + c][col_i].imshow(u_hw, cmap=_BLUE_CMAP)
        gax[rb + c][col_i].set_title(f"Unc={img_unc_scores[idx]:.3f}", fontsize=7)

    fig_grid.tight_layout()
    grid_path = out_dir / "cross_task_summary_grid.pdf"
    fig_grid.savefig(grid_path, format="pdf", bbox_inches="tight")
    plt.close(fig_grid)
    saved.append(grid_path)

    print(f"\n✓  {len(saved)} PDF(s) saved to  {out_dir}")
    for s in saved:
        print(f"    {s.name}")


if __name__ == "__main__":
    main()

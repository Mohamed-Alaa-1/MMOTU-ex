from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim > 2:
        array = np.squeeze(array)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value > min_value:
        return (255.0 * (array - min_value) / (max_value - min_value)).astype(np.uint8)
    return np.zeros_like(array, dtype=np.uint8)


def overlay_cam_on_image(original_img_np: np.ndarray, cam_np: np.ndarray, seg_mask_np: Optional[np.ndarray] = None, alpha: float = 0.45) -> np.ndarray:
    """Overlay a CAM heatmap and an optional segmentation contour on an RGB image."""
    image_rgb = np.asarray(original_img_np)
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    cam_uint8 = _normalize_to_uint8(cam_np)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if image_rgb.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap, alpha, 0)

    if seg_mask_np is not None and np.max(seg_mask_np) > 0:
        mask = np.asarray(seg_mask_np)
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        if mask.shape != overlay.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    return overlay


def _resolve_mask_array_from_row(row: Dict) -> np.ndarray:
    mask_path = row.get("mask_path")
    if mask_path and isinstance(mask_path, str) and Path(mask_path).exists():
        mask_image = Image.open(mask_path).convert("L")
        return (np.array(mask_image) > 0).astype(np.uint8) * 255

    image_path = row.get("image_path")
    if image_path and isinstance(image_path, str):
        image_path = Path(image_path)
        inferred_candidates = [
            image_path.parent.parent / "annotations" / f"{image_path.stem}.PNG",
            image_path.parent.parent / "annotations" / image_path.name,
            image_path.parent.parent / "masks" / f"{image_path.stem}.PNG",
        ]
        for candidate in inferred_candidates:
            if candidate.exists():
                mask_image = Image.open(candidate).convert("L")
                return (np.array(mask_image) > 0).astype(np.uint8) * 255

    return np.zeros((224, 224), dtype=np.uint8)


def save_comparison_figure(image_path: str, seg_mask: np.ndarray, cam_results_dict: dict, metrics_dict: dict, save_path: str):
    """Legacy helper retained for method-wise CAM comparison figures."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    methods = list(cam_results_dict.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(nrows=max(1, n_methods), ncols=4, figsize=(16, 4 * max(1, n_methods)))

    if n_methods == 1:
        axes = [axes]

    for i, method in enumerate(methods):
        cam = cam_results_dict[method]
        metrics = metrics_dict.get(method, {})

        overlay = overlay_cam_on_image(img, cam, seg_mask)
        bin_cam = (_normalize_to_uint8(cam) > 127).astype(np.uint8) * 255

        ax_orig = axes[i][0]
        ax_orig.imshow(img)
        ax_orig.set_title(f"Original ({method})")
        ax_orig.axis('off')

        ax_seg = axes[i][1]
        ax_seg.imshow(seg_mask, cmap='gray')
        ax_seg.set_title("Seg Mask")
        ax_seg.axis('off')

        ax_over = axes[i][2]
        ax_over.imshow(overlay)
        ax_over.set_title("CAM Overlay")
        ax_over.axis('off')

        ax_bin = axes[i][3]
        ax_bin.imshow(bin_cam, cmap='gray')
        m_str = f"SC:{metrics.get('sc',0):.2f} CC:{metrics.get('cc',0):.2f}\nWCIS:{metrics.get('wcis',0):.2f} Ex:{metrics.get('exbale',0):.2f}"
        ax_bin.set_title(f"Binarized\n{m_str}")
        ax_bin.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def save_qualitative_comparison_figure(rows: List[Dict], save_path: str):
    """Save a two-row qualitative comparison figure with Original, Mask, and four explanation maps."""
    method_order = ["gradcam", "scorecam", "eigencam", "saliency"]
    column_titles = ["Original Image", "Seg. Mask", "Grad CAM", "Score CAM", "Eigen CAM", "Saliency"]

    fig, axes = plt.subplots(nrows=len(rows), ncols=6, figsize=(18, 6.5), constrained_layout=True)
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, row in enumerate(rows):
        image = row["image"]
        mask = row.get("mask")
        if mask is None:
            mask = _resolve_mask_array_from_row(row)
        cams = row["cams"]
        row_label = row.get("row_label", "")

        for col_idx, title in enumerate(column_titles):
            axes[row_idx, col_idx].set_title(title, fontsize=10, pad=6)
            axes[row_idx, col_idx].axis('off')

        axes[row_idx, 0].imshow(image)
        axes[row_idx, 1].imshow(mask, cmap='gray')

        for offset, method_name in enumerate(method_order, start=2):
            overlay = overlay_cam_on_image(image, cams[method_name], mask)
            axes[row_idx, offset].imshow(overlay)

        axes[row_idx, 0].text(-0.18, 0.5, row_label, transform=axes[row_idx, 0].transAxes,
                              rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

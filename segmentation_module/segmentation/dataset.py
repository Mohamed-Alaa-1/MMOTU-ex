"""
segmentation/dataset.py

Grayscale, single-channel, 256x256 dataset for LightweightAuraViT, reusing
the existing patient-level splits (results/splits.csv) without
regenerating them. Separate from data/dataset.py (classification), which
loads RGB at 224x224 for the five CNN/Swin backbones. See
segmentation_extension_plan.md Section 1, points 1 and 2 for why this
diverges (grayscale for the two num_channels touch points in the model,
native 256x256 to avoid touching the pos_embed/num_patches coupling).

Grayscale conversion validated against a real MMOTU-style ultrasound
sample: 99.51% of pixels have RGB channels within 5 gray levels of each
other (mean |R-G|=1.8, mean |G-B|=2.3), confirming the image is native
grayscale and converting loses essentially no information for the large
majority of pixels. The remaining 0.49% of pixels, with channel
divergence up to 184/255, are burned-in caliper crosshairs and on-screen
measurement text, not tissue. See detect_overlay_risk() below.

Mask color handling validated against the same real sample: masks in this
dataset are not guaranteed to use a consistent fill color (one real
example used a pure black background with an olive (64, 64, 0) fill,
which is not white and not obviously "255"). Converting to grayscale
(luminosity) before thresholding handles any fill color correctly as long
as the background is pure or near-pure black, since even a dim fill color
still produces a comfortably nonzero luminosity value (64, 64, 0
converts to approximately 57 out of 255). The threshold below is set to
10, not a literal >0, specifically to reject faint anti-aliasing halo
pixels around a mask's edge (which can appear as very low but nonzero
luminosity, for example 1 to 5, if the mask was saved with any smoothing)
without rejecting any legitimate fill color, which will realistically
always be well above 10.

Overlay inpainting (added after the corpus-wide overlay scan found 96.6%
of MMOTU images contain burned-in caliper crosshairs):
  - inpaint_overlay_pixels() detects colored pixels via channel divergence
    and fills them with surrounding tissue values using OpenCV Navier-Stokes
    inpainting. Applied to the raw RGB image before grayscale conversion,
    so the ViT encoder never sees the caliper pattern.
  - Training images additionally receive ColorJitter and optional
    GaussianBlur augmentation to prevent the model memorising any residual
    luminosity artefacts that survive the inpainting threshold.
  - Val/test images receive inpainting but no augmentation, preserving
    deterministic evaluation.
"""

from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Anti-aliasing guard: a mask pixel must exceed this grayscale luminosity
# to count as foreground. See module docstring for the rationale and the
# real-data validation (olive fill (64,64,0) converts to ~57, comfortably
# above this).
MASK_FOREGROUND_LUMINOSITY_THRESHOLD = 10

# Overlay detection threshold: pixels whose maximum pairwise RGB channel
# difference exceeds this value are treated as colored (caliper/UI) content.
# Calibrated on a real JPEG MMOTU sample; see detect_overlay_risk() for the
# full noise-floor analysis. Do not lower below ~30 without re-checking
# against real JPEG images (JPEG chroma subsampling produces a noise floor
# peaking around divergence 17 at the 99.5th percentile).
OVERLAY_CHANNEL_DIFF_THRESHOLD = 40

# OpenCV inpainting neighbourhood radius (pixels). Caliper crosshairs in
# this dataset are 1-3 pixels wide; radius 3 comfortably covers the line
# width while pulling from genuine surrounding tissue texture.
INPAINT_RADIUS = 3


# ---------------------------------------------------------------------------
# Overlay detection and inpainting
# ---------------------------------------------------------------------------

def inpaint_overlay_pixels(
    image_rgb: Image.Image,
    channel_diff_threshold: int = OVERLAY_CHANNEL_DIFF_THRESHOLD,
    inpaint_radius: int = INPAINT_RADIUS,
) -> Image.Image:
    """Detect and inpaint colored overlay pixels (caliper crosshairs,
    manufacturer text, measurement annotations) in an RGB PIL image.

    Detection: pixels whose maximum pairwise RGB channel difference exceeds
    channel_diff_threshold are marked as overlay content. For JPEG sources
    this threshold should stay >= 30 to avoid flagging ordinary chroma
    subsampling noise as overlay; the default (40) sits in the gap between
    the JPEG noise floor (divergence <= ~17 at 99.5th percentile on real
    samples) and genuine colored UI content (divergence >= ~57).

    Inpainting: OpenCV Navier-Stokes inpainting (INPAINT_NS) fills the
    detected pixels by propagating surrounding tissue texture inward.
    Produces visually seamless results for thin (1-3 pixel) lines and
    compact text blobs. For thick patches the fill looks like smeared tissue,
    which is acceptable since those regions are anatomically irrelevant
    (they are corner annotations, not boundary-adjacent calipers).

    This function is a no-op (returns the original image unmodified) if no
    pixels exceed the threshold, so it is safe to apply unconditionally to
    every image including clean ones, with no runtime cost beyond the
    channel-divergence check.

    Args:
        image_rgb:             PIL Image in RGB mode.
        channel_diff_threshold: Per-pixel max(|R-G|, |G-B|, |R-B|) above
                                which a pixel is treated as overlay.
        inpaint_radius:        OpenCV inpainting neighbourhood radius.

    Returns:
        PIL Image (RGB) with overlay pixels filled by surrounding tissue
        values, or the original image unchanged if no overlay was detected.
    """
    img_np = np.array(image_rgb).astype(np.int32)  # int32 to avoid uint8 subtraction wrap
    r, g, b = img_np[..., 0], img_np[..., 1], img_np[..., 2]
    max_diff = np.maximum(
        np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b))
    )
    overlay_mask = (max_diff > channel_diff_threshold).astype(np.uint8)

    if overlay_mask.sum() == 0:
        return image_rgb  # clean image: skip inpainting entirely

    # OpenCV expects BGR; inpaint mask is single-channel uint8
    img_bgr = cv2.cvtColor(np.array(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(img_bgr, overlay_mask, inpaint_radius, cv2.INPAINT_NS)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(inpainted_rgb)


def detect_overlay_risk(image_path: str, channel_diff_threshold: int = 40,
                         flag_pixel_fraction: float = 0.0005) -> dict:
    """Flags images likely to contain burned-in caliper crosshairs, on-screen
    measurement text, or other UI overlays, by measuring how many pixels
    have RGB channels that diverge significantly from each other. Native
    ultrasound tissue is close to grayscale (R approximately equals G
    approximately equals B); colored UI elements (calipers are
    conventionally yellow or green specifically to stand out against
    grayscale tissue) do not share this property.

    This matters beyond just image quality: calipers are placed by the
    original sonographer directly at the boundary of the structure being
    measured, meaning caliper pixel locations are spatially correlated
    with the true lesion boundary. A model, especially a patch-based ViT
    that encodes raw pixel content directly, could learn to key off caliper
    crosshair patterns as a shortcut for lesion boundary detection, which
    would not generalize to any real deployment image without calipers
    burned in. Converting to grayscale before training does not remove
    this risk on its own, since the caliper marks still produce a distinct
    luminosity pattern relative to surrounding tissue.

    channel_diff_threshold defaults to 40, not a naive small value like 5.
    Measured on a real JPEG ultrasound sample, per-pixel channel divergence
    has a median of 3 and a 90th percentile of only 7 across the entire
    image, from ordinary JPEG chroma subsampling noise, not overlay
    content. That noise floor extends out to roughly the 99.5th percentile
    (divergence 17) before a distinct second population appears: genuine
    colored content (caliper crosshairs, on-screen text) starts around
    divergence 57 (99.8th percentile) and reaches 184 at the maximum. A
    threshold of 5 flagged 31.8% of pixels on this real sample, almost
    entirely compression noise; a threshold of 40, sitting in the gap
    between the two populations, flagged 0.27%, consistent with a small
    number of genuinely colored UI elements rather than broad image noise.
    Use a smaller threshold only for known-lossless (PNG) source images,
    where this compression noise floor does not apply.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image).astype(int)
    r, g, b = image_np[..., 0], image_np[..., 1], image_np[..., 2]

    max_channel_diff = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b)))
    flagged_mask = max_channel_diff > channel_diff_threshold
    flagged_fraction = float(flagged_mask.mean())

    return {
        "image_path": image_path,
        "flagged_pixel_fraction": flagged_fraction,
        "likely_overlay": flagged_fraction > flag_pixel_fraction,
        "max_channel_divergence": int(max_channel_diff.max()),
        "mean_channel_divergence": float(max_channel_diff.mean()),
    }


def scan_dataset_for_overlay_risk(splits_csv: str) -> pd.DataFrame:
    """Runs detect_overlay_risk over every image in results/splits.csv and
    returns a DataFrame sorted by flagged_pixel_fraction descending, so the
    most suspicious images surface first for manual review. Intended to be
    run once as a standalone QA pass before committing to full training,
    not called from inside the Dataset's __getitem__ (which would repeat
    this scan every epoch for no benefit, since the underlying image files
    do not change during training)."""
    df = pd.read_csv(splits_csv)
    results = []
    for _, row in df.iterrows():
        try:
            result = detect_overlay_risk(row["image_path"])
            result["split"] = row.get("split", None)
            results.append(result)
        except Exception as e:
            results.append({
                "image_path": row["image_path"], "flagged_pixel_fraction": float("nan"),
                "likely_overlay": False, "max_channel_divergence": -1,
                "mean_channel_divergence": float("nan"), "split": row.get("split", None),
                "error": str(e),
            })
    result_df = pd.DataFrame(results).sort_values("flagged_pixel_fraction", ascending=False)
    n_flagged = int(result_df["likely_overlay"].sum())
    if n_flagged > 0:
        print(f"scan_dataset_for_overlay_risk: {n_flagged}/{len(result_df)} images flagged "
              f"as likely containing burned-in overlays. Review before training; see "
              f"detect_overlay_risk() docstring for why this matters for a segmentation model.")
    return result_df


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _base_image_transform(image_size: int) -> transforms.Compose:
    """Deterministic transform for val/test: resize, grayscale, normalise.
    No augmentation so evaluation is reproducible."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # Single-channel normalisation stats, distinct from the
        # 3-channel ImageNet stats used by data/dataset.py, since this
        # branch operates on grayscale ultrasound directly rather than
        # RGB-normalised natural images. 0.5/0.5 maps [0,1] to [-1,1].
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def _augmented_image_transform(image_size: int) -> transforms.Compose:
    """Training transform: adds ColorJitter and optional GaussianBlur
    (applied before Grayscale conversion so they operate on the RGB image,
    where brightness/contrast changes are better defined).

    ColorJitter randomises the luminosity of any residual caliper artefacts
    that survive the inpainting step (e.g., borderline pixels just below the
    channel_diff_threshold). GaussianBlur softens sharp line edges.
    Both make it harder for the model to memorise caliper patterns even when
    a few stray pixels slip through.

    Applied only to training images; val/test use _base_image_transform()
    for deterministic evaluation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # ColorJitter before Grayscale: operates on RGB so it affects
        # relative channel balance, not just overall brightness.
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        # Mild blur with 30% probability: destroys sub-pixel caliper line
        # sharpness without visibly degrading tissue texture.
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3
        ),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def _mask_transform(image_size: int) -> transforms.Compose:
    """Mask transform: NEAREST-interpolation resize only.
    Never augmented (brightness/blur would corrupt binary labels)."""
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MMOTUSegmentationDataset(Dataset):
    """Grayscale 256×256 segmentation dataset for LightweightAuraViT.

    Two new parameters relative to the original version:

    apply_inpainting (default True):
        Call inpaint_overlay_pixels() on each image before applying the
        transform. Removes caliper crosshairs and UI overlays that are
        spatially correlated with the tumor boundary; see the module
        docstring and segmentation_extension_plan.md Section 1a for the
        full rationale. Safe to set True on val/test: the inpainting is
        deterministic (no randomness), so evaluation remains reproducible.

    augment (default False):
        Use the training-specific transform (_augmented_image_transform)
        instead of the deterministic base transform. Should be True only
        for the training DataLoader, never for val/test. Ignored if a
        custom transform is passed.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 256,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        return_path: bool = False,
        apply_inpainting: bool = True,
        augment: bool = False,
    ):
        self.df = df
        self.image_size = image_size
        self.return_path = return_path
        self.apply_inpainting = apply_inpainting

        # If the caller supplies a custom transform, use it as-is and ignore
        # the augment flag. This preserves backward compatibility for any code
        # that passed a transform directly (e.g., existing tests).
        if transform is not None:
            self.transform = transform
        elif augment:
            self.transform = _augmented_image_transform(image_size)
        else:
            self.transform = _base_image_transform(image_size)

        self.mask_transform = mask_transform or _mask_transform(image_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        mask_path = row["mask_path"]

        # --- Load image ---
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size))

        # --- Inpaint caliper overlays (on RGB, before grayscale conversion) ---
        # This breaks the spatial correlation between caliper crosshair
        # pixels and the mask boundary that a ViT could otherwise exploit.
        # The inpainting is deterministic (no random seed), so applying it
        # to val/test images does not compromise evaluation reproducibility.
        if self.apply_inpainting:
            image = inpaint_overlay_pixels(image)

        # --- Load and binarise mask ---
        try:
            if mask_path and pd.notna(mask_path) and Path(mask_path).exists():
                mask = Image.open(mask_path).convert("L")
                # Threshold at MASK_FOREGROUND_LUMINOSITY_THRESHOLD rather
                # than a literal >0: real masks in this dataset use varying
                # non-white fill colors (see module docstring), and >0
                # would also accept faint anti-aliasing halo pixels around
                # a mask edge as foreground, subtly inflating the mask
                # boundary by a pixel or two.
                mask_np = (
                    np.array(mask) > MASK_FOREGROUND_LUMINOSITY_THRESHOLD
                ).astype(np.uint8) * 255
                mask = Image.fromarray(mask_np, mode="L")
            else:
                mask = Image.new("L", image.size, 0)
        except Exception:
            mask = Image.new("L", image.size, 0)

        image_t = self.transform(image)
        mask_t = self.mask_transform(mask)
        # Binarise post-resize: NEAREST interpolation should already
        # preserve 0/255 exactly, but guard against any interpolation
        # artefact producing intermediate values before ToTensor scales
        # to [0, 1].
        mask_t = (mask_t > 0.5).float()

        if self.return_path:
            return image_t, mask_t, img_path, mask_path
        return image_t, mask_t


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_segmentation_dataloaders(
    splits_csv: str,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    apply_inpainting: bool = True,
):
    """Build train/val/test DataLoaders from a patient-level splits CSV.

    Training DataLoader:
        - apply_inpainting=True (caliper removal, deterministic)
        - augment=True (ColorJitter + GaussianBlur, stochastic)
        - drop_last=True (required: ASPP BatchNorm crashes on batch size 1
          in train mode; see segmentation_extension_plan.md Section 4)

    Val/Test DataLoaders:
        - apply_inpainting=True (same inpainting for fair comparison)
        - augment=False (deterministic evaluation)
        - drop_last=False

    Args:
        splits_csv:       Path to the CSV with image_path, mask_path, split.
        image_size:       Spatial resolution fed to the model (256 default).
        batch_size:       Batch size for all three loaders.
        num_workers:      DataLoader worker processes (set 0 for Windows
                          debugging to avoid multiprocessing overhead).
        apply_inpainting: Set False to disable caliper inpainting for
                          ablation experiments (e.g., to measure the
                          effect of inpainting on Dice).
    """
    df = pd.read_csv(splits_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_dataset = MMOTUSegmentationDataset(
        train_df, image_size=image_size,
        apply_inpainting=apply_inpainting, augment=True,
    )
    val_dataset = MMOTUSegmentationDataset(
        val_df, image_size=image_size,
        apply_inpainting=apply_inpainting, augment=False,
    )
    test_dataset = MMOTUSegmentationDataset(
        test_df, image_size=image_size,
        apply_inpainting=apply_inpainting, augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader

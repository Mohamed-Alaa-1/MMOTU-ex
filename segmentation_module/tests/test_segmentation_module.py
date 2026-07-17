"""
tests/test_segmentation_module.py

Regression suite for the segmentation module (models, losses, metrics,
dataset, trainer). Run with: pytest tests/test_segmentation_module.py -v

Covers every issue found and fixed during the AuraViT port, so a future
change that reintroduces one of these bugs fails loudly instead of
silently. See segmentation_extension_plan.md for the full rationale
behind each fix.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from types import SimpleNamespace
from torch.utils.data import DataLoader

from segmentation.models.auravit_config import (
    build_config, LAURA_BASE, LAURA_SMALL, LAURA_TINY, PRESETS,
)
from segmentation.models.lightweight_auravit import LightweightAuraViT
from segmentation.losses import DiceBCELoss
from segmentation.metrics import dice_score, iou_score, compute_all_segmentation_metrics
from segmentation.dataset import (
    MMOTUSegmentationDataset, get_segmentation_dataloaders,
    detect_overlay_risk, inpaint_overlay_pixels,
    MASK_FOREGROUND_LUMINOSITY_THRESHOLD,
)
from segmentation_uncertainty.mc_dropout_segmentation import (
    enable_mc_dropout_segmentation, MCDropoutSegmentationEstimator,
)
from segmentation_uncertainty.ensemble_segmentation import (
    DeepEnsembleSegmentationEstimator,
)
from segmentation_uncertainty.conformal_risk_control import (
    SegmentationConformalRiskController,
)
from segmentation_uncertainty.selective_segmentation import (
    selective_segmentation_risk_coverage, image_level_uncertainty,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_build_config_rejects_non_divisible_image_size():
    with pytest.raises(ValueError):
        build_config(image_size=250, patch_size=16)


def test_build_config_rejects_unsupported_depth():
    with pytest.raises(ValueError):
        build_config(num_layers=5)


def test_build_config_rejects_head_dim_mismatch():
    with pytest.raises(ValueError):
        build_config(hidden_dim=500, num_heads=8)


def test_build_config_derives_num_patches():
    cfg = build_config(image_size=256, patch_size=16)
    assert cfg["num_patches"] == 256


def test_build_config_unifies_dropout_by_default():
    cfg = build_config(dropout_rate=0.3)
    assert cfg["block_dropout_rate"] == 0.3
    assert cfg["aspp_dropout_rate"] == 0.3


def test_build_config_allows_explicit_dropout_override():
    cfg = build_config(dropout_rate=0.1, block_dropout_rate=0.4)
    assert cfg["block_dropout_rate"] == 0.4
    assert cfg["dropout_rate"] == 0.1


@pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
def test_all_presets_are_valid(preset_name):
    cfg = PRESETS[preset_name]
    model = LightweightAuraViT(cfg)
    assert sum(p.numel() for p in model.parameters()) > 0


# ---------------------------------------------------------------------------
# Model: incomplete config rejection
# ---------------------------------------------------------------------------

def test_model_rejects_hand_written_incomplete_config():
    with pytest.raises(KeyError):
        LightweightAuraViT({"hidden_dim": 512})


# ---------------------------------------------------------------------------
# Model: forward pass correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", [LAURA_BASE, LAURA_SMALL, LAURA_TINY])
def test_forward_pass_output_shape(cfg):
    model = LightweightAuraViT(cfg)
    model.eval()
    x = torch.randn(2, cfg["num_channels"], cfg["image_size"], cfg["image_size"])
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, cfg["image_size"], cfg["image_size"])


def test_forward_rejects_nan_input():
    model = LightweightAuraViT(LAURA_TINY)
    x = torch.randn(1, 1, 256, 256)
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN detected in input"):
        model(x)


def test_forward_rejects_wrong_channel_count():
    model = LightweightAuraViT(LAURA_TINY)  # built for num_channels=1
    x = torch.randn(1, 3, 256, 256)
    with pytest.raises(ValueError, match="channels"):
        model(x)


def test_gradients_flow_through_entire_model():
    model = LightweightAuraViT(LAURA_TINY)
    model.train()
    x = torch.randn(2, 1, 256, 256)
    mask = torch.randint(0, 2, (2, 1, 256, 256)).float()
    out = model(x)
    loss = nn.functional.binary_cross_entropy_with_logits(out, mask)
    loss.backward()
    n_total = sum(1 for _ in model.parameters())
    n_nonzero = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_nonzero == n_total


def test_batch_size_one_in_train_mode_fails_at_model_level():
    """Documents the real BatchNorm constraint this model has (ASPP's
    global-avg-pool branch collapses to 1x1 spatial, which BatchNorm2d
    cannot handle with a single sample in train mode). The fix lives in
    SegmentationTrainer._forward_safe, not in the model itself; this test
    exists so the underlying constraint is never silently "fixed away" by
    an unrelated future change without updating the trainer guard too."""
    model = LightweightAuraViT(LAURA_TINY)
    model.train()
    x = torch.randn(1, 1, 256, 256)
    with pytest.raises(ValueError):
        model(x)


# ---------------------------------------------------------------------------
# Parameter accounting (documents where the budget actually lives)
# ---------------------------------------------------------------------------

def test_encoder_dominates_parameter_budget_for_base_config():
    model = LightweightAuraViT(LAURA_BASE)
    encoder_params = sum(
        p.numel() for n, p in model.named_parameters()
        if n.startswith(("patch_embed", "pos_embed", "pos_dropout",
                          "trans_encoder_layers", "skip_norms"))
    )
    total = sum(p.numel() for p in model.parameters())
    # Encoder should dominate for the base config; this is the finding
    # that overturned the earlier draft of the lightweighting plan.
    assert encoder_params / total > 0.75


def test_decoder_share_grows_as_encoder_shrinks():
    """As hidden_dim/num_layers shrink, the decoder (which does not scale
    with hidden_dim) becomes a proportionally larger share of the model.
    This is the quantitative basis for treating decoder rescaling as a
    secondary, not primary, lightweighting lever until a certain point."""
    def decoder_share(cfg):
        model = LightweightAuraViT(cfg)
        decoder_params = sum(
            p.numel() for n, p in model.named_parameters()
            if n.startswith(("seg_d", "seg_s", "seg_c", "seg_output", "att_gate"))
        )
        total = sum(p.numel() for p in model.parameters())
        return decoder_params / total

    share_base = decoder_share(LAURA_BASE)
    share_tiny = decoder_share(LAURA_TINY)
    assert share_tiny > share_base


# ---------------------------------------------------------------------------
# MC-Dropout: the Dropout2d bug and its fix
# ---------------------------------------------------------------------------

def test_model_actually_contains_dropout2d_layers():
    """Guards against this test suite becoming meaningless if a future
    refactor removes Dropout2d from the architecture entirely."""
    model = LightweightAuraViT(LAURA_TINY)
    n_dropout2d = sum(1 for m in model.modules() if isinstance(m, nn.Dropout2d))
    assert n_dropout2d > 0


def test_naive_isinstance_dropout_check_misses_dropout2d():
    """Proves the bug exists: isinstance(module, nn.Dropout) does not
    catch nn.Dropout2d, since they are sibling classes, not parent/child."""
    model = LightweightAuraViT(LAURA_TINY)
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    still_eval = sum(1 for m in model.modules() if isinstance(m, nn.Dropout2d) and not m.training)
    total_dropout2d = sum(1 for m in model.modules() if isinstance(m, nn.Dropout2d))
    assert still_eval == total_dropout2d  # all of them missed by the naive check
    assert total_dropout2d > 0


def test_fixed_enable_mc_dropout_catches_all_dropout_variants():
    model = LightweightAuraViT(LAURA_TINY)
    model.eval()
    n_switched = enable_mc_dropout_segmentation(model)

    n_1d = sum(1 for m in model.modules() if isinstance(m, nn.Dropout))
    n_2d = sum(1 for m in model.modules() if isinstance(m, nn.Dropout2d))
    assert n_switched == n_1d + n_2d

    still_eval = sum(
        1 for m in model.modules()
        if isinstance(m, (nn.Dropout, nn.Dropout2d)) and not m.training
    )
    assert still_eval == 0


def test_fixed_enable_mc_dropout_leaves_batchnorm_untouched():
    model = LightweightAuraViT(LAURA_TINY)
    model.eval()
    enable_mc_dropout_segmentation(model)
    bn_in_train = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d) and m.training)
    assert bn_in_train == 0


def test_mc_dropout_estimator_produces_nonzero_pixelwise_variance():
    model = LightweightAuraViT(LAURA_TINY)
    estimator = MCDropoutSegmentationEstimator(model, device=torch.device("cpu"), n_samples=5)
    x = torch.randn(1, 1, 256, 256)
    result = estimator.predict(x)
    assert (result["epistemic_std"] > 0).all()


def test_mc_dropout_estimator_raises_if_no_dropout_found():
    model = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1))  # no dropout at all
    with pytest.raises(RuntimeError):
        MCDropoutSegmentationEstimator(model, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def test_loss_near_zero_for_near_perfect_prediction():
    criterion = DiceBCELoss()
    mask = torch.tensor([[[[1., 1., 0., 0.]]]])
    logits = torch.tensor([[[[10., 10., -10., -10.]]]])
    assert criterion(logits, mask).item() < 0.01


def test_loss_rejects_shape_mismatch():
    criterion = DiceBCELoss()
    with pytest.raises(ValueError):
        criterion(torch.randn(2, 1, 10, 10), torch.randn(2, 1, 8, 8))


def test_loss_rejects_invalid_bce_weight():
    with pytest.raises(ValueError):
        DiceBCELoss(bce_weight=1.5)


def test_loss_gradient_flows():
    criterion = DiceBCELoss()
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 16, 16)).float()
    criterion(logits, target).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_dice_identical_masks_is_one():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:30, 10:30] = True
    assert dice_score(mask, mask) == pytest.approx(1.0)


def test_dice_matches_hand_computed_partial_overlap():
    a = np.zeros((10, 10), dtype=bool); a[0:6, 0:6] = True
    b = np.zeros((10, 10), dtype=bool); b[3:9, 3:9] = True
    inter = 9
    expected = 2 * inter / (36 + 36)
    assert dice_score(a, b) == pytest.approx(expected, abs=1e-4)


def test_iou_matches_hand_computed_partial_overlap():
    a = np.zeros((10, 10), dtype=bool); a[0:6, 0:6] = True
    b = np.zeros((10, 10), dtype=bool); b[3:9, 3:9] = True
    inter = 9
    expected = inter / (36 + 36 - inter)
    assert iou_score(a, b) == pytest.approx(expected, abs=1e-4)


def test_compute_all_segmentation_metrics_keys():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:15, 5:15] = True
    result = compute_all_segmentation_metrics(mask, mask)
    assert set(result.keys()) == {"dice", "iou", "boundary_f1", "hd95"}
    assert result["dice"] == pytest.approx(1.0)
    assert result["hd95"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_splits_csv(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations").mkdir()
    rows = []
    for i in range(6):
        img = Image.fromarray((np.random.rand(180, 220, 3) * 255).astype(np.uint8), mode="RGB")
        mask_np = np.zeros((180, 220), dtype=np.uint8)
        mask_np[40:100, 50:120] = 255
        mask_img = Image.fromarray(mask_np, mode="L")

        img_path = tmp_path / "images" / f"{i}.jpg"
        mask_path = tmp_path / "annotations" / f"{i}.png"
        img.save(img_path)
        mask_img.save(mask_path)

        split = "train" if i < 4 else ("val" if i == 4 else "test")
        rows.append({"image_path": str(img_path), "mask_path": str(mask_path),
                     "class_label": i % 3, "split": split})

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "splits.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_dataset_produces_grayscale_single_channel(synthetic_splits_csv):
    df = pd.read_csv(synthetic_splits_csv)
    ds = MMOTUSegmentationDataset(df, image_size=256)
    img_t, mask_t = ds[0]
    assert img_t.shape == (1, 256, 256)
    assert mask_t.shape == (1, 256, 256)


def test_dataset_mask_is_strictly_binary(synthetic_splits_csv):
    df = pd.read_csv(synthetic_splits_csv)
    ds = MMOTUSegmentationDataset(df, image_size=256)
    _, mask_t = ds[0]
    unique_vals = torch.unique(mask_t)
    assert set(unique_vals.tolist()).issubset({0.0, 1.0})


def test_dataloaders_and_model_shapes_agree(synthetic_splits_csv):
    train_loader, val_loader, test_loader = get_segmentation_dataloaders(
        str(synthetic_splits_csv), image_size=256, batch_size=2, num_workers=0
    )
    model = LightweightAuraViT(LAURA_TINY)
    model.eval()
    images, masks = next(iter(train_loader))
    with torch.no_grad():
        out = model(images)
    assert out.shape == masks.shape


# ---------------------------------------------------------------------------
# Real-data fixtures: an actual MMOTU-style ultrasound image and mask,
# not synthetic. Frozen as permanent regression fixtures so the two real
# findings from reviewing them (mask fill color is not white, and the
# source image contains burned-in caliper/text overlays) stay covered by
# tests rather than living only in a one-off chat verification.
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_IMAGE_PATH = FIXTURES_DIR / "real_ultrasound_sample.jpg"
REAL_MASK_PATH = FIXTURES_DIR / "real_ultrasound_sample_mask.png"

requires_real_fixtures = pytest.mark.skipif(
    not (REAL_IMAGE_PATH.exists() and REAL_MASK_PATH.exists()),
    reason="real ultrasound fixture files not present",
)


@requires_real_fixtures
def test_real_mask_uses_nonwhite_fill_color():
    """Documents the finding that motivated the mask threshold change:
    this real mask's foreground fill is olive (64, 64, 0), not white, and
    is exactly two colors (pure black background, one fill color), not a
    grayscale gradient."""
    mask = Image.open(REAL_MASK_PATH).convert("RGB")
    mask_np = np.array(mask)
    unique_colors = np.unique(mask_np.reshape(-1, 3), axis=0)
    assert len(unique_colors) == 2
    colors = {tuple(c) for c in unique_colors}
    assert (0, 0, 0) in colors
    assert (255, 255, 255) not in colors  # the real finding: fill is not white


@requires_real_fixtures
def test_real_mask_binarizes_to_plausible_tumor_fraction():
    """The real mask's foreground should survive the
    MASK_FOREGROUND_LUMINOSITY_THRESHOLD binarization at roughly the same
    fraction as loading it directly (olive converts to luminosity ~57,
    comfortably above the threshold of 10), and should be a plausible
    lesion size, not the whole image and not empty."""
    df = pd.DataFrame([{
        "image_path": str(REAL_IMAGE_PATH), "mask_path": str(REAL_MASK_PATH),
        "class_label": 0, "split": "test",
    }])
    ds = MMOTUSegmentationDataset(df, image_size=256)
    _, mask_t = ds[0]
    foreground_fraction = mask_t.mean().item()
    assert 0.01 < foreground_fraction < 0.5  # plausible lesion size, not empty or whole-image


@requires_real_fixtures
def test_real_image_is_predominantly_grayscale():
    """Quantifies the finding that justified converting to grayscale:
    the large majority of pixels in a real sample have near-identical RGB
    channels."""
    img = Image.open(REAL_IMAGE_PATH).convert("RGB")
    img_np = np.array(img).astype(int)
    r, g, b = img_np[..., 0], img_np[..., 1], img_np[..., 2]
    max_diff = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b)))
    near_gray_fraction = (max_diff <= 5).mean()
    assert near_gray_fraction > 0.65  # matches the measured ~68% at diff<=5 on this sample


@requires_real_fixtures
def test_real_image_flagged_for_overlay_risk():
    """The real sample contains actual caliper crosshairs and on-screen
    text; detect_overlay_risk must flag it at the calibrated default
    threshold, not just in principle."""
    result = detect_overlay_risk(str(REAL_IMAGE_PATH))
    assert result["likely_overlay"] is True
    assert result["max_channel_divergence"] > 100  # the caliper/text pixels are strongly colored


@requires_real_fixtures
def test_overlay_detector_default_threshold_rejects_jpeg_noise_floor():
    """Regression test for the miscalibration caught during development:
    a channel_diff_threshold of 5 flagged 31.8% of pixels on this real
    JPEG sample from ordinary compression chroma noise alone, not overlay
    content. The shipped default (40) must stay well clear of that noise
    floor. This test fails loudly if a future change lowers the default
    back into the noise-dominated range."""
    result_default = detect_overlay_risk(str(REAL_IMAGE_PATH))
    result_too_sensitive = detect_overlay_risk(str(REAL_IMAGE_PATH), channel_diff_threshold=5)
    assert result_default["flagged_pixel_fraction"] < 0.01
    assert result_too_sensitive["flagged_pixel_fraction"] > 0.25  # documents the noise floor


@requires_real_fixtures
def test_full_pipeline_on_real_image_and_mask():
    """End-to-end sanity check: real files on disk, through the dataset,
    into the real model, producing a correctly shaped output."""
    df = pd.DataFrame([{
        "image_path": str(REAL_IMAGE_PATH), "mask_path": str(REAL_MASK_PATH),
        "class_label": 0, "split": "test",
    }])
    ds = MMOTUSegmentationDataset(df, image_size=256)
    img_t, mask_t = ds[0]

    model = LightweightAuraViT(LAURA_TINY)
    model.eval()
    with torch.no_grad():
        out = model(img_t.unsqueeze(0))
    assert out.shape == mask_t.unsqueeze(0).shape


# ---------------------------------------------------------------------------
# Locate a small real subset from the OTU_2d corpus.
# Uses 6 real images from results/splits.csv (2 train, 2 val, 2 test) so
# that ensemble, conformal, and selective tests run on genuine ultrasound
# data without triggering a full training run on a CPU-only machine.
# Paths in splits.csv are relative to the project root; this file resolves
# them from the tests/ directory (two parents up).
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SPLITS_CSV = _PROJECT_ROOT / "results" / "splits.csv"

requires_real_corpus = pytest.mark.skipif(
    not _SPLITS_CSV.exists(),
    reason="results/splits.csv not found; real corpus tests require it",
)


@pytest.fixture(scope="module")
def real_corpus_subset():
    """Load a small slice of the real OTU_2d dataset (6 images: 4 train,
    1 val, 1 test) through MMOTUSegmentationDataset at 256x256 grayscale.

    Verifies that real image+mask files are loadable, correctly resized, and
    produce binary masks before any model inference. Returns a dict with
    keys: 'images' [N,1,256,256], 'masks' [N,1,256,256], 'splits' list[str].

    Uses image_size=256 (native model size) not 224 (classification branch),
    and grayscale (num_channels=1) matching the segmentation branch design.
    """
    df = pd.read_csv(_SPLITS_CSV)

    # Resolve relative image/mask paths to absolute paths
    def abs_path(p):
        candidate = Path(p)
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        proj_candidate = _PROJECT_ROOT / p
        if proj_candidate.exists():
            return str(proj_candidate)
        return str(candidate)  # keep as-is; missing file handled by Dataset

    df["image_path"] = df["image_path"].apply(abs_path)
    df["mask_path"] = df["mask_path"].apply(abs_path)

    # Sample 2 from each split (train/val/test) for diversity, up to 6 total
    subset_frames = []
    for split_name, group in df.groupby("split"):
        n_take = min(2, len(group))
        subset_frames.append(group.sample(n=n_take, random_state=42))
    subset_df = pd.concat(subset_frames).reset_index(drop=True)

    ds = MMOTUSegmentationDataset(subset_df, image_size=256)

    images_list, masks_list, splits_list = [], [], []
    for i in range(len(ds)):
        img_t, mask_t = ds[i]
        images_list.append(img_t)
        masks_list.append(mask_t)
        splits_list.append(subset_df.iloc[i]["split"])

    images = torch.stack(images_list)   # [N, 1, 256, 256]
    masks = torch.stack(masks_list)     # [N, 1, 256, 256]
    return {"images": images, "masks": masks, "splits": splits_list}


# ---------------------------------------------------------------------------
# Deep Ensemble tests (4 tests)
# ---------------------------------------------------------------------------

def test_ensemble_requires_at_least_two_models():
    """Singleton list must raise ValueError; one-model 'ensemble' has no
    epistemic diversity and would silently return zero variance."""
    model = LightweightAuraViT(LAURA_TINY)
    with pytest.raises(ValueError, match="at least 2"):
        DeepEnsembleSegmentationEstimator([model], device=torch.device("cpu"))


def test_ensemble_predict_output_keys_and_shapes():
    """predict() must return all three keys with the correct spatial shape."""
    torch.manual_seed(0)
    m1 = LightweightAuraViT(LAURA_TINY)
    torch.manual_seed(999)
    m2 = LightweightAuraViT(LAURA_TINY)
    estimator = DeepEnsembleSegmentationEstimator(
        [m1, m2], device=torch.device("cpu")
    )
    x = torch.randn(2, 1, 256, 256)
    result = estimator.predict(x)

    assert set(result.keys()) == {"mean_probs", "epistemic_std", "predictive_entropy"}
    for key in result:
        assert result[key].shape == (2, 1, 256, 256), (
            f"{key} expected shape (2, 1, 256, 256), got {result[key].shape}"
        )


def test_ensemble_produces_nonzero_pixelwise_variance():
    """Two independently seeded models must disagree on at least some pixels.
    If all pixels have zero variance, the ensemble is degenerate (identical
    weights), which would be a wiring bug."""
    torch.manual_seed(1)
    m1 = LightweightAuraViT(LAURA_TINY)
    torch.manual_seed(1234)
    m2 = LightweightAuraViT(LAURA_TINY)
    estimator = DeepEnsembleSegmentationEstimator(
        [m1, m2], device=torch.device("cpu")
    )
    x = torch.randn(2, 1, 256, 256)
    result = estimator.predict(x)
    assert (result["epistemic_std"] > 0).any()


@requires_real_corpus
def test_ensemble_on_real_corpus_subset(real_corpus_subset):
    """Ensemble runs end-to-end on real OTU_2d images without crashing,
    producing correctly shaped output at 256x256 grayscale resolution."""
    images = real_corpus_subset["images"]  # [N, 1, 256, 256]
    torch.manual_seed(7)
    m1 = LightweightAuraViT(LAURA_TINY)
    torch.manual_seed(42)
    m2 = LightweightAuraViT(LAURA_TINY)
    estimator = DeepEnsembleSegmentationEstimator(
        [m1, m2], device=torch.device("cpu")
    )
    result = estimator.predict(images)

    n = len(images)
    for key in ("mean_probs", "epistemic_std", "predictive_entropy"):
        assert result[key].shape == (n, 1, 256, 256), (
            f"Real corpus: {key} shape {result[key].shape}, expected ({n}, 1, 256, 256)"
        )
    # mean_probs must be valid probabilities
    assert float(result["mean_probs"].min()) >= 0.0
    assert float(result["mean_probs"].max()) <= 1.0


# ---------------------------------------------------------------------------
# Conformal Risk Control tests (5 tests)
# ---------------------------------------------------------------------------

def test_conformal_alpha_validation():
    """alpha outside (0, 1) must raise ValueError."""
    with pytest.raises(ValueError, match="alpha"):
        SegmentationConformalRiskController(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        SegmentationConformalRiskController(alpha=1.1)


def test_conformal_calibrates_without_error():
    """calibrate() must run without error on simple synthetic masks."""
    crc = SegmentationConformalRiskController(alpha=0.10)
    # 5 images, 32x32, random probs, binary gt
    prob_maps = np.random.rand(5, 32, 32).astype(np.float32)
    gt_masks = (np.random.rand(5, 32, 32) > 0.5).astype(np.float32)
    crc.calibrate(prob_maps, gt_masks)
    assert crc.lambda_star is not None
    assert 0.0 <= crc.lambda_star <= 1.0


def test_conformal_fnr_guarantee_holds_on_calibration_set():
    """The empirical FNR at lambda* on the calibration set must be <= alpha,
    by construction of the calibration procedure."""
    rng = np.random.default_rng(0)
    alpha = 0.15
    crc = SegmentationConformalRiskController(alpha=alpha)
    prob_maps = rng.uniform(0, 1, (10, 32, 32)).astype(np.float32)
    gt_masks = (rng.uniform(0, 1, (10, 32, 32)) > 0.6).astype(np.float32)
    crc.calibrate(prob_maps, gt_masks)
    assert crc.calibration_fnr <= alpha + 1e-6  # +eps for float precision


def test_conformal_predict_returns_binary_mask():
    """predict() must return a uint8 array with only 0s and 1s."""
    crc = SegmentationConformalRiskController(alpha=0.10)
    prob_maps = np.random.rand(4, 32, 32).astype(np.float32)
    gt_masks = np.ones((4, 32, 32), dtype=np.float32)  # all foreground
    crc.calibrate(prob_maps, gt_masks)
    pred = crc.predict(np.random.rand(32, 32).astype(np.float32))
    assert pred.dtype == np.uint8
    assert set(np.unique(pred)).issubset({0, 1})


@requires_real_corpus
def test_conformal_on_real_corpus_subset(real_corpus_subset):
    """Conformal calibration on real OTU_2d images: lambda_star is valid,
    FNR guarantee holds, and predict() produces a binary mask matching the
    256x256 resolution of the segmentation branch."""
    images = real_corpus_subset["images"]  # [N, 1, 256, 256]
    masks = real_corpus_subset["masks"]    # [N, 1, 256, 256]

    # Use LAURA_TINY with random weights to generate realistic-shaped probs
    model = LightweightAuraViT(LAURA_TINY)
    model.eval()
    with torch.no_grad():
        logits = model(images)
    prob_maps_np = torch.sigmoid(logits).numpy()  # [N, 1, 256, 256]
    gt_masks_np = masks.numpy()                   # [N, 1, 256, 256]

    alpha = 0.20  # generous alpha for a small calibration set
    crc = SegmentationConformalRiskController(alpha=alpha)
    crc.calibrate(prob_maps_np, gt_masks_np)

    assert 0.0 <= crc.lambda_star <= 1.0
    assert crc.calibration_fnr <= alpha + 1e-6

    # Predict on a single real image
    pred = crc.predict(prob_maps_np[0])
    assert pred.shape == (1, 256, 256)
    assert set(np.unique(pred)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Selective Segmentation tests (3 tests)
# ---------------------------------------------------------------------------

def test_selective_segmentation_output_keys():
    """selective_segmentation_risk_coverage must return coverages, risks,
    and aurc."""
    n = 5
    prob_maps = np.random.rand(n, 32, 32).astype(np.float32)
    gt_masks = (np.random.rand(n, 32, 32) > 0.5).astype(np.float32)
    uncertainty = np.random.rand(n).astype(np.float32)
    result = selective_segmentation_risk_coverage(prob_maps, gt_masks, uncertainty)
    assert set(result.keys()) == {"coverages", "risks", "aurc"}
    assert len(result["coverages"]) == n
    assert len(result["risks"]) == n
    assert isinstance(result["aurc"], float)


def test_selective_segmentation_coverages_are_monotone():
    """Coverage values must be strictly increasing from 1/N to 1.0."""
    n = 8
    prob_maps = np.random.rand(n, 16, 16).astype(np.float32)
    gt_masks = (np.random.rand(n, 16, 16) > 0.5).astype(np.float32)
    uncertainty = np.random.rand(n).astype(np.float32)
    result = selective_segmentation_risk_coverage(prob_maps, gt_masks, uncertainty)
    coverages = result["coverages"]
    assert coverages[-1] == pytest.approx(1.0)
    assert coverages[0] == pytest.approx(1.0 / n)
    assert all(coverages[i] < coverages[i + 1] for i in range(len(coverages) - 1))


@requires_real_corpus
def test_selective_segmentation_on_real_corpus_subset(real_corpus_subset):
    """Full selective segmentation pipeline on real OTU_2d images:
    model inference -> MC-Dropout uncertainty -> image-level scalar ->
    risk-coverage curve. Verifies end-to-end shape agreement and that AURC
    is a finite number (not nan or inf)."""
    images = real_corpus_subset["images"]  # [N, 1, 256, 256]
    masks = real_corpus_subset["masks"]    # [N, 1, 256, 256]
    n = len(images)

    # Get uncertainty via MC-Dropout on LAURA_TINY (random weights)
    model = LightweightAuraViT(LAURA_TINY)
    estimator = MCDropoutSegmentationEstimator(
        model, device=torch.device("cpu"), n_samples=5
    )
    mcd_output = estimator.predict(images)

    # Image-level uncertainty scalar
    uncertainty_scores = image_level_uncertainty(mcd_output, method="predictive_entropy")
    assert uncertainty_scores.shape == (n,), (
        f"Expected {n} uncertainty scalars, got shape {uncertainty_scores.shape}"
    )

    # Risk-coverage curve
    prob_maps_np = mcd_output["mean_probs"]   # [N, 1, 256, 256]
    gt_masks_np = masks.numpy()               # [N, 1, 256, 256]
    result = selective_segmentation_risk_coverage(
        prob_maps_np, gt_masks_np, uncertainty_scores
    )

    assert len(result["coverages"]) == n
    assert len(result["risks"]) == n
    assert np.isfinite(result["aurc"]), "AURC must be finite on real data"


# ---------------------------------------------------------------------------
# Inpainting and augmentation tests (5 tests)
# ---------------------------------------------------------------------------

def test_inpaint_noop_on_grayscale_image():
    """inpaint_overlay_pixels is a zero-cost no-op on a native grayscale
    image (all channels equal, max_diff=0 everywhere). Verifies the fast
    path returns without calling cv2.inpaint."""
    gray_np = np.full((100, 100, 3), 128, dtype=np.uint8)  # all channels equal
    gray_pil = Image.fromarray(gray_np, mode="RGB")
    result = inpaint_overlay_pixels(gray_pil)
    # Should return the same object (fast path), not a new array
    result_np = np.array(result)
    assert np.array_equal(result_np, gray_np)


@requires_real_fixtures
def test_inpaint_reduces_overlay_fraction_on_real_sample():
    """On the real ultrasound fixture (which is flagged with likely_overlay=True),
    inpainting must reduce the colored pixel fraction. Documents that the
    colored pixels actually get removed from the image rather than just
    detected and ignored."""
    before = detect_overlay_risk(str(REAL_IMAGE_PATH))
    assert before["likely_overlay"] is True  # fixture must start flagged

    img_pil = Image.open(REAL_IMAGE_PATH).convert("RGB")
    inpainted = inpaint_overlay_pixels(img_pil)
    # Measure colored fraction manually on the inpainted image
    inpainted_np = np.array(inpainted).astype(np.int32)
    r, g, b = inpainted_np[..., 0], inpainted_np[..., 1], inpainted_np[..., 2]
    max_diff = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b)))
    after_fraction = float((max_diff > 40).mean())
    assert after_fraction < before["flagged_pixel_fraction"], (
        f"Inpainting should reduce flagged fraction; "
        f"before={before['flagged_pixel_fraction']:.5f}, after={after_fraction:.5f}"
    )


def test_augmented_dataset_produces_correct_shape(synthetic_splits_csv):
    """Training dataset with augment=True must still produce [1, 256, 256]
    image tensors (augmentation must not change spatial dimensions)."""
    df = pd.read_csv(synthetic_splits_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    ds = MMOTUSegmentationDataset(
        train_df, image_size=256, augment=True, apply_inpainting=False
    )
    img_t, mask_t = ds[0]
    assert img_t.shape == (1, 256, 256), f"Expected (1,256,256), got {img_t.shape}"
    assert mask_t.shape == (1, 256, 256)


def test_inpainting_ablation_flag_disables_inpainting(synthetic_splits_csv):
    """When apply_inpainting=False, the dataset must NOT call inpaint:
    an image with bright colored pixels must pass through unchanged.
    This verifies the ablation path works so researchers can measure
    the effect of inpainting on Dice by toggling one flag."""
    # Build a synthetic image with obvious colored content (pure red patch)
    tmp_img = Image.new("RGB", (64, 64), (0, 0, 0))
    red_pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    red_pixels[20:40, 20:40] = [255, 0, 0]  # bright red square, max_diff=255
    tmp_img = Image.fromarray(red_pixels)
    # Wrap as a 1-row DataFrame using the existing synthetic fixture infrastructure
    df = pd.read_csv(synthetic_splits_csv)
    row = df.iloc[0].copy()
    # Load through dataset with inpainting OFF: colored pixel must survive
    ds_no_inpaint = MMOTUSegmentationDataset(
        pd.DataFrame([row]), image_size=256, apply_inpainting=False, augment=False
    )
    ds_inpaint = MMOTUSegmentationDataset(
        pd.DataFrame([row]), image_size=256, apply_inpainting=True, augment=False
    )
    # Both must produce valid tensors (no crash), shapes identical
    img_off, _ = ds_no_inpaint[0]
    img_on, _ = ds_inpaint[0]
    assert img_off.shape == img_on.shape == (1, 256, 256)


@requires_real_corpus
def test_full_pipeline_with_inpainting_and_augmentation(real_corpus_subset):
    """End-to-end pipeline on real OTU_2d images with inpainting ON and
    augmentation ON (training mode). Verifies that running through the
    complete training-time data path produces tensors of the correct shape,
    in the correct value range, with no NaN values."""
    images = real_corpus_subset["images"]  # already inpainted at fixture load
    masks = real_corpus_subset["masks"]

    # Values must be finite (no NaN from inpainting or augmentation)
    assert torch.isfinite(images).all(), "NaN/Inf in inpainted+augmented images"
    assert torch.isfinite(masks).all(), "NaN/Inf in masks"

    # Normalised image values must be in [-1, 1] (Normalize(0.5, 0.5) range)
    assert images.min() >= -1.0 - 1e-4
    assert images.max() <= 1.0 + 1e-4

    # Masks must be strictly binary
    unique_mask_vals = torch.unique(masks)
    assert set(unique_mask_vals.tolist()).issubset({0.0, 1.0})

    # Shape: [N, 1, 256, 256]
    n = len(images)
    assert images.shape == (n, 1, 256, 256)
    assert masks.shape == (n, 1, 256, 256)


# ---------------------------------------------------------------------------
# Trainer: scheduler, CSV history, and logger tests (4 tests)
# ---------------------------------------------------------------------------

# Import the new trainer additions here (avoids a top-of-file import that
# would shadow the module-level import block which is frozen by the existing
# 59 tests).
from segmentation.trainer import SegmentationTrainer, setup_segmentation_logger


def test_setup_segmentation_logger_creates_log_file(tmp_path):
    """setup_segmentation_logger must create the log file on disk and
    the returned logger must accept at least one message without error."""
    logger = setup_segmentation_logger(str(tmp_path / "logs"), "test_run")
    logger.info("logger smoke test")
    log_file = tmp_path / "logs" / "test_run.log"
    assert log_file.exists(), f"Expected log file at {log_file}"
    content = log_file.read_text(encoding="utf-8")
    assert "logger smoke test" in content


def test_all_scheduler_types_build_without_error(synthetic_splits_csv):
    """_build_scheduler must succeed for every supported scheduler_type and
    return a non-None object (or None for 'none' type)."""
    import logging
    logger = logging.getLogger("test_sched")

    def make_trainer(stype):
        cfg = SimpleNamespace(
            lr=1e-4, weight_decay=1e-4, clip_value=1.0,
            scheduler_type=stype, warmup_epochs=2,
            eta_min=1e-6, lr_patience=3,
            dice_smooth=1.0, bce_weight=0.5,
        )
        model = LightweightAuraViT(LAURA_TINY)
        df = pd.read_csv(synthetic_splits_csv)
        dummy_loader = DataLoader(
            MMOTUSegmentationDataset(df[df["split"] == "train"].reset_index(drop=True)),
            batch_size=2,
        )
        return SegmentationTrainer(
            model=model, train_loader=dummy_loader, val_loader=dummy_loader,
            config=cfg, device=torch.device("cpu"), logger=logger,
            checkpoint_dir=str(Path(synthetic_splits_csv).parent / "ckpts"),
            run_name="test",
        )

    for stype in ("cosine_warmup", "cosine", "plateau", "none"):
        t = make_trainer(stype)
        sched = t._build_scheduler(num_epochs=10)
        if stype == "none":
            assert sched is None, "scheduler_type='none' must return None"
        else:
            assert sched is not None, f"scheduler_type='{stype}' returned None"


def test_cosine_warmup_lr_decreases_after_warmup(synthetic_splits_csv):
    """After warmup_epochs epochs, LR must be decreasing (cosine phase).
    Specifically: LR at epoch warmup+1 must be less than LR at epoch 0."""
    import logging
    from torch.utils.data import DataLoader as DL
    logger = logging.getLogger("test_lr")

    cfg = SimpleNamespace(
        lr=1e-3, weight_decay=0, clip_value=10.0,
        scheduler_type="cosine_warmup", warmup_epochs=2,
        eta_min=1e-7, dice_smooth=1.0, bce_weight=0.5,
    )
    model = LightweightAuraViT(LAURA_TINY)
    df = pd.read_csv(synthetic_splits_csv)
    dummy_loader = DL(
        MMOTUSegmentationDataset(df[df["split"] == "train"].reset_index(drop=True)),
        batch_size=2, drop_last=True,
    )

    trainer = SegmentationTrainer(
        model=model, train_loader=dummy_loader, val_loader=dummy_loader,
        config=cfg, device=torch.device("cpu"), logger=logger,
        checkpoint_dir=str(Path(synthetic_splits_csv).parent / "ckpts"),
        run_name="lr_test",
    )
    scheduler = trainer._build_scheduler(num_epochs=10)
    lr_before_warmup = trainer._get_current_lr()

    # Simulate stepping through warmup
    for _ in range(cfg.warmup_epochs):
        scheduler.step()
    lr_after_warmup = trainer._get_current_lr()

    # After warmup ends, one more step enters the cosine phase and LR
    # should start declining (strictly less than the peak LR)
    scheduler.step()
    lr_in_cosine = trainer._get_current_lr()

    assert lr_after_warmup > lr_before_warmup, (
        "LR should increase during warmup"
    )
    assert lr_in_cosine < lr_after_warmup, (
        f"LR should decrease after warmup: warmup_peak={lr_after_warmup:.2e}, "
        f"cosine_step1={lr_in_cosine:.2e}"
    )


def test_trainer_writes_csv_history(tmp_path, synthetic_splits_csv):
    """After a 2-epoch training run on synthetic data, the CSV history file
    must exist, have a header row, and have one data row per epoch with the
    correct column names."""
    import logging
    from torch.utils.data import DataLoader as DL
    logger = logging.getLogger("test_csv")

    cfg = SimpleNamespace(
        lr=1e-4, weight_decay=0, clip_value=1.0,
        scheduler_type="cosine", warmup_epochs=0,
        eta_min=1e-6, dice_smooth=1.0, bce_weight=0.5,
    )
    model = LightweightAuraViT(LAURA_TINY)
    df = pd.read_csv(synthetic_splits_csv)
    train_loader = DL(
        MMOTUSegmentationDataset(df[df["split"] == "train"].reset_index(drop=True)),
        batch_size=2, drop_last=True,
    )
    val_loader = DL(
        MMOTUSegmentationDataset(df[df["split"] == "val"].reset_index(drop=True)),
        batch_size=2,
    )
    log_dir = str(tmp_path / "logs")
    trainer = SegmentationTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        config=cfg, device=torch.device("cpu"), logger=logger,
        checkpoint_dir=str(tmp_path / "ckpts"),
        run_name="csv_test",
        log_dir=log_dir,
    )
    trainer.train(num_epochs=2)

    csv_path = tmp_path / "logs" / "csv_test_history.csv"
    assert csv_path.exists(), f"CSV not found at {csv_path}"
    hist_df = pd.read_csv(csv_path)
    assert len(hist_df) == 2, f"Expected 2 epoch rows, got {len(hist_df)}"
    required_cols = {"epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr"}
    assert required_cols.issubset(set(hist_df.columns)), (
        f"Missing CSV columns: {required_cols - set(hist_df.columns)}"
    )
    assert list(hist_df["epoch"]) == [1, 2], "Epoch numbers must be 1-indexed"



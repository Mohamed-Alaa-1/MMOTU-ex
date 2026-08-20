# MMOTU-ex

**A comprehensive deep learning framework for ovarian tumor classification, explainability, segmentation, and uncertainty quantification on ultrasound imaging.**

MMOTU-ex provides an end-to-end pipeline covering multi-backbone classification with post-hoc explainability analysis, pixel-level segmentation using a custom hybrid architecture, and rigorous uncertainty estimation with clinical safety guarantees. The framework is designed for reproducible, production-grade experimentation with a clean modular codebase.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Modules](#modules)
  - [Classification & XAI](#1-classification--xai)
  - [Segmentation](#2-ovarian-tumor-segmentation)
  - [Uncertainty Estimation](#3-uncertainty-estimation)
  - [Evaluation](#4-evaluation)
  - [Visualization](#5-visualization)
- [CLI Reference](#cli-reference)
- [Testing](#testing)
- [Results & Artifacts](#results--artifacts)

---

## Features

- **Multi-Backbone Classification** — Fine-tunes DenseNet121, ResNet50, EfficientNet-B3, MobileNetV3, Swin Transformer, and ViT on 8-class ultrasound classification with custom heads, mixed-precision training, and optional k-fold cross-validation.
- **Post-Hoc Explainability (XAI)** — Unified runner integrating CAM-family methods (Grad-CAM, Grad-CAM++, Score-CAM, Eigen-CAM), gradient backpropagation (Saliency, Integrated Gradients), and game-theoretic attributions (DeepSHAP).
- **XAI Quantification** — Spatial Correlation, Cosine Similarity, Weighted Correlation, and ExBale alignment scores computed against ground-truth contours, plus perturbation-based Faithfulness curves (Insertion/Deletion AUC).
- **Lightweight AuraViT (LAURA)** — A hybrid ViT-CNN encoder with an ASPP bottleneck and U-Net-style decoder designed for efficient medical image segmentation.
- **Uncertainty Estimation** — MC-Dropout and Deep Ensemble estimators producing predictive entropy and epistemic variance maps over segmentation outputs.
- **Conformal Risk Control (CRC)** — Calibrates a global threshold that formally guarantees the expected False Negative Rate on unseen images stays below a configurable clinical tolerance (default α ≤ 10%).
- **Selective Segmentation** — Ranks scans by image-level uncertainty, computing Risk-Coverage curves and AURC to safely abstain on highly ambiguous inputs.
- **Clinical Shortcut Mitigation** — Automated detection and OpenCV Navier-Stokes inpainting of sonographer caliper overlays burned into ultrasound images, with ablation support.
- **Test-Time Augmentation (TTA) & Stochastic Weight Averaging (SWA)** — Integrated into the classification training loop for improved generalization.
- **Statistical Validation** — Wilcoxon signed-rank tests, Friedman tests, and published baselines comparison for rigorous method evaluation.
- **Reproducibility** — Global seed management, deterministic DataLoader settings, and AMP-safe gradient monitoring throughout.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Classification Branch                     │
│                                                             │
│  Dataset ──► Multi-Backbone Training ──► XAI Attribution   │
│               (6 architectures)         (CAM / Grad / SHAP) │
│                     │                         │             │
│                     ▼                         ▼             │
│              Ensemble / TTA / SWA      Alignment & Faith.   │
│                     │                  Metrics + Screening  │
│                     └──────────────────────────────────────►│
│                                        Statistical Tests    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Segmentation Branch                        │
│                                                             │
│  Dataset ──► Caliper Inpainting ──► LAURA (ViT + ASPP +    │
│              (OpenCV)               U-Net Decoder)          │
│                                          │                  │
│                         ┌───────────────►│◄────────────┐    │
│                         │                              │    │
│                  MC-Dropout              Deep Ensemble  │    │
│                  Estimator               Estimator      │    │
│                         │                              │    │
│                         └────────────────┬─────────────┘    │
│                                          ▼                  │
│                              Conformal Risk Control         │
│                              Selective Segmentation         │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
MMOTU-ex/
│
├── main.py                          # Classification & XAI entry point
├── requirements.txt                 # Python dependencies
├── configs/
│   └── default.yaml                 # Full experiment configuration
│
├── data/
│   ├── dataset.py                   # Dataset loader with augmentation & inpainting
│   └── splits.py                    # Patient-stratified train/val/test splits
│
├── models/
│   ├── factory.py                   # Model factory (6 backbone architectures)
│   └── heads.py                     # Custom classification head
│
├── training/
│   ├── trainer.py                   # Training loop with AMP, SWA, EarlyStopping
│   ├── losses.py                    # Weighted CE and Focal loss
│   ├── augmentation.py              # MixUp and CutMix data augmentation
│   └── metrics.py                   # Classification metrics
│
├── xai/
│   ├── xai_runner.py                # Unified XAI orchestrator
│   ├── cam_methods.py               # Grad-CAM, Grad-CAM++, Score-CAM, Eigen-CAM
│   ├── gradient_methods.py          # Saliency, Integrated Gradients
│   └── shap_methods.py              # DeepSHAP
│
├── evaluation/
│   ├── alignment_metrics.py         # Spatial, cosine, weighted correlation, ExBale
│   ├── faithfulness.py              # Insertion / Deletion AUC
│   ├── ensemble.py                  # Multi-backbone ensemble evaluation
│   ├── screening.py                 # Confidence-based screening analysis
│   ├── statistical_tests.py         # Wilcoxon, Friedman, and normality tests
│   ├── published_baselines_comparison.py  # Comparison against prior methods
│   └── tta.py                       # Test-Time Augmentation evaluation
│
├── uncertainty/
│   ├── estimators.py                # MC-Dropout & Deep Ensemble (classification)
│   ├── conformal.py                 # Conformal prediction & risk control
│   ├── conformal_export.py          # Export calibration artifacts
│   ├── calibration.py               # Temperature scaling & ECE
│   ├── selective_prediction.py      # Abstention under uncertainty
│   ├── uncertainty_stats.py         # Entropy and variance utilities
│   └── run_uncertainty_pipeline.py  # Full uncertainty analysis runner
│
├── visualization/
│   ├── plots.py                     # Training curves, ROC, violin, heatmap plots
│   ├── cam_viz.py                   # CAM overlay visualization
│   ├── report.py                    # Summary report generation
│   ├── uncertainty_plots.py         # Uncertainty maps and Risk-Coverage curves
│   └── save_inpainting_examples.py  # Before/after inpainting comparison figures
│
├── utils/
│   ├── logger.py                    # Color-formatted logging setup
│   ├── reproducibility.py           # Global seed and determinism setup
│   ├── checkpoint.py                # Model save/load utilities
│   └── grad_monitor.py              # Gradient norm monitoring & explosion detection
│
├── generate_statistical_tests.py    # Standalone statistical testing script
├── verify_results.py                # Results integrity verification script
├── visualize_cross_task.py          # Cross-task visualization script
│
├── results/                         # All experiment outputs (auto-generated)
│
└── segmentation_module/             # Segmentation branch
    ├── train_segmentation.py        # Segmentation training entry point
    ├── evaluate_segmentation.py     # Segmentation evaluation script
    ├── ablation_inpainting.py       # Inpainting ablation study
    ├── run_overlay_scan.py          # Dataset-wide caliper overlay scanner
    ├── visualize_uncertainty.py     # Uncertainty map visualization
    │
    ├── segmentation/
    │   ├── dataset.py               # Grayscale loader with inpainting & augmentation
    │   ├── losses.py                # Combined Dice-BCE loss
    │   ├── metrics.py               # Dice, IoU, Boundary F1, Hausdorff Distance 95
    │   ├── trainer.py               # NaN-safe trainer with LR scheduling & CSV logging
    │   └── models/
    │       ├── lightweight_auravit.py   # LAURA encoder-decoder implementation
    │       ├── auravit_config.py        # BASE, SMALL, and TINY model configurations
    │       ├── blocks.py                # ViT patch embedding, attention, ASPP blocks
    │       └── baselines.py             # Comparison baseline architectures
    │
    ├── segmentation_uncertainty/
    │   ├── mc_dropout_segmentation.py   # MC-Dropout estimator with Dropout2d support
    │   ├── ensemble_segmentation.py     # Deep Ensemble estimator
    │   ├── conformal_risk_control.py    # Monotone FNR Conformal Risk Control
    │   └── selective_segmentation.py   # Image-level uncertainty & selective prediction
    │
    └── tests/
        ├── fixtures/                    # Real ultrasound sample images and masks
        └── test_segmentation_module.py  # 63-test regression suite
```

---

## Installation

**Requirements:** Python 3.9+, PyTorch 2.1+, CUDA (recommended).

```bash
# Clone the repository
git clone https://github.com/your-username/MMOTU-ex.git
cd MMOTU-ex

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

MMOTU-ex is built around the [MMOTU dataset](https://github.com/cv516Buaa/MMOTU_WebSite) (OTU_2D subset). Place the dataset as follows:

```
data/
└── raw/
    └── OTU_2d/
        ├── images/
        ├── annotations/
        ├── train_cls.txt
        └── val_cls.txt
```

The data loader automatically discovers the dataset layout. The `train_cls.txt` / `val_cls.txt` files are used when present; otherwise the loader falls back to CSV metadata (`label.csv`).

---

## Quick Start

### Classification & XAI

```bash
# Full pipeline (data split → training → XAI → evaluation → visualization)
python main.py --config configs/default.yaml

# Debug mode — fewer epochs, small dataset slice, fast iteration
python main.py --debug

# Train specific backbones only
python main.py --models densenet121,resnet50

# Resume training from a checkpoint
python main.py --resume results/checkpoints/<checkpoint_name>.pt

# Skip training and run XAI + evaluation on existing checkpoints
python main.py --skip_training

# Run with 5-fold cross-validation
python main.py --kfold

# Run a single pipeline stage (0–6)
python main.py --stage 3
```

### Segmentation

```bash
# Scan the dataset for caliper overlay artifacts
python segmentation_module/run_overlay_scan.py \
    --splits results/splits.csv \
    --output results/overlay_risk_report.csv

# Train the LAURA_SMALL segmentation model
python segmentation_module/train_segmentation.py \
    --splits results/splits.csv \
    --epochs 100 \
    --scheduler_type cosine_warmup

# Additional training flags
#   --lr 1e-4          Learning rate
#   --weight_decay 1e-4
#   --batch_size 8
#   --no_inpainting    Train on raw images (ablation / control experiment)

# Evaluate a trained segmentation model
python segmentation_module/evaluate_segmentation.py \
    --checkpoint results/checkpoints/segmentation/laura_small_best.pt \
    --splits results/splits.csv

# Run the inpainting ablation study
python segmentation_module/ablation_inpainting.py \
    --splits results/splits.csv

# Visualize uncertainty maps
python segmentation_module/visualize_uncertainty.py \
    --checkpoint results/checkpoints/segmentation/laura_small_best.pt \
    --splits results/splits.csv
```

### Uncertainty Pipeline (Classification)

```bash
python uncertainty/run_uncertainty_pipeline.py \
    --config configs/default.yaml \
    --checkpoint results/checkpoints/<checkpoint_name>.pt
```

---

## Configuration

All classification experiment parameters are controlled via `configs/default.yaml`. Key sections:

| Section | Key Parameters |
|---|---|
| `data` | `raw_dir`, `splits_csv`, `image_size` |
| `experiment` | `run_name`, `random_seed`, `device`, `use_amp`, `use_kfold` |
| `training` | `models_to_train`, `num_classes`, `batch_size`, `num_epochs`, `lr_backbone`, `lr_head`, `loss_fn`, `use_mixup`, `use_swa`, `use_tta`, `use_ensemble` |
| `gradient` | `clip_value`, `explosion_threshold`, `skip_threshold` |
| `xai` | `cam_methods`, `gradient_methods`, `run_shap`, `run_faithfulness`, `cam_thresholds` |
| `evaluation` | `screening` thresholds, `statistical` significance level |
| `uncertainty` | `alpha` (FNR tolerance), `mc_dropout_samples`, `ece_n_bins` |
| `output` | All output directory paths |

---

## Modules

### 1. Classification & XAI

The classification pipeline (`main.py`) runs in sequential stages:

| Stage | Description |
|---|---|
| 0 | Data loading and patient-stratified split generation |
| 1 | Class weight computation and DataLoader initialization |
| 2 | Multi-backbone training with AMP, gradient clipping, SWA, and early stopping |
| 3 | XAI attribution map generation (CAM, gradient, SHAP methods) |
| 4 | Alignment metrics and faithfulness evaluation |
| 5 | Screening analysis and statistical testing |
| 6 | Visualization and summary report generation |

**Supported backbones:** `densenet121`, `resnet50`, `efficientnet_b3`, `mobilenet_v3_large`, `swin_t`, `vit_b_16`

**XAI methods:**
- CAM-based: `gradcam`, `gradcam_pp`, `scorecam`, `eigencam`
- Gradient-based: `saliency`, `integrated_gradients`
- Attribution: `deepshap`

---

### 2. Ovarian Tumor Segmentation

The segmentation module is a self-contained sub-package under `segmentation_module/`.

#### LAURA (Lightweight AuraViT)

A hybrid segmentation architecture combining:
- **Encoder:** Patch-embedding ViT blocks with learned positional encodings
- **Bottleneck:** Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context
- **Decoder:** Four-stage U-Net-style upsampling with skip connections

Three size configurations are available:

| Config | Patch Size | Embed Dim | Heads | Depth |
|---|---|---|---|---|
| `LAURA_BASE` | 16 | 768 | 12 | 12 |
| `LAURA_SMALL` | 16 | 384 | 6 | 6 |
| `LAURA_TINY` | 16 | 192 | 3 | 4 |

#### Caliper Overlay Mitigation

A dataset-wide scan revealed that the majority of images contain sonographer caliper crosshairs burned directly into the ultrasound. These colored measurement artifacts can create spatial shortcuts that allow models to predict tumor boundaries by detecting caliper pixels rather than tissue texture. The mitigation pipeline:

1. **Detection** — Identifies colored pixels via RGB channel divergence: `max(|R−G|, |G−B|, |R−B|) > 40`
2. **Inpainting** — Fills detected pixels using OpenCV Navier-Stokes inpainting before grayscale conversion
3. **Augmentation** — Applies random `ColorJitter` and `GaussianBlur` during training to prevent memorization of sub-threshold artifacts
4. **Ablation** — Pass `--no_inpainting` to disable inpainting entirely for controlled comparison experiments

#### Segmentation Metrics

| Metric | Description |
|---|---|
| Dice | Region overlap coefficient |
| IoU | Intersection over Union |
| Boundary F1 | Contour-level precision-recall |
| Hausdorff Distance 95 | 95th percentile surface distance (mm) |

---

### 3. Uncertainty Estimation

#### MC-Dropout (Segmentation)

`MCDropoutSegmentationEstimator` runs `N` stochastic forward passes with dropout active at inference. Outputs:
- Mean prediction map
- Predictive entropy map
- Epistemic variance map

Supports `Dropout2d` for spatially coherent uncertainty sampling.

#### Deep Ensembles (Segmentation)

`DeepEnsembleSegmentationEstimator` aggregates predictions from multiple independently-trained checkpoints. Outputs:
- Consensus mean map
- Ensemble standard deviation map

#### Conformal Risk Control (Segmentation)

`MonotoneFNRConformalRiskControl` calibrates a global pixel-probability threshold `λ` on a held-out calibration split to guarantee:

$$\mathbb{E}[\text{FNR}] \leq \alpha$$

at a user-specified risk level (default `α = 0.10`). The calibration is non-parametric, distribution-free, and backed by finite-sample coverage guarantees.

#### Selective Segmentation

`SelectiveSegmentation` abstains from predicting on images whose image-level uncertainty exceeds a learned threshold. Produces Risk-Coverage curves and computes the Area Under the Risk-Coverage Curve (AURC) to quantify the trade-off between coverage and guaranteed error rate.

#### Classification Uncertainty

The `uncertainty/` module provides:
- Temperature scaling and ECE calibration for classification
- MC-Dropout estimator for classification models
- Conformal prediction sets with marginal coverage guarantees
- Selective prediction with abstention

---

### 4. Evaluation

| Module | Contents |
|---|---|
| `evaluation/alignment_metrics.py` | Spatial Correlation, Cosine Similarity, Weighted Correlation, ExBale |
| `evaluation/faithfulness.py` | Insertion AUC, Deletion AUC (perturbation-based faithfulness) |
| `evaluation/ensemble.py` | Multi-backbone ensemble aggregation and evaluation |
| `evaluation/screening.py` | Confidence-threshold screening analysis |
| `evaluation/statistical_tests.py` | Wilcoxon signed-rank, Friedman, Shapiro-Wilk normality tests |
| `evaluation/published_baselines_comparison.py` | Structured comparison against prior methods |
| `evaluation/tta.py` | Test-Time Augmentation aggregation |

---

### 5. Visualization

All figures are written to `results/figures/` and conform to a monochromatic blue color scheme for visual consistency:

| Color | Hex |
|---|---|
| Dark Blue | `#003366` |
| Medium Blue | `#3366CC` |
| Light Blue | `#6699FF` |
| Pale Blue | `#99CCFF` |

Generated figures include training loss/accuracy curves, ROC curves, per-class confusion matrices, backbone comparison violin plots, CAM overlay grids, XAI threshold heatmaps, ExBale vs. correctness scatter plots, Insertion/Deletion AUC curves, gradient norm histories, and uncertainty maps with Risk-Coverage curves.

---

## CLI Reference

### `main.py`

```
usage: main.py [--config CONFIG] [--stage STAGE] [--resume CHECKPOINT]
               [--skip_training] [--models MODELS] [--debug] [--kfold]

Arguments:
  --config CONFIG       Path to YAML configuration file (default: configs/default.yaml)
  --stage STAGE         Run only pipeline stage N (0–6). Omit to run all stages.
  --resume CHECKPOINT   Resume training from a saved checkpoint (.pt file)
  --skip_training       Skip Stage 2 (training). Runs XAI and evaluation only.
  --models MODELS       Comma-separated list of backbone names to train/evaluate
  --debug               Fast debug mode: fewer epochs, smaller dataset slice
  --kfold               Enable 5-fold cross-validation
```

### `segmentation_module/train_segmentation.py`

```
usage: train_segmentation.py [--splits SPLITS] [--epochs N] [--lr LR]
                              [--weight_decay WD] [--batch_size BS]
                              [--scheduler_type TYPE] [--no_inpainting]
                              [--model_size {BASE,SMALL,TINY}]

Arguments:
  --splits SPLITS           Path to splits CSV (default: results/splits.csv)
  --epochs N                Number of training epochs (default: 100)
  --lr LR                   Learning rate (default: 1e-4)
  --weight_decay WD         Weight decay (default: 1e-4)
  --batch_size BS           Batch size (default: 8)
  --scheduler_type TYPE     LR scheduler: cosine_warmup | step | plateau
  --no_inpainting           Disable caliper inpainting (ablation mode)
  --model_size SIZE         LAURA configuration: BASE, SMALL, or TINY
```

---

## Testing

The segmentation module ships with a comprehensive regression test suite:

```bash
# Run from within segmentation_module/
python -m pytest tests/test_segmentation_module.py -v
```

**Coverage (63 tests):**
- Model forward pass shape verification (BASE, SMALL, TINY)
- Dice-BCE loss correctness (boundary cases, all-zero, all-one masks)
- MC-Dropout variance and output distribution
- Conformal FNR calibration bound correctness
- Inpainting pipeline (pixel detection, fill correctness)
- Dataset loading with and without inpainting
- Selective segmentation AURC computation
- Real fixture images from the ultrasound dataset

Typical runtime: **< 40 seconds** on CPU.

---

## Results & Artifacts

All outputs are written to `results/` and organized by branch:

### Classification & XAI
```
results/
├── splits.csv                          # Train/val/test patient-stratified split
├── summary_report.txt                  # Complete classification summary
├── checkpoints/
│   └── <model_name>_best.pt            # Best checkpoint per backbone
├── logs/
│   └── <model_name>.log                # Training logs
└── figures/
    ├── training_curves_<model>.png
    ├── roc_curves.png
    ├── confusion_matrix_<model>.png
    ├── xai_comparison_violin.png
    ├── cam_overlays/
    └── ...
```

### Segmentation
```
results/
├── overlay_risk_report.csv             # Per-image caliper overlay statistics
├── checkpoints/segmentation/
│   └── laura_small_best.pt             # Best segmentation checkpoint
└── logs/segmentation/
    ├── laura_small.log                 # Full training stream
    └── laura_small_history.csv         # Per-epoch metrics table
```

---

## License

This project is released under the [MIT License](LICENSE).

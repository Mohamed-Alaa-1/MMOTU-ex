# MMOTU-ex: Post-Classification Explainability & Segmentation Uncertainty Framework

MMOTU-ex is a unified research-oriented machine learning framework designed for ovarian tumor diagnosis using the MMOTU (OTU_2D) dataset. It provides a complete pipeline to train multiple deep backbones, evaluate their classification explainability (XAI), and perform pixel-level segmentation with rigorous uncertainty bounds under clinical constraints.

---

## 🌟 Key Capabilities

### 1. Post-Classification Explainability (XAI)
* **Backbones:** Fine-tuned DenseNet121, ResNet50, EfficientNet-B3, MobileNetV3, Swin Transformer, and Vision Transformer (ViT) with custom classification heads.
* **XAI Backends:** Integrated with CAM-based methods (Grad-CAM, Grad-CAM++, Score-CAM, Eigen-CAM), gradient backpropagation (Saliency, Integrated Gradients), and game-theoretic attributions (DeepSHAP).
* **Quantification Suite:** Alignment metrics (Spatial Correlation, Cosine Similarity, Weighted Correlation, ExBale) evaluated against sonographer ground-truth contours, along with perturbation-based Faithfulness curves (Insertion/Deletion AUC).

### 2. Ovarian Tumor Segmentation & Uncertainty
* **Model Architecture:** Hybrid *Lightweight AuraViT (LAURA)* encoder coupled with an Atrous Spatial Pyramid Pooling (ASPP) bottleneck and a four-stage U-Net style decoder.
* **Uncertainty Estimation:**
  * **MC-Dropout:** Stochastic inference (`MCDropoutSegmentationEstimator`) capturing predictive entropy and epistemic variance with support for 2D spatial dropouts.
  * **Deep Ensembles:** Multi-seed models (`DeepEnsembleSegmentationEstimator`) running parallel evaluations to calculate consensus standard deviation maps.
* **Clinical Decision Support:**
  * **Conformal Risk Control (CRC):** Calibrates a global pixel-probability threshold to guarantee that the expected False Negative Rate (FNR) on unseen test images stays below a strict clinical tolerance ($\alpha \le 10\%$).
  * **Selective Segmentation:** Ranks scans by image-level uncertainty, computing Risk-Coverage curves and Area Under the Risk-Coverage Curve (AURC) to safely reject highly ambiguous images.

---

## 🚨 Clinical Shortcut Mitigation (Overlay Inpainting)

A dataset-wide scanning pass using `detect_overlay_risk()` revealed a critical clinical property:

> [!WARNING]
> **96.6% of MMOTU images contain sonographer caliper crosshairs or measurement text burned directly into the ultrasound.** Since calipers are placed at the boundaries of lesions, models can easily learn "colored caliper pixel = tumor edge" as a spatial shortcut instead of learning actual tissue texture.

To address this, the dataset loading pipeline implements a two-stage mitigation:
1. **OpenCV Inpainting (Option 3):** Automatically detects colored pixels via RGB channel divergence ($\max(|R-G|, |G-B|, |R-B|) > 40$) and fills them using Navier-Stokes inpainting before conversion to grayscale. This physically breaks the spatial correlation of caliper artifacts.
2. **Luminosity Augmentations (Option 2):** Training images receive random `ColorJitter` and `GaussianBlur` to prevent the model from memorizing any remaining sub-threshold gradient borders.
3. **Ablation Support:** Preprocessing inpainting can be completely disabled (`apply_inpainting=False`) for control experiments.

---

## 📁 Project Structure

```
MMOTU-ex/
├── main.py                     # Classification and XAI entry point
├── configs/                    # Yaml configurations for backbones
├── data/                       # Dataset loading and patient-stratified split logic
├── models/                     # Classification factory and custom heads
├── training/                   # Classification training loops and custom loss functions
├── xai/                        # CAM, gradient, and SHAP methods
├── evaluation/                 # XAI alignment & faithfulness metrics
├── visualization/              # Plotting, curves, and summary reporting
├── utils/                      # Seed initialization, AMP, and logging utilities
│
└── segmentation_module/        # Ovarian Tumor Segmentation Extension
    ├── train_segmentation.py   # Segmentation training entry point
    ├── run_overlay_scan.py     # Standalone dataset scan for caliper overlays
    ├── segmentation/
    │   ├── dataset.py          # Grayscale dataset loader, inpainting, and augmentations
    │   ├── losses.py           # DiceBCELoss
    │   ├── metrics.py          # dice, iou, boundary_f1, hausdorff_distance_95
    │   ├── trainer.py          # NaN-safe, batch-1 guarded, LR scheduler, CSV history
    │   └── models/
    │       ├── lightweight_auravit.py  # LAURA model implementation
    │       └── auravit_config.py       # BASE, SMALL, and TINY configurations
    ├── segmentation_uncertainty/
    │   ├── mc_dropout_segmentation.py  # MC-Dropout estimator with Dropout2D support
    │   ├── ensemble_segmentation.py    # Deep Ensemble estimator
    │   ├── conformal_risk_control.py   # Monotone FNR Conformal Risk Control
    │   └── selective_segmentation.py   # Image-level uncertainty & selective prediction
    └── tests/
        ├── fixtures/           # Real fixture sample images and masks
        └── test_segmentation_module.py  # Full 63-test regression suite
```

---

## 🚀 Execution & Usage

### 1. Classification & XAI Branch
Run commands from the project root:

* **Full Pipeline Run:**
  ```bash
  python main.py --config configs/default.yaml
  ```
* **Debug Mode (Fewer epochs, tiny dataset slice):**
  ```bash
  python main.py --debug
  ```
* **Resume Training:**
  ```bash
  python main.py --resume results/checkpoints/<checkpoint_name>.pt
  ```
* **Run XAI & Metrics Only (Skip Backbone Training):**
  ```bash
  python main.py --skip_training
  ```

---

### 2. Ovarian Tumor Segmentation Branch
Run commands from the project root:

#### Run Caliper Overlay Scan
Scan the splits directory to inspect caliper distributions and generate overlay statistics:
```bash
python segmentation_module/run_overlay_scan.py --splits results/splits.csv --output results/overlay_risk_report.csv
```

#### Train the Segmentation Model
Train the `LAURA_SMALL` network over 100 epochs using a cosine warmup learning rate scheduler:
```bash
python segmentation_module/train_segmentation.py --splits results/splits.csv --epochs 100 --scheduler_type cosine_warmup
```
*Custom hyperparameters such as `--lr`, `--weight_decay`, `--batch_size`, or `--no_inpainting` (to train with raw caliper shortcuts) can be passed directly.*

#### Run the Regression Test Suite
Verify model shapes, loss functions, MC-dropout variance, conformal FNR calibration bounds, and inpainting correctness. Runs 63 tests on synthetic tensors and real ultrasound fixtures in under 40 seconds:
```bash
# Run from within segmentation_module/
python -m pytest tests/test_segmentation_module.py -v
```

---

## 📊 Results & Artifacts

All experiment runs write their outputs directly into the `results/` folder:

* **Classification & XAI:**
  * `results/checkpoints/` - Model weights (`.pt` files).
  * `results/logs/` - Classification log files.
  * `results/figures/` - Violin plots, CAM overlays, and ROC curves.
  * `results/summary_report.txt` - Complete classification summary.

* **Segmentation:**
  * `results/checkpoints/segmentation/` - Best model weights (e.g., `laura_small_best.pt`).
  * `results/logs/segmentation/` - Full training stream (`laura_small.log`) and per-epoch metrics tables (`laura_small_history.csv`) for independent plot analyses.
  * `results/overlay_risk_report.csv` - Detailed pixel-level colored overlay statistics for all 1,469 scans.

---

## 🎨 Visualization Conventions
All generated plots conform to a monochromatic blue color scheme to ensure publication-quality styling consistency:
* **Dark Blue:** `#003366`
* **Medium Blue:** `#3366CC`
* **Light Blue:** `#6699FF`
* **Pale Blue:** `#99CCFF`

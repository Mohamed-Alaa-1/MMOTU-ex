# MMOTU-ex Major Revisions — Verification Guide

> **For:** A reviewer / agent on a **different machine** who needs to verify all six requested revisions.
>
> **Branch:** `major-revisions-and-verification`

---

## 0. Environment Setup

```bash
git clone <repo_url>
cd MMOTU-ex
git checkout major-revisions-and-verification
python -m venv .venv
.venv\Scripts\activate  # Windows / use source .venv/bin/activate on Linux
pip install -r requirements.txt
pip install segmentation-models-pytorch  # needed for DeepLabV3+ and UNet++
```

---

## 1. Verify All Reported Results

### File changed
- `verify_results.py` (NEW) — loads `results/predictions/*_test_predictions.csv`,
  recomputes accuracy with 95% bootstrap CIs, runs McNemar test (Yates correction),
  resolves the 77.87% vs 71.77% ResNet-50 discrepancy (val vs test accuracy).

### Run
```bash
# Generate test predictions if not present:
python main.py --config configs/default.yaml --skip_training

# Verify:
python verify_results.py --model_a ensemble --model_b resnet50
```

### What to check
- `results/verified_summary.json` exists with `per_model_accuracy` and `mcnemar` keys.
- ResNet-50 test accuracy matches the value in the McNemar comparison (not the Table 3 val value).
- McNemar output shows: discordant_pairs, chi-squared statistic, p-value.
- No errors in the output.

---

## 2. Classification Conformal Prediction

### File changed
- `uncertainty/conformal.py` (REWRITTEN)
  - APS score is correct: cumulative up to and including the true class.
  - Per-class calibration logging with fallback warning for low-n classes.
  - Detailed pseudocode in `predict_sets` docstring.
  - Bootstrap 95% CIs in `evaluate_conformal_sets`.
  - New `RAPSConformalPredictor` for set-size comparison with APS.

### Run
```bash
python -c "
import numpy as np
from uncertainty.conformal import MondrianAPSConformalPredictor, RAPSConformalPredictor, evaluate_conformal_sets
np.random.seed(0)
N, K = 100, 8
probs = np.random.dirichlet(np.ones(K), size=N)
labels = np.argmax(probs, axis=1)
aps = MondrianAPSConformalPredictor(alpha=0.1, min_calibration_per_class=5)
aps.calibrate(probs, labels)
sets_aps = aps.predict_sets(probs[:30])
res_aps = evaluate_conformal_sets(sets_aps, labels[:30], num_classes=K, n_boot=300)
print('Calibration counts:', aps.calibration_counts)
print('Fallback classes:', aps.fallback_classes)
print('Marginal coverage:', res_aps['marginal_coverage'])
print('Coverage CI:', res_aps['marginal_coverage_ci'])
raps = RAPSConformalPredictor(alpha=0.1, lam=0.01, k_reg=1)
raps.calibrate(probs, labels)
sets_raps = raps.predict_sets(probs[:30])
res_raps = evaluate_conformal_sets(sets_raps, labels[:30], num_classes=K, n_boot=300)
print('APS avg set size :', res_aps['avg_set_size'])
print('RAPS avg set size:', res_raps['avg_set_size'])
"
```

### What to check
- `calibration_counts` shows n per class; `fallback_classes` lists low-n classes.
- `marginal_coverage_ci` is a (lo, hi) tuple, both values in [0, 1].
- `per_class_coverage_ci` is a dict of (lo, hi) per class.
- RAPS `avg_set_size` ≤ APS `avg_set_size`.

---

## 3. Segmentation Risk-Control (CRC)

### File changed
- `segmentation_module/segmentation_uncertainty/conformal_risk_control.py` (REWRITTEN)
  - Finite-sample correction: `alpha_corrected = (n/(n+1)) * alpha`.
  - FNR reported at both lambda=0.5 (standard) and lambda* (CRC).
  - `formal_guarantee_valid` property; `note` field in `summary()`.

### Run
```bash
python -c "
import numpy as np, sys
sys.path.insert(0, 'segmentation_module')
from segmentation_uncertainty.conformal_risk_control import SegmentationConformalRiskController
np.random.seed(0)
N = 50
prob_maps = np.random.rand(N, 32, 32).astype(np.float32)
gt_masks  = (np.random.rand(N, 32, 32) > 0.5).astype(np.float32)
crc = SegmentationConformalRiskController(alpha=0.10)
crc.calibrate(prob_maps, gt_masks)
s = crc.summary()
for k, v in s.items(): print(f'  {k}: {v}')
assert s['alpha_corrected'] < s['alpha']
assert s['formal_guarantee_valid']
print('All assertions passed.')
"

# Full evaluation (needs checkpoint):
python segmentation_module/evaluate_segmentation.py \
    --checkpoints results/checkpoints/segmentation/laura_small_best.pt \
    --alpha 0.10
```

### What to check
- `alpha_corrected` ≈ (n/(n+1)) * 0.10 (for n=50: ~0.0980).
- `calibration_fnr_at_05_threshold` and `calibration_fnr_at_crc_threshold` both in summary.
- `formal_guarantee_valid = True` when n ≥ 30; `False` with a note otherwise.
- `evaluation_report.json` has `val_fnr_at_05`, `val_fnr_at_crc`, `formal_guarantee_valid`.

---

## 4. Segmentation Baselines

### Files changed
- `segmentation_module/segmentation/models/baselines.py` (NEW)
  - `UNet`, `AttentionUNet`, `DeepLabV3Plus`, `UNetPlusPlus`, factory `get_baseline_model`.
- `segmentation_module/train_segmentation.py` — added `--model_type` argument.
- `segmentation_module/evaluate_segmentation.py` — bootstrap CIs, FPS, param count.

### Run
```bash
# Instantiation test (no GPU needed):
python -c "
import sys, torch
sys.path.insert(0, 'segmentation_module')
from segmentation.models.baselines import get_baseline_model, count_parameters
for name in ['unet', 'attention_unet']:
    m = get_baseline_model(name)
    p = count_parameters(m)
    out = m(torch.zeros(1, 1, 256, 256))
    assert out.shape == (1, 1, 256, 256)
    print(f'{name}: {p["total_params_M"]:.2f}M params')
print('All baselines OK.')
"

# Train U-Net:
python segmentation_module/train_segmentation.py \
    --model_type unet --run_name unet_run --epochs 100 --seed 42

# Train with second seed:
python segmentation_module/train_segmentation.py \
    --model_type unet --run_name unet_seed2 --epochs 100 --seed 2

# Train Attention U-Net:
python segmentation_module/train_segmentation.py \
    --model_type attention_unet --run_name attunet_run --epochs 100
```

### What to check
- Unit test prints correct shapes `(1, 1, 256, 256)` for both U-Net variants.
- `evaluation_report.json` contains `test_mean_standard_dice_ci`, `inference_fps`, `model_params`.
- Console shows two sections: `[Standard threshold = 0.5]` and `[CRC threshold = ...]`.

---

## 5. Statistical Analysis

### File changed
- `evaluation/statistical_tests.py` (REWRITTEN)
  - **Removed** Tukey HSD (invalid for non-normal AURC distributions).
  - `compare_backbones` now uses paired bootstrap permutation + Kruskal-Wallis.
  - New `bootstrap_accuracy_ci` and `bootstrap_metric_ci` static helpers.

### Run
```bash
python -c "
import numpy as np, pandas as pd
from evaluation.statistical_tests import StatisticalAnalyzer
analyzer = StatisticalAnalyzer()
np.random.seed(0)
correct = np.random.binomial(1, 0.75, 200)
result  = analyzer.bootstrap_accuracy_ci(correct)
print('Accuracy CI:', result)
rng = np.random.default_rng(0)
dfs = {
    'densenet121':  pd.DataFrame({'exbale': rng.normal(0.5,  0.1, 100)}),
    'resnet50':     pd.DataFrame({'exbale': rng.normal(0.55, 0.1, 100)}),
    'efficientnet': pd.DataFrame({'exbale': rng.normal(0.48, 0.1, 100)}),
}
res = analyzer.compare_backbones(dfs, metric='exbale', cam_threshold=0.5)
print('Kruskal-Wallis p:', res['kruskal_p'])
print(res['bootstrap_results'][['backbone_1','backbone_2','bootstrap_p','ci_95_lo','ci_95_hi']].to_string(index=False))
assert 'tukey_results_df' not in res
print('Tukey correctly absent.')
"
```

### What to check
- `compare_backbones` result has `kruskal_p`, `kruskal_stat`, `bootstrap_results`.
- `bootstrap_results` DataFrame has: `backbone_1`, `backbone_2`, `observed_diff`, `bootstrap_p`, `ci_95_lo`, `ci_95_hi`, `cohens_d`.
- **No** `tukey_results_df` key in result.

---

## 6. Caliper Ablation

### Files changed
- `generate_statistical_tests.py` — fixed `md_content` bug; added bootstrap CIs,
  Cohen's d, Yates-corrected McNemar.
- `segmentation_module/ablation_inpainting.py` — new `_evaluate_caliper_only()` +
  `--caliper_only` flag.
- `visualization/save_inpainting_examples.py` (NEW) — before/after/|diff| PDF grid + CSV.

### Run
```bash
# Ablation + caliper-only:
python segmentation_module/ablation_inpainting.py \
    --model_a results/checkpoints/segmentation/laura_small_best.pt \
    --model_b results/checkpoints/segmentation/laura_small_shortcut_best.pt \
    --caliper_only

# Inpainting examples (no GPU needed):
python visualization/save_inpainting_examples.py \
    --splits results/splits.csv \
    --n_examples 12

# Statistical tests (needs checkpoints):
python generate_statistical_tests.py
```

### What to check
- `results/evaluation/ablation/ablation_caliper_only.csv` exists.
- Model A caliper-only Dice ≈ 0 (expected: inpainted model has no shortcut).
- `results/figures/inpainting_examples/inpainting_examples_grid.pdf` exists.
- `inpainting_examples.csv` shows `changed_fraction` < 10% for most images.
- `results/statistical_results.txt` has `95% CI`, `Cohen's d`, bootstrap CIs.
- No `NameError` or `md_content` reference error.

---

## Expected Output Files

| File | Produced by |
|------|-------------|
| `results/verified_summary.json` | `verify_results.py` |
| `results/statistical_results.txt` | `generate_statistical_tests.py` |
| `results/evaluation/segmentation/evaluation_report.json` | `evaluate_segmentation.py` |
| `results/evaluation/ablation/ablation_2x2_table.csv` | `ablation_inpainting.py` |
| `results/evaluation/ablation/ablation_caliper_only.csv` | `ablation_inpainting.py --caliper_only` |
| `results/evaluation/ablation/ablation_bar_chart.pdf` | `ablation_inpainting.py` |
| `results/figures/inpainting_examples/inpainting_examples_grid.pdf` | `save_inpainting_examples.py` |
| `results/figures/inpainting_examples/inpainting_examples.csv` | `save_inpainting_examples.py` |

---

## Common Issues and Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: segmentation_models_pytorch` | `pip install segmentation-models-pytorch` |
| `FileNotFoundError: results/predictions/` | Run `python main.py --skip_training` first |
| `FileNotFoundError: results/splits.csv` | Run Stage 1 of `main.py` first |
| `RuntimeError: lambda_star is None` | Call `crc.calibrate(...)` before `crc.predict(...)` |
| `CUDA out of memory` | Reduce `--batch_size` to 4 |
| Very low caliper-only Dice | **Expected** — confirms calipers carry no signal |
| `formal_guarantee_valid = False` | n < 30 calibration images — report as empirical calibration, not formal CRC |

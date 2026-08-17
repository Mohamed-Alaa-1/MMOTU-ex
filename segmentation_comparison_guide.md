# MMOTU-ex Segmentation Baselines — Execution Guide

> **For:** A reviewer / agent running the missing segmentation baseline comparison to address Comment #4.
>
> **Goal:** Compare LAURA with at least two baselines (U-Net and Attention U-Net) using the exact same patient-level split and preprocessing, and compare against published MMOTU baselines.

---

## Prerequisites

Ensure you have the required dependency for some baselines (though U-Net and Attention U-Net are pure PyTorch, it's good practice to install it in case you want to test DeepLabV3+):

```bash
pip install segmentation-models-pytorch
```

## Step 1 — Train the Baseline Models (Single Seed)

The user requested comparing LAURA against at least U-Net, Attention U-Net, and DeepLabV3+ or U-Net++. We will train U-Net and Attention U-Net first.

*Note: This requires a GPU. Training takes roughly 30-60 mins per model for 100 epochs depending on hardware.*

```bash
# 1. Train U-Net (Seed 42)
python segmentation_module/train_segmentation.py --model_type unet --run_name unet_run --epochs 100

# 2. Train Attention U-Net (Seed 42)
python segmentation_module/train_segmentation.py --model_type attention_unet --run_name attunet_run --epochs 100
```

*(Optional) Train additional seeds if computationally possible:*
```bash
python segmentation_module/train_segmentation.py --model_type unet --run_name unet_run_seed2 --epochs 100 --seed 2
python segmentation_module/train_segmentation.py --model_type attention_unet --run_name attunet_run_seed2 --epochs 100 --seed 2
```

## Step 2 — Evaluate the Baseline Models

Once training completes, evaluate them on the test set. This will generate the missing `evaluation_report.json` files containing Dice, IoU, recall/FNR, precision (with 95% CIs), FPS, and parameter counts.

```bash
# 1. Evaluate U-Net
python segmentation_module/evaluate_segmentation.py \
    --checkpoints results/checkpoints/segmentation/unet_run_best.pt \
    --model_type unet \
    --out_dir results/evaluation/unet

# 2. Evaluate Attention U-Net
python segmentation_module/evaluate_segmentation.py \
    --checkpoints results/checkpoints/segmentation/attunet_run_best.pt \
    --model_type attention_unet \
    --out_dir results/evaluation/attunet
```

*(Optional) Evaluate the second seeds if you trained them:*
```bash
python segmentation_module/evaluate_segmentation.py \
    --checkpoints results/checkpoints/segmentation/unet_run_seed2_best.pt \
    --model_type unet \
    --out_dir results/evaluation/unet_seed2

python segmentation_module/evaluate_segmentation.py \
    --checkpoints results/checkpoints/segmentation/attunet_run_seed2_best.pt \
    --model_type attention_unet \
    --out_dir results/evaluation/attunet_seed2
```

## Step 3 — Generate the Published Baselines Comparison Table

We've added a new script `evaluation/published_baselines_comparison.py` to aggregate these reports and compare them directly against the published results from Zhao et al. 2022 (Table 2).

Run the script, passing the paths to all the generated JSON reports:

```bash
python evaluation/published_baselines_comparison.py \
    --laura   results/evaluation/segmentation/evaluation_report.json \
    --unet    results/evaluation/unet/evaluation_report.json \
    --attunet results/evaluation/attunet/evaluation_report.json
```

*(If you ran the second seeds, include them via `--unet2` and `--attunet2` etc.)*

### What to check:
1. The script will print a well-formatted table to the console.
2. It will save `results/segmentation_comparison_table.csv`.
3. Verify that the table includes Dice, IoU, Precision, Recall, Params, and FPS for the models you ran, alongside the published Zhao et al. values.
4. Note that all models use exactly the same patient-level splits and preprocessing.

"""
evaluate_segmentation.py

Standalone evaluation script for the Ovarian Tumor Segmentation module.
Runs Conformal Risk Control and Selective Segmentation analysis on the test set.

Usage:
    # Single model (uses MC-Dropout for uncertainty):
    python segmentation_module/evaluate_segmentation.py --checkpoints results/checkpoints/segmentation/laura_small_best.pt

    # Deep Ensemble (uses the discrepancy between multiple models for uncertainty):
    python segmentation_module/evaluate_segmentation.py --checkpoints results/checkpoints/segmentation/laura_small_best.pt results/checkpoints/segmentation/laura_base_model2_best.pt

The script will:
1. Automatically reconstruct the correct model architectures from the checkpoint configs.
2. Use the 'val' split to calibrate the Conformal Risk threshold (guaranteeing FNR <= alpha).
3. Evaluate the calibrated model on the 'test' split.
4. Generate the Risk-Coverage (AURC) curve data.
5. Save a summary report and CSVs to results/evaluation/segmentation/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from segmentation.models.lightweight_auravit import LightweightAuraViT
from segmentation.dataset import MMOTUSegmentationDataset
from segmentation_uncertainty.mc_dropout_segmentation import MCDropoutSegmentationEstimator
from segmentation_uncertainty.ensemble_segmentation import DeepEnsembleSegmentationEstimator
from segmentation_uncertainty.conformal_risk_control import SegmentationConformalRiskController
from segmentation_uncertainty.selective_segmentation import selective_segmentation_risk_coverage
from segmentation.metrics import dice_score


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Uncertainty & Safety for Segmentation Models")
    p.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="Path(s) to model checkpoint(s). If multiple, a Deep Ensemble is used.",
    )
    p.add_argument(
        "--splits", default="results/splits.csv",
        help="Path to splits CSV.",
    )
    p.add_argument(
        "--out_dir", default="results/evaluation/segmentation",
        help="Output directory for reports and plots.",
    )
    p.add_argument(
        "--alpha", type=float, default=0.10,
        help="Conformal Risk Control target FNR bound.",
    )
    p.add_argument(
        "--mc_samples", type=int, default=10,
        help="Number of MC-Dropout forward passes (only used if passing 1 checkpoint).",
    )
    return p.parse_args()


def load_models(checkpoint_paths: List[str], device: torch.device) -> List[torch.nn.Module]:
    """Loads one or more models, reconstructing their architecture from the saved config."""
    models = []
    for path in checkpoint_paths:
        ckpt_path = Path(path)
        if not ckpt_path.is_absolute():
            ckpt_path = _PROJECT_ROOT / ckpt_path
            
        print(f"Loading checkpoint: {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location=device)
        
        # Reconstruct config
        config = ckpt["config"]
        model = LightweightAuraViT(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        models.append(model)
        print(f"  -> Reconstructed as {config.__class__.__name__} (params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M)")
        
    return models


def collect_dataset_tensors(df: pd.DataFrame, device: torch.device):
    """Loads an entire split into memory as [N, 1, 256, 256] tensors.
    Fine for MMOTU validation/test splits which are ~200-300 images (fits easily in RAM/VRAM)."""
    dataset = MMOTUSegmentationDataset(df, image_size=256, apply_inpainting=True, augment=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    all_images = []
    all_masks = []
    
    for images, masks in tqdm(loader, desc="Loading data"):
        all_images.append(images)
        all_masks.append(masks)
        
    return torch.cat(all_images, dim=0).to(device), torch.cat(all_masks, dim=0).to(device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(" Segmentation Evaluation & Uncertainty Analysis ")
    print("="*60)
    
    # 1. Load Data
    splits_df = pd.read_csv(_PROJECT_ROOT / args.splits)
    val_df = splits_df[splits_df["split"] == "val"].reset_index(drop=True)
    test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)
    
    print(f"\nLoading validation set ({len(val_df)} images) for Conformal Calibration...")
    val_images, val_masks = collect_dataset_tensors(val_df, device)
    
    print(f"Loading test set ({len(test_df)} images) for Final Evaluation...")
    test_images, test_masks = collect_dataset_tensors(test_df, device)

    # 2. Load Estimator
    print("\nInitializing Uncertainty Estimator...")
    models = load_models(args.checkpoints, device)
    
    if len(models) == 1:
        print(f"Single model detected. Using MCDropoutSegmentationEstimator (samples={args.mc_samples}).")
        estimator = MCDropoutSegmentationEstimator(models[0], num_samples=args.mc_samples)
    else:
        print(f"Multiple models ({len(models)}) detected. Using DeepEnsembleSegmentationEstimator.")
        # Support heterogeneous ensembles (e.g. LAURA_BASE + LAURA_SMALL)
        estimator = DeepEnsembleSegmentationEstimator(models)

    # 3. Conformal Risk Control
    print(f"\nCalibrating Conformal Risk Control (Target FNR <= {args.alpha*100}%)...")
    crc = SegmentationConformalRiskController(estimator)
    
    calib_info = crc.calibrate(val_images, val_masks, target_alpha=args.alpha)
    calib_lambda = calib_info["lambda_hat"]
    print(f"  -> Calibrated Threshold (λ_hat): {calib_lambda:.4f}")
    
    # 4. Standard vs Conformal Evaluation on Test Set
    print("\nEvaluating on Test Set...")
    standard_dices = []
    conformal_dices = []
    conformal_fnrs = []
    
    # Process test set in chunks to avoid OOM if test set is large
    chunk_size = 16
    for i in tqdm(range(0, len(test_images), chunk_size), desc="Test Inference"):
        img_chunk = test_images[i:i+chunk_size]
        mask_chunk = test_masks[i:i+chunk_size].cpu().numpy()
        
        # Conformal predictions (calibrated)
        conf_preds = crc.predict(img_chunk).cpu().numpy()
        
        # Standard predictions (0.5 threshold on mean probs)
        raw_est = estimator.predict(img_chunk)
        std_preds = (raw_est["mean_probs"] >= 0.5).cpu().numpy()
        
        for j in range(len(img_chunk)):
            gt = mask_chunk[j, 0] >= 0.5
            
            # Dice
            standard_dices.append(dice_score(std_preds[j, 0], gt))
            conformal_dices.append(dice_score(conf_preds[j, 0], gt))
            
            # FNR (False Negatives / Actual Positives)
            actual_pos = gt.sum()
            if actual_pos > 0:
                false_neg = (gt & ~conf_preds[j, 0]).sum()
                conformal_fnrs.append(false_neg / actual_pos)

    mean_std_dice = np.mean(standard_dices)
    mean_conf_dice = np.mean(conformal_dices)
    mean_conf_fnr = np.mean(conformal_fnrs)
    
    print(f"\nTest Set Results:")
    print(f"  Standard Dice (thresh=0.5): {mean_std_dice:.4f}")
    print(f"  Conformal Dice (thresh={calib_lambda:.4f}): {mean_conf_dice:.4f}")
    print(f"  Conformal FNR: {mean_conf_fnr:.4f} (Target: <= {args.alpha})")

    # 5. Selective Segmentation (AURC)
    print("\nCalculating Risk-Coverage (AURC)...")
    aurc_results = selective_segmentation_risk_coverage(estimator, test_images, test_masks)
    
    aurc_value = aurc_results["aurc"]
    coverages = aurc_results["coverages"]
    risks = aurc_results["risks"]
    print(f"  -> AURC: {aurc_value:.4f}")

    # Save AURC data
    aurc_df = pd.DataFrame({"coverage": coverages, "risk_1_minus_dice": risks})
    aurc_df.to_csv(out_dir / "selective_risk_coverage.csv", index=False)
    
    # Save Report
    report = {
        "models_used": [Path(p).name for p in args.checkpoints],
        "estimator": "DeepEnsemble" if len(models) > 1 else "MCDropout",
        "test_images": len(test_images),
        "conformal_target_alpha_fnr": args.alpha,
        "conformal_calibrated_lambda": calib_lambda,
        "test_mean_standard_dice": float(mean_std_dice),
        "test_mean_conformal_dice": float(mean_conf_dice),
        "test_mean_conformal_fnr": float(mean_conf_fnr),
        "test_aurc": float(aurc_value)
    }
    
    with open(out_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # Generate AURC Plot (BLUE_DARK #003366 theme as per convention)
    plt.figure(figsize=(8, 6))
    plt.plot(coverages, risks, color="#003366", linewidth=2.5, label=f"AURC = {aurc_value:.4f}")
    plt.fill_between(coverages, risks, alpha=0.2, color="#6699FF")
    plt.xlabel("Coverage (Fraction of Test Set Accepted)")
    plt.ylabel("Risk (1 - Mean Dice)")
    plt.title("Selective Segmentation: Risk-Coverage Curve")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "risk_coverage_curve.png", dpi=300)
    plt.close()
    
    print(f"\nSuccess! All artifacts saved to {out_dir}")

if __name__ == "__main__":
    main()

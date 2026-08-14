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
import time
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
from segmentation_uncertainty.selective_segmentation import selective_segmentation_risk_coverage, image_level_uncertainty
from segmentation.metrics import dice_score, iou_score
from segmentation.models.baselines import count_parameters


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


def bootstrap_metric_ci(
    values: np.ndarray,
    n_boot: int = 5000,
    ci: float = 0.95,
) -> tuple:
    """Bootstrap confidence interval for the mean of `values`."""
    rng = np.random.default_rng(seed=42)
    values = np.asarray(values, dtype=float)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - (1 - ci) / 2)))
    return lo, hi


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
        estimator = DeepEnsembleSegmentationEstimator(models, device)

    # 3. Conformal Risk Control
    print(f"\nCalibrating Conformal Risk Control (Target FNR <= {args.alpha*100}%)...")
    crc = SegmentationConformalRiskController(alpha=args.alpha)

    print("  -> Generating probability maps for calibration set...")
    val_est = []
    chunk_size = 16
    for i in tqdm(range(0, len(val_images), chunk_size), desc="Val Inference"):
        chunk = val_images[i:i+chunk_size]
        res = estimator.predict(chunk)
        val_est.append(res["mean_probs"])
    val_prob_maps = np.concatenate(val_est, axis=0)

    crc.calibrate(val_prob_maps, val_masks.cpu().numpy())
    crc_summary = crc.summary()
    calib_lambda = crc.lambda_star

    print(f"  -> CRC Calibrated Threshold (λ*): {calib_lambda:.4f}")
    print(f"  -> Standard Threshold:            {crc.lambda_star_standard:.4f}")
    print(f"  -> alpha_corrected (finite-sample): {crc_summary['alpha_corrected']:.6f}")
    print(f"  -> Val FNR @ λ*=0.50 (standard):  {crc_summary['calibration_fnr_at_05_threshold']:.4f}")
    print(f"  -> Val FNR @ λ*={calib_lambda:.2f} (CRC):      {crc_summary['calibration_fnr_at_crc_threshold']:.4f}")
    print(f"  -> Formal CRC guarantee valid:    {crc_summary['formal_guarantee_valid']}")
    print(f"  -> Note: {crc_summary['note']}")
    
    # 4. Standard vs Conformal Evaluation on Test Set
    print("\nEvaluating on Test Set...")
    # Per-image metric arrays
    standard_dices  = []
    conformal_dices = []
    standard_ious   = []
    conformal_ious  = []
    standard_fnrs   = []
    conformal_fnrs  = []
    standard_precs  = []
    conformal_precs = []

    test_prob_maps_list   = []
    test_uncertainty_list = []

    # --- Inference timing ---
    timing_start = time.time()
    n_test_images = len(test_images)

    chunk_size = 16
    for i in tqdm(range(0, len(test_images), chunk_size), desc="Test Inference"):
        img_chunk  = test_images[i:i+chunk_size]
        mask_chunk = test_masks[i:i+chunk_size].cpu().numpy()

        raw_est   = estimator.predict(img_chunk)
        prob_maps = raw_est["mean_probs"]

        test_prob_maps_list.append(prob_maps)
        test_uncertainty_list.append(image_level_uncertainty(raw_est))

        conf_preds = crc.predict(prob_maps)           # CRC threshold
        std_preds  = (prob_maps >= 0.5).astype(np.uint8)  # standard 0.5

        for j in range(len(img_chunk)):
            gt = mask_chunk[j, 0] >= 0.5

            # Dice and IoU
            standard_dices.append(dice_score(std_preds[j, 0],   gt))
            conformal_dices.append(dice_score(conf_preds[j, 0], gt))
            standard_ious.append(iou_score(std_preds[j, 0],   gt))
            conformal_ious.append(iou_score(conf_preds[j, 0], gt))

            actual_pos = int(gt.sum())
            if actual_pos > 0:
                # FNR = missed positives / actual positives
                std_fn  = int((gt & ~std_preds[j, 0].astype(bool)).sum())
                conf_fn = int((gt & ~conf_preds[j, 0].astype(bool)).sum())
                standard_fnrs.append(std_fn  / actual_pos)
                conformal_fnrs.append(conf_fn / actual_pos)

                # Precision = true positives / predicted positives
                std_pred_pos  = int(std_preds[j, 0].sum())
                conf_pred_pos = int(conf_preds[j, 0].sum())
                std_tp  = int((std_preds[j, 0].astype(bool)  & gt).sum())
                conf_tp = int((conf_preds[j, 0].astype(bool) & gt).sum())
                standard_precs.append(std_tp  / max(std_pred_pos,  1))
                conformal_precs.append(conf_tp / max(conf_pred_pos, 1))

    inference_time_s = time.time() - timing_start
    fps = n_test_images / inference_time_s

    # --- Aggregated results with bootstrap CIs ---
    standard_dices  = np.array(standard_dices)
    conformal_dices = np.array(conformal_dices)
    standard_ious   = np.array(standard_ious)
    conformal_ious  = np.array(conformal_ious)
    standard_fnrs   = np.array(standard_fnrs)
    conformal_fnrs  = np.array(conformal_fnrs)
    standard_precs  = np.array(standard_precs)
    conformal_precs = np.array(conformal_precs)

    mean_std_dice  = float(standard_dices.mean())
    mean_conf_dice = float(conformal_dices.mean())
    mean_conf_fnr  = float(conformal_fnrs.mean()) if len(conformal_fnrs) > 0 else float('nan')
    mean_std_fnr   = float(standard_fnrs.mean())  if len(standard_fnrs)  > 0 else float('nan')

    ci_std_dice  = bootstrap_metric_ci(standard_dices)
    ci_conf_dice = bootstrap_metric_ci(conformal_dices)
    ci_std_iou   = bootstrap_metric_ci(standard_ious)
    ci_conf_iou  = bootstrap_metric_ci(conformal_ious)
    ci_std_fnr   = bootstrap_metric_ci(standard_fnrs)  if len(standard_fnrs)  > 0 else (float('nan'), float('nan'))
    ci_conf_fnr  = bootstrap_metric_ci(conformal_fnrs) if len(conformal_fnrs) > 0 else (float('nan'), float('nan'))

    print(f"\nTest Set Results:")
    print(f"  [Standard threshold = 0.5]")
    print(f"    Dice    : {mean_std_dice:.4f}  95% CI [{ci_std_dice[0]:.4f}, {ci_std_dice[1]:.4f}]")
    print(f"    IoU     : {standard_ious.mean():.4f}  95% CI [{ci_std_iou[0]:.4f}, {ci_std_iou[1]:.4f}]")
    print(f"    FNR     : {mean_std_fnr:.4f}  95% CI [{ci_std_fnr[0]:.4f}, {ci_std_fnr[1]:.4f}]")
    print(f"  [CRC threshold = {calib_lambda:.4f}  (target FNR <= {args.alpha})]")
    print(f"    Dice    : {mean_conf_dice:.4f}  95% CI [{ci_conf_dice[0]:.4f}, {ci_conf_dice[1]:.4f}]")
    print(f"    IoU     : {conformal_ious.mean():.4f}  95% CI [{ci_conf_iou[0]:.4f}, {ci_conf_iou[1]:.4f}]")
    print(f"    FNR     : {mean_conf_fnr:.4f}  95% CI [{ci_conf_fnr[0]:.4f}, {ci_conf_fnr[1]:.4f}]")
    print(f"  Inference FPS: {fps:.1f}  ({inference_time_s:.1f}s for {n_test_images} images)")
    print(f"  Formal CRC guarantee valid (n>={30}): {crc_summary['formal_guarantee_valid']}")

    # 5. Selective Segmentation (AURC)
    print("\nCalculating Risk-Coverage (AURC)...")
    all_test_prob_maps = np.concatenate(test_prob_maps_list, axis=0)
    all_test_uncertainty = np.concatenate(test_uncertainty_list, axis=0)
    all_test_masks = test_masks.cpu().numpy()
    
    aurc_results = selective_segmentation_risk_coverage(
        prob_maps=all_test_prob_maps,
        gt_masks=all_test_masks,
        uncertainty_scores=all_test_uncertainty
    )
    
    aurc_value = aurc_results["aurc"]
    coverages = aurc_results["coverages"]
    risks = aurc_results["risks"]
    print(f"  -> AURC: {aurc_value:.4f}")

    # Save AURC data
    aurc_df = pd.DataFrame({"coverage": coverages, "risk_1_minus_dice": risks})
    aurc_df.to_csv(out_dir / "selective_risk_coverage.csv", index=False)
    
    # Model parameter count
    param_info = {}
    for i, m in enumerate(models):
        total    = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        param_info[f"model_{i}"] = {
            "total_params_M": round(total / 1e6, 3),
            "trainable_params_M": round(trainable / 1e6, 3),
        }

    # Save Report
    report = {
        "models_used": [Path(p).name for p in args.checkpoints],
        "estimator": "DeepEnsemble" if len(models) > 1 else "MCDropout",
        "test_images": len(test_images),
        "model_params": param_info,
        "inference_fps": round(fps, 2),
        "conformal_target_alpha_fnr": args.alpha,
        "conformal_alpha_corrected": crc_summary["alpha_corrected"],
        "conformal_calibrated_lambda": calib_lambda,
        "formal_guarantee_valid": crc_summary["formal_guarantee_valid"],
        "crc_note": crc_summary["note"],
        # Standard threshold (0.5) metrics
        "val_fnr_at_05": crc_summary["calibration_fnr_at_05_threshold"],
        "test_mean_standard_dice": float(mean_std_dice),
        "test_mean_standard_dice_ci": list(ci_std_dice),
        "test_mean_standard_iou": float(standard_ious.mean()),
        "test_mean_standard_iou_ci": list(ci_std_iou),
        "test_mean_standard_fnr": float(mean_std_fnr),
        "test_mean_standard_fnr_ci": list(ci_std_fnr),
        # CRC threshold metrics
        "val_fnr_at_crc": crc_summary["calibration_fnr_at_crc_threshold"],
        "test_mean_conformal_dice": float(mean_conf_dice),
        "test_mean_conformal_dice_ci": list(ci_conf_dice),
        "test_mean_conformal_iou": float(conformal_ious.mean()),
        "test_mean_conformal_iou_ci": list(ci_conf_iou),
        "test_mean_conformal_fnr": float(mean_conf_fnr),
        "test_mean_conformal_fnr_ci": list(ci_conf_fnr),
        "test_aurc": float(aurc_value),
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

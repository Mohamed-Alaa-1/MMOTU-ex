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
from segmentation.metrics import (
    dice_score, iou_score,
    boundary_f1_score, hausdorff_distance_95,
)
from segmentation.models.baselines import count_parameters


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Uncertainty & Safety for Segmentation Models")
    p.add_argument(
        "--checkpoints", nargs="+", default=["results/checkpoints/segmentation/laura_small_best.pt"],
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
    p.add_argument(
        "--model_type", default="laura",
        choices=["laura", "unet", "attention_unet", "deeplabv3plus", "unetplusplus"],
        help="Architecture type of the checkpoint(s). Required when loading non-LAURA baselines.",
    )
    return p.parse_args()


def load_models(checkpoint_paths: List[str], device: torch.device,
                model_type: str = "laura") -> List[torch.nn.Module]:
    """Load one or more models from checkpoint files.

    Args:
        checkpoint_paths: List of .pt checkpoint file paths.
        device:           Target device (CPU or CUDA).
        model_type:       'laura' (default) reconstructs from ckpt['config'].
                          Any other value uses get_baseline_model(model_type).
                          This is needed because baseline checkpoints may not
                          store a 'config' key.
    """
    from segmentation.models.baselines import get_baseline_model
    models = []
    for path in checkpoint_paths:
        ckpt_path = Path(path)
        if not ckpt_path.is_absolute():
            ckpt_path = _PROJECT_ROOT / ckpt_path

        print(f"Loading checkpoint: {ckpt_path.name}  (model_type={model_type})")
        ckpt = torch.load(ckpt_path, map_location=device)

        if model_type == "laura":
            # LAURA: architecture is reconstructed from the saved config
            config = ckpt["config"]
            model = LightweightAuraViT(config).to(device)
            arch_label = config.__class__.__name__
        else:
            # Baseline: instantiate by name; state dict keys must match
            model = get_baseline_model(model_type, in_channels=1, num_classes=1).to(device)
            arch_label = model_type.upper()

        state_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
        missing, unexpected = model.load_state_dict(ckpt[state_key], strict=False)
        if missing:
            print(f"  [WARNING] Missing keys in checkpoint ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"  [WARNING] Unexpected keys in checkpoint ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

        model.eval()
        models.append(model)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  -> {arch_label}  ({n_params/1e6:.2f}M params)")

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
    calib_df = splits_df[splits_df["split"] == "calib"].reset_index(drop=True)
    test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)
    
    print(f"\nLoading calibration set ({len(calib_df)} images) for Conformal Calibration...")
    calib_images, calib_masks = collect_dataset_tensors(calib_df, device)
    
    print(f"Loading test set ({len(test_df)} images) for Final Evaluation...")
    test_images, test_masks = collect_dataset_tensors(test_df, device)

    # 2. Load Estimator
    print("\nInitializing Uncertainty Estimator...")
    models = load_models(args.checkpoints, device, model_type=args.model_type)
    
    if len(models) == 1:
        print(f"Single model detected. Using MCDropoutSegmentationEstimator (samples={args.mc_samples}).")
        estimator = MCDropoutSegmentationEstimator(models[0], device=device, n_samples=args.mc_samples)
    else:
        print(f"Multiple models ({len(models)}) detected. Using DeepEnsembleSegmentationEstimator.")
        # Support heterogeneous ensembles (e.g. LAURA_BASE + LAURA_SMALL)
        estimator = DeepEnsembleSegmentationEstimator(models, device)

    # 3. Conformal Risk Control
    print(f"\nCalibrating Conformal Risk Control (Target FNR <= {args.alpha*100}%)...")
    crc = SegmentationConformalRiskController(alpha=args.alpha)

    print("  -> Generating probability maps for calibration set...")
    calib_est = []
    chunk_size = 16
    for i in tqdm(range(0, len(calib_images), chunk_size), desc="Calib Inference"):
        chunk = calib_images[i:i+chunk_size]
        res = estimator.predict(chunk)
        calib_est.append(res["mean_probs"])
    calib_prob_maps = np.concatenate(calib_est, axis=0)

    crc.calibrate(calib_prob_maps, calib_masks.cpu().numpy())
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
    standard_dices   = []
    conformal_dices  = []
    standard_ious    = []
    conformal_ious   = []
    standard_fnrs    = []
    conformal_fnrs   = []
    standard_precs   = []
    conformal_precs  = []
    standard_recalls  = []
    conformal_recalls = []
    standard_bfs     = []
    conformal_bfs    = []
    standard_hd95s   = []
    conformal_hd95s  = []

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

        conf_preds = crc.predict(prob_maps)            # CRC threshold
        std_preds  = (prob_maps >= 0.5).astype(np.uint8)  # standard 0.5

        for j in range(len(img_chunk)):
            gt       = mask_chunk[j, 0] >= 0.5
            std_pred = std_preds[j, 0].astype(bool)
            conf_pred= conf_preds[j, 0].astype(bool)

            # Dice and IoU
            standard_dices.append(dice_score(std_pred,   gt))
            conformal_dices.append(dice_score(conf_pred, gt))
            standard_ious.append(iou_score(std_pred,   gt))
            conformal_ious.append(iou_score(conf_pred, gt))

            # Boundary F1 (BF-Score) and HD95
            standard_bfs.append(boundary_f1_score(std_pred,   gt))
            conformal_bfs.append(boundary_f1_score(conf_pred, gt))
            standard_hd95s.append(hausdorff_distance_95(std_pred,   gt))
            conformal_hd95s.append(hausdorff_distance_95(conf_pred, gt))

            actual_pos = int(gt.sum())
            if actual_pos > 0:
                # FNR = missed positives / actual positives
                std_fn  = int((gt & ~std_pred).sum())
                conf_fn = int((gt & ~conf_pred).sum())
                standard_fnrs.append(std_fn  / actual_pos)
                conformal_fnrs.append(conf_fn / actual_pos)

                # Recall = true positives / actual positives  (= 1 - FNR)
                std_tp  = int((std_pred  & gt).sum())
                conf_tp = int((conf_pred & gt).sum())
                standard_recalls.append(std_tp  / actual_pos)
                conformal_recalls.append(conf_tp / actual_pos)

                # Precision = true positives / predicted positives
                std_pred_pos  = int(std_pred.sum())
                conf_pred_pos = int(conf_pred.sum())
                standard_precs.append(std_tp  / max(std_pred_pos,  1))
                conformal_precs.append(conf_tp / max(conf_pred_pos, 1))

    inference_time_s = time.time() - timing_start
    fps = n_test_images / inference_time_s

    # --- Aggregated results with bootstrap CIs ---
    standard_dices    = np.array(standard_dices)
    conformal_dices   = np.array(conformal_dices)
    standard_ious     = np.array(standard_ious)
    conformal_ious    = np.array(conformal_ious)
    standard_fnrs     = np.array(standard_fnrs)
    conformal_fnrs    = np.array(conformal_fnrs)
    standard_precs    = np.array(standard_precs)
    conformal_precs   = np.array(conformal_precs)
    standard_recalls  = np.array(standard_recalls)
    conformal_recalls = np.array(conformal_recalls)
    standard_bfs      = np.array(standard_bfs)
    conformal_bfs     = np.array(conformal_bfs)
    # Filter out NaN HD95 values (empty masks) before aggregating
    standard_hd95s_clean  = np.array([v for v in standard_hd95s  if not np.isnan(v)])
    conformal_hd95s_clean = np.array([v for v in conformal_hd95s if not np.isnan(v)])

    def _safe_mean(arr): return float(arr.mean()) if len(arr) > 0 else float('nan')
    def _safe_ci(arr):   return bootstrap_metric_ci(arr) if len(arr) > 0 else (float('nan'), float('nan'))

    mean_std_dice     = _safe_mean(standard_dices)
    mean_conf_dice    = _safe_mean(conformal_dices)
    mean_std_fnr      = _safe_mean(standard_fnrs)
    mean_conf_fnr     = _safe_mean(conformal_fnrs)
    mean_std_prec     = _safe_mean(standard_precs)
    mean_conf_prec    = _safe_mean(conformal_precs)
    mean_std_recall   = _safe_mean(standard_recalls)
    mean_conf_recall  = _safe_mean(conformal_recalls)
    mean_std_bf       = _safe_mean(standard_bfs)
    mean_conf_bf      = _safe_mean(conformal_bfs)
    mean_std_hd95     = _safe_mean(standard_hd95s_clean)
    mean_conf_hd95    = _safe_mean(conformal_hd95s_clean)

    ci_std_dice    = _safe_ci(standard_dices)
    ci_conf_dice   = _safe_ci(conformal_dices)
    ci_std_iou     = _safe_ci(standard_ious)
    ci_conf_iou    = _safe_ci(conformal_ious)
    ci_std_fnr     = _safe_ci(standard_fnrs)
    ci_conf_fnr    = _safe_ci(conformal_fnrs)
    ci_std_prec    = _safe_ci(standard_precs)
    ci_conf_prec   = _safe_ci(conformal_precs)
    ci_std_recall  = _safe_ci(standard_recalls)
    ci_conf_recall = _safe_ci(conformal_recalls)
    ci_std_bf      = _safe_ci(standard_bfs)
    ci_conf_bf     = _safe_ci(conformal_bfs)
    ci_std_hd95    = _safe_ci(standard_hd95s_clean)
    ci_conf_hd95   = _safe_ci(conformal_hd95s_clean)

    print(f"\nTest Set Results:")
    print(f"  [Standard threshold = 0.5]")
    print(f"    Dice      : {mean_std_dice:.4f}  95% CI [{ci_std_dice[0]:.4f}, {ci_std_dice[1]:.4f}]")
    print(f"    IoU       : {_safe_mean(standard_ious):.4f}  95% CI [{ci_std_iou[0]:.4f}, {ci_std_iou[1]:.4f}]")
    print(f"    Precision : {mean_std_prec:.4f}  95% CI [{ci_std_prec[0]:.4f}, {ci_std_prec[1]:.4f}]")
    print(f"    Recall    : {mean_std_recall:.4f}  95% CI [{ci_std_recall[0]:.4f}, {ci_std_recall[1]:.4f}]")
    print(f"    FNR       : {mean_std_fnr:.4f}  95% CI [{ci_std_fnr[0]:.4f}, {ci_std_fnr[1]:.4f}]")
    print(f"    BF-Score  : {mean_std_bf:.4f}  95% CI [{ci_std_bf[0]:.4f}, {ci_std_bf[1]:.4f}]")
    print(f"    HD95      : {mean_std_hd95:.2f}px  95% CI [{ci_std_hd95[0]:.2f}, {ci_std_hd95[1]:.2f}]")
    print(f"  [CRC threshold = {calib_lambda:.4f}  (target FNR <= {args.alpha})]")
    print(f"    Dice      : {mean_conf_dice:.4f}  95% CI [{ci_conf_dice[0]:.4f}, {ci_conf_dice[1]:.4f}]")
    print(f"    IoU       : {_safe_mean(conformal_ious):.4f}  95% CI [{ci_conf_iou[0]:.4f}, {ci_conf_iou[1]:.4f}]")
    print(f"    Precision : {mean_conf_prec:.4f}  95% CI [{ci_conf_prec[0]:.4f}, {ci_conf_prec[1]:.4f}]")
    print(f"    Recall    : {mean_conf_recall:.4f}  95% CI [{ci_conf_recall[0]:.4f}, {ci_conf_recall[1]:.4f}]")
    print(f"    FNR       : {mean_conf_fnr:.4f}  95% CI [{ci_conf_fnr[0]:.4f}, {ci_conf_fnr[1]:.4f}]")
    print(f"    BF-Score  : {mean_conf_bf:.4f}  95% CI [{ci_conf_bf[0]:.4f}, {ci_conf_bf[1]:.4f}]")
    print(f"    HD95      : {mean_conf_hd95:.2f}px  95% CI [{ci_conf_hd95[0]:.2f}, {ci_conf_hd95[1]:.2f}]")
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
        # --- Standard threshold (0.5) metrics ---
        "val_fnr_at_05": crc_summary["calibration_fnr_at_05_threshold"],
        "test_mean_standard_dice":          float(mean_std_dice),
        "test_mean_standard_dice_ci":       list(ci_std_dice),
        "test_mean_standard_iou":           float(_safe_mean(standard_ious)),
        "test_mean_standard_iou_ci":        list(ci_std_iou),
        "test_mean_standard_precision":     float(mean_std_prec),
        "test_mean_standard_precision_ci":  list(ci_std_prec),
        "test_mean_standard_recall":        float(mean_std_recall),
        "test_mean_standard_recall_ci":     list(ci_std_recall),
        "test_mean_standard_fnr":           float(mean_std_fnr),
        "test_mean_standard_fnr_ci":        list(ci_std_fnr),
        "test_mean_standard_bf_score":      float(mean_std_bf),
        "test_mean_standard_bf_score_ci":   list(ci_std_bf),
        "test_mean_standard_hd95_px":       float(mean_std_hd95),
        "test_mean_standard_hd95_px_ci":    list(ci_std_hd95),
        # --- CRC threshold metrics ---
        "val_fnr_at_crc": crc_summary["calibration_fnr_at_crc_threshold"],
        "test_mean_conformal_dice":         float(mean_conf_dice),
        "test_mean_conformal_dice_ci":      list(ci_conf_dice),
        "test_mean_conformal_iou":          float(_safe_mean(conformal_ious)),
        "test_mean_conformal_iou_ci":       list(ci_conf_iou),
        "test_mean_conformal_precision":    float(mean_conf_prec),
        "test_mean_conformal_precision_ci": list(ci_conf_prec),
        "test_mean_conformal_recall":       float(mean_conf_recall),
        "test_mean_conformal_recall_ci":    list(ci_conf_recall),
        "test_mean_conformal_fnr":          float(mean_conf_fnr),
        "test_mean_conformal_fnr_ci":       list(ci_conf_fnr),
        "test_mean_conformal_bf_score":     float(mean_conf_bf),
        "test_mean_conformal_bf_score_ci":  list(ci_conf_bf),
        "test_mean_conformal_hd95_px":      float(mean_conf_hd95),
        "test_mean_conformal_hd95_px_ci":   list(ci_conf_hd95),
        # --- Uncertainty / Selective ---
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

"""
generate_statistical_tests.py

Calculates the statistical significance tests requested in the review:
  1. Caliper Ablation: Bootstrap CIs + Wilcoxon signed-rank test (per-image Dice).
  2. Classification Ensemble: McNemar's test (Ensemble vs best single model).

Results are saved to 'results/statistical_results.txt'.
"""
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon, chi2
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")

# 1. Ablation imports
from segmentation_module.segmentation.models.lightweight_auravit import LightweightAuraViT
from segmentation_module.segmentation.dataset import MMOTUSegmentationDataset
from segmentation_module.ablation_inpainting import _load_model as load_seg_model, _evaluate as eval_seg

# 2. Classification imports
from evaluation.ensemble import ModelEnsemble
from models.factory import get_model
from utils.checkpoint import load_checkpoint
from data.dataset import MMOTUDataset
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------

def _bootstrap_ci(values: np.ndarray, n_boot: int = 5000, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval for the mean of `values`."""
    rng = np.random.default_rng(seed=42)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - (1 - ci) / 2)))
    return lo, hi


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_content = []

    # -----------------------------------------------------------------------
    # 1. Caliper Ablation Statistical Test
    # -----------------------------------------------------------------------
    print("\n--- 1. Running Caliper Ablation Statistical Test ---")
    model_a_path = "results/checkpoints/segmentation/laura_small_best.pt"
    model_b_path = "results/checkpoints/segmentation/laura_small_shortcut_best.pt"

    try:
        model_a = load_seg_model(model_a_path, device)
        model_b = load_seg_model(model_b_path, device)

        splits_df = pd.read_csv("results/splits.csv")
        test_df   = splits_df[splits_df["split"] == "test"].reset_index(drop=True)

        ds_clean = MMOTUSegmentationDataset(test_df, image_size=256, apply_inpainting=True,  augment=False)
        ds_raw   = MMOTUSegmentationDataset(test_df, image_size=256, apply_inpainting=False, augment=False)

        print("Evaluating Model A on Clean…")
        dice_a_clean, _ = eval_seg(model_a, ds_clean, device, "A × Clean")
        print("Evaluating Model A on Raw…")
        dice_a_raw,   _ = eval_seg(model_a, ds_raw,   device, "A × Raw")
        print("Evaluating Model B on Clean…")
        dice_b_clean, _ = eval_seg(model_b, ds_clean, device, "B × Clean")
        print("Evaluating Model B on Raw…")
        dice_b_raw,   _ = eval_seg(model_b, ds_raw,   device, "B × Raw")

        # --- Bootstrap CIs for each cell ---
        ci_a_clean = _bootstrap_ci(dice_a_clean)
        ci_a_raw   = _bootstrap_ci(dice_a_raw)
        ci_b_clean = _bootstrap_ci(dice_b_clean)
        ci_b_raw   = _bootstrap_ci(dice_b_raw)

        # --- Wilcoxon signed-rank tests ---
        # Test 1: Inpainting Gain (A_clean vs B_clean)
        try:
            stat_gain, p_gain = wilcoxon(dice_a_clean, dice_b_clean, alternative='two-sided')
        except ValueError:
            stat_gain, p_gain = 0.0, 1.0  # all differences == 0

        # Test 2: Shortcut Inflation (B_raw vs B_clean)
        try:
            stat_inf, p_inf = wilcoxon(dice_b_raw, dice_b_clean, alternative='two-sided')
        except ValueError:
            stat_inf, p_inf = 0.0, 1.0

        # Cohen's d effect sizes
        def cohens_d(a, b):
            na, nb = len(a), len(b)
            if na < 2 or nb < 2:
                return 0.0
            pooled = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
            return float((np.mean(a)-np.mean(b)) / pooled) if pooled > 0 else 0.0

        d_gain = cohens_d(dice_a_clean, dice_b_clean)
        d_inf  = cohens_d(dice_b_raw,   dice_b_clean)

        output_content.append("=== 1. Caliper Ablation Effect Sizes ===\n")
        output_content.append("2×2 Matrix (Mean Dice  [95% CI]):")
        output_content.append(f"  Model A (Inpainted) | Clean: {dice_a_clean.mean():.4f} [{ci_a_clean[0]:.4f}, {ci_a_clean[1]:.4f}]  | Raw: {dice_a_raw.mean():.4f} [{ci_a_raw[0]:.4f}, {ci_a_raw[1]:.4f}]")
        output_content.append(f"  Model B (Shortcut)  | Clean: {dice_b_clean.mean():.4f} [{ci_b_clean[0]:.4f}, {ci_b_clean[1]:.4f}]  | Raw: {dice_b_raw.mean():.4f} [{ci_b_raw[0]:.4f}, {ci_b_raw[1]:.4f}]")
        output_content.append("")
        output_content.append("Test 1: Inpainting Gain (Model A Clean vs Model B Clean)")
        output_content.append(f"  Mean Difference: {np.mean(dice_a_clean)-np.mean(dice_b_clean):+.4f} Dice")
        output_content.append(f"  Cohen's d:       {d_gain:+.4f}")
        output_content.append(f"  Wilcoxon stat:   {stat_gain:.4f}")
        output_content.append(f"  p-value:         {p_gain:.4e}")
        output_content.append(f"  Significant (p<0.05): {p_gain < 0.05}")
        output_content.append("")
        output_content.append("Test 2: Shortcut Inflation (Model B Raw vs Model B Clean)")
        output_content.append(f"  Mean Difference: {np.mean(dice_b_raw)-np.mean(dice_b_clean):+.4f} Dice")
        output_content.append(f"  Cohen's d:       {d_inf:+.4f}")
        output_content.append(f"  Wilcoxon stat:   {stat_inf:.4f}")
        output_content.append(f"  p-value:         {p_inf:.4e}")
        output_content.append(f"  Significant (p<0.05): {p_inf < 0.05}\n")

    except Exception as e:
        print(f"Error in Ablation Test: {e}")
        output_content.append("=== 1. Caliper Ablation Effect Sizes ===")
        output_content.append(f"Error running test: {e}\n")

    # -----------------------------------------------------------------------
    # 2. Classification Ensemble McNemar Test
    # -----------------------------------------------------------------------
    print("\n--- 2. Running Classification Ensemble McNemar Test ---")
    try:
        config = OmegaConf.load("configs/default.yaml")
        
        from torchvision import transforms
        val_transforms = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        splits_df = pd.read_csv("results/splits.csv")
        test_df   = splits_df[splits_df["split"] == "test"].reset_index(drop=True)
        ds_test   = MMOTUDataset(test_df, transform=val_transforms, return_path=True)
        loader_test = DataLoader(ds_test, batch_size=16, shuffle=False)

        model_names     = ["densenet121", "resnet50", "efficientnet_b3",
                           "mobilenet_v3_large", "swin_t"]
        best_single_name = "resnet50"
        loaded_models    = []
        best_single_model = None

        for name in model_names:
            ckpt_path = f"results/checkpoints/exp_001_{name}_best.pt"
            model, _  = get_model(name, num_classes=config.training.num_classes)
            load_checkpoint(ckpt_path, model)
            model.eval()
            model.to(device)
            loaded_models.append(model)
            if name == best_single_name:
                best_single_model = model

        ensemble = ModelEnsemble(loaded_models, device)

        ens_correct    = []
        single_correct = []

        print("Evaluating Ensemble vs ResNet-50 on Test Set…")
        with torch.no_grad():
            for batch in loader_test:
                imgs   = batch[0].to(device)
                labels = batch[2].to(device)

                # Single model predictions
                single_preds = best_single_model(imgs).argmax(dim=1)
                single_correct.extend((single_preds == labels).cpu().numpy())

                # Ensemble predictions (mean softmax)
                all_probs = [torch.softmax(m(imgs), dim=1) for m in ensemble.models]
                avg_probs = sum(all_probs) / len(all_probs)
                ens_preds = avg_probs.argmax(dim=1)
                ens_correct.extend((ens_preds == labels).cpu().numpy())

        ens_correct    = np.array(ens_correct,    dtype=bool)
        single_correct = np.array(single_correct, dtype=bool)

        ens_acc    = float(ens_correct.mean()) * 100
        single_acc = float(single_correct.mean()) * 100

        # Bootstrap CIs for both
        rng = np.random.default_rng(seed=42)
        def _acc_ci(arr):
            boot = np.array([rng.choice(arr.astype(float), len(arr), replace=True).mean()
                             for _ in range(5000)])
            return float(np.percentile(boot, 2.5))*100, float(np.percentile(boot, 97.5))*100

        ens_ci    = _acc_ci(ens_correct)
        single_ci = _acc_ci(single_correct)

        # McNemar's test (with Yates continuity correction)
        a = int(np.sum( ens_correct &  single_correct))
        b = int(np.sum( ens_correct & ~single_correct))
        c = int(np.sum(~ens_correct &  single_correct))
        d = int(np.sum(~ens_correct & ~single_correct))

        if b + c == 0:
            statistic = 0.0
            pvalue    = 1.0
        else:
            statistic = float((abs(b - c) - 1) ** 2 / (b + c))
            pvalue    = float(chi2.sf(statistic, 1))

        output_content.append("=== 2. Classification Ensemble McNemar Test ===\n")
        output_content.append(f"Ensemble Accuracy:  {ens_acc:.2f}%  95% CI [{ens_ci[0]:.2f}%, {ens_ci[1]:.2f}%]")
        output_content.append(f"ResNet-50 Accuracy: {single_acc:.2f}%  95% CI [{single_ci[0]:.2f}%, {single_ci[1]:.2f}%]")
        output_content.append(f"Difference:         {ens_acc - single_acc:+.2f}%")
        output_content.append("")
        output_content.append("McNemar Contingency Table (both vs each):")
        output_content.append(f"  {'':30s}  ResNet-50 Correct  ResNet-50 Wrong")
        output_content.append(f"  Ensemble Correct    {a:>17d}  {b:>14d}")
        output_content.append(f"  Ensemble Wrong      {c:>17d}  {d:>14d}")
        output_content.append(f"\n  Discordant pairs (b+c): {b+c}")
        output_content.append(f"  Chi-squared statistic:  {statistic:.4f}")
        output_content.append(f"  p-value:                {pvalue:.4e}")
        output_content.append(f"  Significant (p<0.05):   {pvalue < 0.05}\n")

    except Exception as e:
        print(f"Error in Classification Test: {e}")
        output_content.append("=== 2. Classification Ensemble McNemar Test ===")
        output_content.append(f"Error running test: {e}\n")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    Path("results").mkdir(exist_ok=True)
    output_file = "results/statistical_results.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(output_content))
    print(f"\nStatistical results saved to '{output_file}'.")


if __name__ == "__main__":
    main()

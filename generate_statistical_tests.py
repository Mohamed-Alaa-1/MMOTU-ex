"""
generate_statistical_tests.py

This script calculates the statistical significance tests requested in the review:
1. Ablation effect size (Wilcoxon signed-rank test on per-image Dice scores).
2. Ensemble advantage (McNemar's test comparing Ensemble vs best single model).

It appends the outputs to 'list of comments data.md'.
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
from segmentation.models.lightweight_auravit import LightweightAuraViT
from segmentation.dataset import MMOTUSegmentationDataset
from segmentation_module.ablation_inpainting import _load_model as load_seg_model, _evaluate as eval_seg

# 2. Classification imports
from evaluation.ensemble import ModelEnsemble
from models.factory import get_model
from utils.checkpoint import load_checkpoint
from data.dataset import MMOTUDataset
from data.transforms import get_transforms
from omegaconf import OmegaConf

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_content = []

    # --- 1. Ablation Statistical Test ---
    print("\n--- 1. Running Ablation Statistical Test ---")
    model_a_path = "results/checkpoints/segmentation/laura_small_best.pt"
    model_b_path = "results/checkpoints/segmentation/laura_small_shortcut_best.pt"
    
    try:
        model_a = load_seg_model(model_a_path, device)
        model_b = load_seg_model(model_b_path, device)
        
        splits_df = pd.read_csv("results/splits.csv")
        test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)
        
        ds_clean = MMOTUSegmentationDataset(test_df, image_size=256, apply_inpainting=True, augment=False)
        ds_raw = MMOTUSegmentationDataset(test_df, image_size=256, apply_inpainting=False, augment=False)
        
        print("Evaluating Model A on Clean...")
        dice_a_clean, _ = eval_seg(model_a, ds_clean, device, "A × Clean")
        print("Evaluating Model B on Clean...")
        dice_b_clean, _ = eval_seg(model_b, ds_clean, device, "B × Clean")
        print("Evaluating Model B on Raw...")
        dice_b_raw, _ = eval_seg(model_b, ds_raw, device, "B × Raw")
        
        # Test 1: Inpainting Gain (A_clean vs B_clean)
        stat_gain, p_gain = wilcoxon(dice_a_clean, dice_b_clean, alternative='two-sided')
        
        # Test 2: Shortcut Inflation (B_raw vs B_clean)
        stat_inf, p_inf = wilcoxon(dice_b_raw, dice_b_clean, alternative='two-sided')
        
        md_content.append("### 1. Ablation Effect Sizes (Wilcoxon Signed-Rank Test)\n")
        output_content.append("--- Ablation Effect Sizes (Wilcoxon Signed-Rank Test) ---")
        output_content.append(f"Test: Inpainting Gain (Model A Clean vs Model B Clean)")
        output_content.append(f"  Mean Difference: {np.mean(dice_a_clean) - np.mean(dice_b_clean):+.4f} Dice")
        output_content.append(f"  p-value: {p_gain:.4e}")
        output_content.append(f"  Statistically significant (p < 0.05): {p_gain < 0.05}")
        output_content.append(f"Test: Shortcut Inflation (Model B Raw vs Model B Clean)")
        output_content.append(f"  Mean Difference: {np.mean(dice_b_raw) - np.mean(dice_b_clean):+.4f} Dice")
        output_content.append(f"  p-value: {p_inf:.4e}")
        output_content.append(f"  Statistically significant (p < 0.05): {p_inf < 0.05}\n")

    except Exception as e:
        print(f"Error in Ablation Test: {e}")
        output_content.append("--- Ablation Effect Sizes ---\nError running test (checkpoints might be missing).\n")
    

    # --- 2. Classification Ensemble Statistical Test ---
    print("\n--- 2. Running Classification Ensemble Statistical Test ---")
    try:
        config = OmegaConf.load("configs/default.yaml")
        _, val_transforms = get_transforms(config.dataset.image_size)
        
        splits_df = pd.read_csv("results/splits.csv")
        test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)
        ds_test = MMOTUDataset(test_df, transform=val_transforms, return_path=True)
        loader_test = DataLoader(ds_test, batch_size=16, shuffle=False)
        
        model_names = ["densenet121", "resnet50", "efficientnet_b3", "mobilenet_v3_large", "swin_t"]
        loaded_models = []
        best_single_name = "resnet50"
        best_single_model = None
        
        for name in model_names:
            ckpt_path = f"results/checkpoints/exp_001_{name}_best.pt"
            model, _ = get_model(name, num_classes=config.training.num_classes)
            load_checkpoint(ckpt_path, model)
            model.eval()
            model.to(device)
            loaded_models.append(model)
            if name == best_single_name:
                best_single_model = model
                
        ensemble = ModelEnsemble(loaded_models, device)
        
        ens_correct = []
        single_correct = []
        
        print("Evaluating Ensemble vs ResNet-50 on Test Set...")
        with torch.no_grad():
            for batch in loader_test:
                imgs = batch[0].to(device)
                labels = batch[2].to(device)
                
                # Single Model predictions
                single_logits = best_single_model(imgs)
                single_preds = single_logits.argmax(dim=1)
                single_correct.extend((single_preds == labels).cpu().numpy())
                
                # Ensemble predictions
                all_probs = []
                for m in ensemble.models:
                    logits = m(imgs)
                    all_probs.append(torch.softmax(logits, dim=1))
                avg_probs = sum(all_probs) / len(all_probs)
                ens_preds = avg_probs.argmax(dim=1)
                ens_correct.extend((ens_preds == labels).cpu().numpy())
                
        ens_correct = np.array(ens_correct)
        single_correct = np.array(single_correct)
        
        # McNemar's Test
        a = np.sum((ens_correct == 1) & (single_correct == 1))
        b = np.sum((ens_correct == 1) & (single_correct == 0))
        c = np.sum((ens_correct == 0) & (single_correct == 1))
        d = np.sum((ens_correct == 0) & (single_correct == 0))
        
        if b + c == 0:
            statistic = 0.0
            pvalue = 1.0
        else:
            statistic = (abs(b - c) - 1)**2 / (b + c)
            pvalue = chi2.sf(statistic, 1)
        
        ens_acc = np.mean(ens_correct) * 100
        single_acc = np.mean(single_correct) * 100
        
        output_content.append("--- Classification Ensemble Statistical Test (McNemar's Test) ---")
        output_content.append(f"Ensemble Top-1 Accuracy: {ens_acc:.2f}%")
        output_content.append(f"ResNet-50 Top-1 Accuracy: {single_acc:.2f}%")
        output_content.append(f"Difference: {ens_acc - single_acc:+.2f}%")
        output_content.append(f"McNemar's Test Statistic: {statistic:.4f}")
        output_content.append(f"p-value: {pvalue:.4e}")
        output_content.append(f"Statistically significant (p < 0.05): {pvalue < 0.05}\n")
        output_content.append("Contingency Table:")
        output_content.append(f"| | ResNet-50 Correct | ResNet-50 Incorrect |")
        output_content.append(f"| Ensemble Correct | {a} | {b} |")
        output_content.append(f"| Ensemble Incorrect | {c} | {d} |\n")

    except Exception as e:
        print(f"Error in Classification Test: {e}")
        output_content.append("--- Classification Ensemble Statistical Test ---\nError running test (checkpoints might be missing).\n")

    # Write to a TXT file
    output_file = "statistical_results.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(output_content))
    print(f"\nStatistical results saved to '{output_file}'.")

if __name__ == "__main__":
    main()

__version__ = "1.0.0"

import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import gc
import re
import json

from utils.logger import setup_logger
from utils.reproducibility import setup_reproducibility
from utils.checkpoint import load_checkpoint
from data.splits import create_patient_level_splits, load_splits, create_kfold_splits
from data.dataset import get_dataloaders, compute_class_weights, MMOTUDataset
from models.factory import get_model
from training.trainer import Trainer
from xai.xai_runner import XAIRunner
from evaluation.screening import ScreeningAnalyzer
from evaluation.statistical_tests import StatisticalAnalyzer
from evaluation.alignment_metrics import compute_exbale_anchors
from visualization.plots import (
    plot_training_curves, plot_confusion_matrix, plot_backbone_comparison,
    plot_xai_comparison_violin, plot_threshold_heatmap, plot_exbale_vs_correctness,
    plot_screening_results_table, plot_per_class_alignment, plot_roc_curves,
    plot_insertion_deletion, plot_grad_norm_history
)
from visualization.cam_viz import save_qualitative_comparison_figure
from visualization.report import generate_summary_report
from torchvision import transforms

class ConfigNamespace:
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [ConfigNamespace(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, ConfigNamespace(b) if isinstance(b, dict) else b)

def load_config(path: str) -> ConfigNamespace:
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ConfigNamespace(config_dict)

def parse_args():
    parser = argparse.ArgumentParser(description="MMOTU XAI Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--stage", type=int, default=None, help="Run only this stage (0-6)")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--skip_training", action="store_true", help="Skip Stage 2")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of models to run")
    parser.add_argument("--debug", action="store_true", help="Run in fast debug mode")
    parser.add_argument("--kfold", action="store_true", help="Run with 5-fold cross-validation")
    return parser.parse_args()

def setup_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def create_output_dirs(output_config):
    Path(output_config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(output_config.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(output_config.logs_dir).mkdir(parents=True, exist_ok=True)
    Path(output_config.figures_dir).mkdir(parents=True, exist_ok=True)
    Path(output_config.xai_results_dir).mkdir(parents=True, exist_ok=True)
    Path(output_config.viz_dir).mkdir(parents=True, exist_ok=True)

def load_or_discover_metadata(data_config) -> pd.DataFrame:
    raw_dir = Path(data_config.raw_dir)
    
    # Try to find MMOTU style txt files first
    train_cls_path = raw_dir / "train_cls.txt"
    val_cls_path = raw_dir / "val_cls.txt"
    
    if train_cls_path.exists() and val_cls_path.exists():
        print("Found train_cls.txt and val_cls.txt. Loading metadata...")
        data = []
        for cls_file in [train_cls_path, val_cls_path]:
            with open(cls_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    img_name = parts[0]
                    class_label = int(parts[1])
                    
                    img_path = raw_dir / data_config.images_subdir / img_name
                    base_name = Path(img_name).stem
                    mask_path = raw_dir / data_config.masks_subdir / f"{base_name}.PNG"
                    
                    # infer patient
                    m = re.search(r'patient(\d+)', img_name, re.IGNORECASE)
                    patient_id = m.group(1) if m else base_name
                    
                    data.append({
                        'patient_id': patient_id,
                        'image_path': str(img_path),
                        'mask_path': str(mask_path) if mask_path.exists() else None,
                        'class_label': class_label
                    })
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} images from txt files.")
        return df

    meta_path = raw_dir / data_config.metadata_file if data_config.metadata_file else None
    
    if meta_path and meta_path.exists():
        df = pd.read_csv(meta_path)
        return df
        
    # Auto-discover
    print("Auto-discovering dataset...")
    images_dir = raw_dir / data_config.images_subdir
    masks_dir = raw_dir / data_config.masks_subdir
    
    data = []
    class_folders = [f for f in images_dir.iterdir() if f.is_dir()]
    if not class_folders:
        for img_path in images_dir.glob("*.*"):
            if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg']: continue
            
            m = re.search(r'patient(\d+)', img_path.stem, re.IGNORECASE)
            patient_id = m.group(1) if m else img_path.stem
            
            cm = re.search(r'class(\d+)', img_path.stem, re.IGNORECASE)
            class_label = int(cm.group(1)) if cm else 0
            
            mask_path = masks_dir / f"{img_path.stem}.PNG"
            if not mask_path.exists():
                mask_path = masks_dir / img_path.name
            
            data.append({
                'patient_id': patient_id,
                'image_path': str(img_path),
                'mask_path': str(mask_path) if mask_path.exists() else None,
                'class_label': class_label
            })
    else:
        for class_dir in class_folders:
            digits = re.findall(r'\d+', class_dir.name)
            class_label = int(digits[-1]) if digits else 0
            
            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg']: continue
                m = re.search(r'patient(\d+)', img_path.stem, re.IGNORECASE)
                patient_id = m.group(1) if m else img_path.stem
                
                mask_path = masks_dir / class_dir.name / f"{img_path.stem}.PNG"
                if not mask_path.exists():
                    mask_path = masks_dir / img_path.name
                    
                data.append({
                    'patient_id': patient_id,
                    'image_path': str(img_path),
                    'mask_path': str(mask_path) if mask_path.exists() else None,
                    'class_label': class_label
                })
                
    df = pd.DataFrame(data)
    print(f"Discovered {len(df)} images.")
    return df


def _resolve_first_matching_key(mapping: dict, base_name: str) -> str | None:
    if base_name in mapping:
        return base_name
    for key in mapping:
        if key.startswith(f"{base_name}_fold"):
            return key
    return None


def _build_qualitative_rows_for_model(model_name, all_xai_results, trained_models, config, device, logger):
    result_key = _resolve_first_matching_key(all_xai_results, model_name)
    if not result_key:
        return None

    result_df = all_xai_results[result_key].copy()
    if "cam_threshold" in result_df.columns:
        filtered_df = result_df[result_df["cam_threshold"] == 0.5].copy()
        if filtered_df.empty:
            filtered_df = result_df
    else:
        filtered_df = result_df

    high_row = filtered_df[filtered_df["xai_method"] == "eigencam"].sort_values("exbale", ascending=False).head(1)
    low_row = filtered_df[filtered_df["xai_method"] == "gradcam"].sort_values("exbale", ascending=True).head(1)

    if high_row.empty or low_row.empty:
        return None

    checkpoint_key = _resolve_first_matching_key(trained_models, model_name)
    if not checkpoint_key:
        return None

    from PIL import Image
    from torchvision import transforms as _transforms
    from xai.cam_methods import CAMExplainer
    from xai.gradient_methods import GradientExplainer
    from models.factory import get_model
    from utils.checkpoint import load_checkpoint

    model, _ = get_model(model_name, num_classes=config.training.num_classes)
    load_checkpoint(trained_models[checkpoint_key], model)
    model = model.to(device).eval()

    cam_explainer = CAMExplainer(model, model_name, device)
    grad_explainer = GradientExplainer(model, device)

    image_size = config.data.image_size
    image_transform = _transforms.Compose([
        _transforms.Resize((image_size, image_size)),
        _transforms.ToTensor(),
        _transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = _transforms.Compose([
        _transforms.Resize((image_size, image_size), interpolation=_transforms.InterpolationMode.NEAREST),
        _transforms.ToTensor()
    ])
    display_resize = _transforms.Resize((image_size, image_size))

    def build_row(row, row_label: str):
        image_path = row["image_path"]
        mask_path = row.get("mask_path")
        pred_class = int(row["predicted_class"])

        image_pil = Image.open(image_path).convert("RGB")
        display_image = np.array(display_resize(image_pil))
        model_input = image_transform(image_pil).unsqueeze(0).to(device)

        if mask_path and isinstance(mask_path, str) and Path(mask_path).exists():
            mask_pil = Image.open(mask_path).convert("L")
        else:
            mask_pil = Image.new("L", image_pil.size, 0)
        display_mask = np.array(mask_transform(display_resize(mask_pil)).squeeze().cpu().numpy() > 0, dtype=np.uint8) * 255

        cams = {
            "gradcam": cam_explainer.compute_cam(model_input, pred_class, "gradcam"),
            "scorecam": cam_explainer.compute_cam(model_input, pred_class, "scorecam"),
            "eigencam": cam_explainer.compute_cam(model_input, pred_class, "eigencam"),
            "saliency": grad_explainer.compute_saliency(model_input.clone(), pred_class),
        }

        return {
            "image": display_image,
            "mask": display_mask,
            "cams": cams,
            "row_label": row_label,
        }

    high_example = high_row.iloc[0]
    low_example = low_row.iloc[0]

    rows = [
        build_row(high_example, f"High ExBale\n{model_name} Eigen CAM\nExBale = {high_example['exbale']:.2f}"),
        build_row(low_example, f"Low ExBale\n{model_name} Grad CAM\nExBale = {low_example['exbale']:.2f}"),
    ]

    return rows


def generate_qualitative_comparison_figures(all_xai_results, trained_models, config, device, logger):
    special_dir = Path(config.output.figures_dir) / "qualitative_comparisons"
    special_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []
    figure_index = 1

    ordered_models = [m for m in config.training.models_to_train if m in all_xai_results]
    for model_name in ordered_models:
        rows = _build_qualitative_rows_for_model(model_name, all_xai_results, trained_models, config, device, logger)
        if rows is None:
            logger.warning(f"Skipping qualitative figure for {model_name}; matching rows or checkpoint not found.")
            continue

        if model_name == "swin_t":
            primary_path = Path(config.output.figures_dir) / "qualitative_comparison.pdf"
            save_qualitative_comparison_figure(rows, str(primary_path))
            logger.info(f"Saved primary qualitative comparison figure to {primary_path}")

        numbered_path = special_dir / f"qualitative_comparison{figure_index}_{model_name}.pdf"
        save_qualitative_comparison_figure(rows, str(numbered_path))
        generated_files.append(str(numbered_path))
        logger.info(f"Saved qualitative comparison figure to {numbered_path}")
        figure_index += 1

    if generated_files:
        manifest_path = special_dir / "qualitative_comparison_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({"figures": generated_files}, f, indent=2)
        logger.info(f"Saved qualitative figure manifest to {manifest_path}")

    return generated_files

def main():
    args = parse_args()
    config = load_config(args.config)
    
    if args.models:
        config.training.models_to_train = [m.strip() for m in args.models.split(',')]
        
    if args.debug:
        config.training.num_epochs = 3
        config.xai.run_shap = False
        config.xai.run_faithfulness = False
        config.xai.cam_methods = ["gradcam"]
        
    run_stage = lambda s: args.stage is None or args.stage == s
    
    # ── Stage 0: Setup ──
    setup_reproducibility(config.experiment.random_seed)
    device = setup_device(config.experiment.device)
    create_output_dirs(config.output)
    logger = setup_logger(config.output.logs_dir, config.experiment.run_name)
    logger.info("=" * 60)
    logger.info(f"Run: {config.experiment.run_name} | Device: {device} | Debug: {args.debug}")
    
    # ── Stage 1: Data ──
    if run_stage(1):
        # Stage 1: Data Preparation
        logger.info("Stage 1: Data Preparation")

        regenerate_splits = False
        if not Path(config.data.splits_csv).exists():
            regenerate_splits = True
        else:
            # Check if existing splits match the debug/full intent
            existing_splits = pd.read_csv(config.data.splits_csv)
            if not args.debug and len(existing_splits) <= 100:
                logger.info("Existing splits file looks like a debug subset. Regenerating for full run...")
                regenerate_splits = True
            elif args.debug and len(existing_splits) > 100:
                logger.info("Existing splits file is large. Regenerating for debug run...")
                regenerate_splits = True

        if regenerate_splits:
            logger.info("Creating patient-level splits...")
            metadata = load_or_discover_metadata(config.data)

            if args.debug:
                # subset for debug
                metadata = metadata.sample(min(100, len(metadata)), random_state=config.experiment.random_seed).reset_index(drop=True)

            if args.kfold or getattr(config.experiment, 'use_kfold', False):
                n_splits = getattr(config.experiment, 'kfold_n_splits', 5)
                logger.info(f"Generating {n_splits}-fold splits...")
                fold_dfs = create_kfold_splits(metadata, n_splits=n_splits)
                for i, f_df in enumerate(fold_dfs):
                    f_df.to_csv(f"{config.output.results_dir}/splits_fold{i}.csv", index=False)
                # Set default splits_csv to fold 0 for subsequent steps if not looping
                splits_df = fold_dfs[0]
                splits_df.to_csv(config.data.splits_csv, index=False)
            else:
                splits_df = create_patient_level_splits(metadata)
                splits_df.to_csv(config.data.splits_csv, index=False)
            
        train_df, val_df, test_df = load_splits(config.data.splits_csv)
        class_weights = compute_class_weights(train_df)
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Pre-compute wcis stats for normalization
        with open(Path(config.output.results_dir) / "wcis_normalization_stats.json", 'w') as f:
            json.dump({"wcis_global_min": 0.0, "wcis_global_max": 1.0}, f)

        # Compute ExBale Anchors
        logger.info("Computing ExBale anchors...")
        val_transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor()
        ])
        test_dataset = MMOTUDataset(test_df, transform=val_transform, mask_transform=val_transform)
        test_masks = []
        for i in range(min(100, len(test_dataset))):
            _, mask, _ = test_dataset[i]
            test_masks.append(mask.numpy().squeeze() * 255.0)
            
        anchor_stats = compute_exbale_anchors(test_masks)
        with open(Path(config.output.results_dir) / "exbale_anchors.json", 'w') as f:
            json.dump(anchor_stats, f, indent=4)
        logger.info(f"ExBale anchors saved: {anchor_stats}")
            
    # ── Stage 2: Training ──
    trained_models = {}
    if run_stage(2) and not args.skip_training:
        logger.info("=" * 60)
        logger.info("Stage 2: Training")
        
        n_folds = getattr(config.experiment, 'kfold_n_splits', 5) if (args.kfold or getattr(config.experiment, 'use_kfold', False)) else 1
        
        for fold in range(n_folds):
            if n_folds > 1:
                logger.info(f"--- Training Fold {fold} ---")
                splits_path = f"{config.output.results_dir}/splits_fold{fold}.csv"
            else:
                splits_path = config.data.splits_csv
                
            train_df, val_df, _ = load_splits(splits_path)
            class_weights = compute_class_weights(train_df)
            
            for model_name in config.training.models_to_train:
                run_name = f"{config.experiment.run_name}_{model_name}"
                if n_folds > 1:
                    run_name += f"_fold{fold}"
                    
                logger.info(f"=== Training {model_name} (Fold {fold}) ===")
                
                model, in_features = get_model(model_name, num_classes=config.training.num_classes, dropout=config.training.dropout)
                model = model.to(device)
                
                # Inject configs
                config.training.use_amp = config.experiment.use_amp
                config.training.gradient = config.gradient
                
                # New Enhancement Flags
                config.training.use_mixup = getattr(config.training, 'use_mixup', True)
                config.training.use_swa = getattr(config.training, 'use_swa', True)
                
                train_loader, val_loader, _ = get_dataloaders(splits_path, config)
                
                trainer = Trainer(model, train_loader, val_loader, config.training, device, logger, config.output.checkpoints_dir, run_name, class_weights=class_weights)
                
                best_ckpt_path = trainer.train()
                
                if n_folds == 1:
                    trained_models[model_name] = best_ckpt_path
                else:
                    trained_models[f"{model_name}_fold{fold}"] = best_ckpt_path
                    
                logger.info(f"Best checkpoint saved: {best_ckpt_path}")
                
                torch.cuda.empty_cache()
                gc.collect()
            
    elif args.skip_training:
        # Populate trained_models from existing files
        for model_name in config.training.models_to_train:
            ckpt_path = Path(config.output.checkpoints_dir) / f"{config.experiment.run_name}_{model_name}_best.pt"
            if ckpt_path.exists():
                trained_models[model_name] = str(ckpt_path)
            else:
                logger.warning(f"Skipped training but checkpoint not found for {model_name}: {ckpt_path}")

    # ── Stage 3: XAI ──
    all_xai_results = {}
    if run_stage(3):
        logger.info("=" * 60)
        logger.info("Stage 3: XAI Generation")
        _, _, test_df = load_splits(config.data.splits_csv)
        
        val_transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        test_dataset_with_paths = MMOTUDataset(test_df, transform=val_transform, mask_transform=mask_transform, return_path=True)
        
        # Inject wcis path into config dynamically
        config.xai.wcis_stats_path = str(Path(config.output.results_dir) / "wcis_normalization_stats.json")
        
        for trained_key, ckpt_path in trained_models.items():
            logger.info(f"=== XAI for {trained_key} ===")
            
            # Extract base model name for get_model
            base_model_name = trained_key
            if "_fold" in trained_key:
                base_model_name = trained_key.split("_fold")[0]
                
            model, _ = get_model(base_model_name, num_classes=config.training.num_classes)
            load_checkpoint(ckpt_path, model)
            model = model.to(device).eval()
            
            xai_runner = XAIRunner(model, trained_key, test_dataset_with_paths, config.xai, device, logger)
            
            if config.xai.run_shap:
                train_df, _, _ = load_splits(config.data.splits_csv)
                bg_dataset = MMOTUDataset(train_df, transform=val_transform, mask_transform=mask_transform)
                xai_runner.init_shap(bg_dataset)
                
            results_df = xai_runner.run(output_dir=config.output.xai_results_dir)
            all_xai_results[model_name] = results_df
            
            torch.cuda.empty_cache()
            gc.collect()

    # ── Stage 3.5: Ensemble Evaluation ──
    if (run_stage(3) or run_stage(4)) and len(trained_models) > 1 and getattr(config.training, 'use_ensemble', True):
        logger.info("=" * 60)
        logger.info("Stage 3.5: Ensemble Evaluation")
        from evaluation.ensemble import build_ensemble_from_checkpoints
        from training.metrics import compute_classification_metrics
        
        _, _, test_df = load_splits(config.data.splits_csv)
        val_transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_dataset = MMOTUDataset(test_df, transform=val_transform, return_path=True)
        
        ensemble = build_ensemble_from_checkpoints(trained_models, config.training.models_to_train, config, device)
        if len(ensemble.models) > 0:
            probs, labels, _ = ensemble.predict_dataset(test_dataset)
            preds = np.argmax(probs, axis=1)
            metrics = compute_classification_metrics(preds, labels, probs, num_classes=config.training.num_classes)
            
            logger.info(f"Ensemble Test Top-1: {metrics['top1_acc']:.4f}")
            logger.info(f"Ensemble Test Macro F1: {metrics['macro_f1']:.4f}")
            
            # Save ensemble results
            ensemble_res_path = Path(config.output.results_dir) / "ensemble_metrics.json"
            with open(ensemble_res_path, 'w') as f:
                json.dump(metrics, f, indent=4)

    # If skipping early stages but wanting later stages, load XAI results
    if not run_stage(3) and (run_stage(4) or run_stage(5) or run_stage(6)):
        for model_name in config.training.models_to_train:
            res_path = Path(config.output.xai_results_dir) / f"xai_results_{model_name}.csv"
            if res_path.exists():
                all_xai_results[model_name] = pd.read_csv(res_path)

    # ── Stage 4: Evaluation ──
    if run_stage(4):
        logger.info("=" * 60)
        logger.info("Stage 4: Evaluation")
        
        all_sweeps = {}
        all_per_class = {}
        
        if config.training.use_tta:
            from evaluation.tta import compute_tta_predictions
            from training.metrics import compute_classification_metrics, compute_top2_accuracy
            import json as _json

            # Rebuild test_dataset_with_paths if not in scope (e.g. --stage 4 standalone run)
            if 'test_dataset_with_paths' not in dir():
                from torchvision import transforms as _transforms
                _, _, _test_df = load_splits(config.data.splits_csv)
                _val_transform = _transforms.Compose([
                    _transforms.Resize((config.data.image_size, config.data.image_size)),
                    _transforms.ToTensor(),
                    _transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                _mask_transform = _transforms.Compose([
                    _transforms.Resize((config.data.image_size, config.data.image_size),
                                       interpolation=_transforms.InterpolationMode.NEAREST),
                    _transforms.ToTensor()
                ])
                test_dataset_with_paths = MMOTUDataset(
                    _test_df, transform=_val_transform,
                    mask_transform=_mask_transform, return_path=True
                )

            tta_all_metrics = {}
            for model_name, ckpt_path in trained_models.items():
                # Prefer SWA checkpoint if it exists, fall back to best checkpoint
                swa_path = Path(config.output.checkpoints_dir) / f"{config.experiment.run_name}_{model_name}_swa.pt"
                load_path = str(swa_path) if swa_path.exists() else ckpt_path
                logger.info(f"TTA loading: {load_path}")

                tta_model, _ = get_model(model_name, num_classes=config.training.num_classes)
                load_checkpoint(load_path, tta_model)
                tta_model = tta_model.to(device)
                tta_model.eval()  # Explicit eval AFTER loading weights

                probs, labels, paths = compute_tta_predictions(tta_model, test_dataset_with_paths, device)
                preds = np.argmax(probs, axis=1)

                metrics = compute_classification_metrics(
                    preds, np.array(labels), probs,
                    num_classes=config.training.num_classes
                )
                metrics['top2_acc'] = compute_top2_accuracy(
                    torch.tensor(probs), torch.tensor(np.array(labels))
                )
                tta_all_metrics[model_name] = metrics

                tta_out_path = Path(config.output.results_dir) / f"tta_metrics_{model_name}.json"
                with open(tta_out_path, 'w') as f:
                    m = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in metrics.items()}
                    _json.dump(m, f, indent=4)
                logger.info(f"TTA {model_name}: Top1={metrics['top1_acc']:.4f} F1={metrics['macro_f1']:.4f}")

            torch.cuda.empty_cache()
            gc.collect()

        for model_name, results_df in all_xai_results.items():
            if results_df.empty: continue
            logger.info(f"=== Evaluating {model_name} ===")
            analyzer = ScreeningAnalyzer(results_df)
            
            sweep_df = analyzer.threshold_sweep(sc_thresholds=config.evaluation.screening.sc_thresholds, 
                                                cc_thresholds=config.evaluation.screening.cc_thresholds)
            
            optimal = analyzer.find_optimal_thresholds(sweep_df)
            per_class = analyzer.per_class_reliability()
            roc_info = analyzer.exbale_roc_analysis()
            
            logger.info(f"Optimal thresholds: SC<{optimal['sc_thresh']}, CC<{optimal['cc_thresh']}")
            logger.info(f"ExBale ROC-AUC: {roc_info['auc_roc']:.4f}")
            
            sweep_df.to_csv(f"{config.output.results_dir}/{model_name}_threshold_sweep.csv", index=False)
            per_class.to_csv(f"{config.output.results_dir}/{model_name}_per_class_reliability.csv")
            
            all_sweeps[model_name] = sweep_df
            all_per_class[model_name] = per_class
            
            # Enhancement 7: Correct vs Incorrect ExBale
            stat = StatisticalAnalyzer()
            logger.info(f"Running correct vs incorrect ExBale analysis for {model_name}...")
            cvs_df = stat.correct_vs_incorrect_exbale(results_df, cam_threshold=0.5)
            cvs_df.to_csv(f"{config.output.results_dir}/{model_name}_correct_vs_incorrect.csv", index=False)
            
        # Statistical Tests
        stat = StatisticalAnalyzer()
        if "densenet121" in all_xai_results and not all_xai_results["densenet121"].empty:
            xai_comparison = stat.compare_xai_methods(all_xai_results["densenet121"])
            logger.info("Statistical Tests computed for DenseNet121.")
            
        gradcam_results = {}
        for name, df in all_xai_results.items():
            if not df.empty and 'xai_method' in df.columns:
                gradcam_results[name] = df[df['xai_method'] == 'gradcam']
                
        if gradcam_results:
            backbone_comparison = stat.compare_backbones(gradcam_results)
            if not backbone_comparison['tukey_results_df'].empty:
                backbone_comparison['tukey_results_df'].to_csv(f"{config.output.results_dir}/backbone_tukey_hsd.csv", index=False)

    # ── Stage 5: Visualizations ──
    if run_stage(5):
        logger.info("=" * 60)
        logger.info("Stage 5: Visualization")
        
        for model_name in config.training.models_to_train:
            log_path = Path(config.output.logs_dir) / f"{config.experiment.run_name}_{model_name}_training_log.csv"
            if log_path.exists():
                plot_training_curves(str(log_path), f"{config.output.figures_dir}/{model_name}_training.png")
                plot_grad_norm_history(str(log_path), f"{config.output.figures_dir}/{model_name}_grad_norm.png")
                
            if model_name in all_xai_results and not all_xai_results[model_name].empty:
                df = all_xai_results[model_name]
                plot_xai_comparison_violin(df, save_path=f"{config.output.figures_dir}/{model_name}_xai_violin.png")
                plot_exbale_vs_correctness(df, save_path=f"{config.output.figures_dir}/{model_name}_exbale_vs_conf.png")
                plot_insertion_deletion(df, save_path=f"{config.output.figures_dir}/{model_name}_faithfulness.png")
                
            sweep_path = Path(config.output.results_dir) / f"{model_name}_threshold_sweep.csv"
            if sweep_path.exists():
                sweep_df = pd.read_csv(sweep_path)
                plot_threshold_heatmap(sweep_df, save_path=f"{config.output.figures_dir}/{model_name}_threshold_heatmap.pdf")
                plot_screening_results_table(sweep_df, save_path=f"{config.output.figures_dir}/{model_name}_screening_table.png")
                
            per_class_path = Path(config.output.results_dir) / f"{model_name}_per_class_reliability.csv"
            if per_class_path.exists():
                pc_df = pd.read_csv(per_class_path).set_index('class_label')
                plot_per_class_alignment(pc_df, save_path=f"{config.output.figures_dir}/{model_name}_per_class.png")
                
        if all_xai_results:
            plot_backbone_comparison(all_xai_results, save_path=f"{config.output.figures_dir}/backbone_comparison.png")
            generate_qualitative_comparison_figures(all_xai_results, trained_models, config, device, logger)
            
    # ── Stage 6: Summary ──
    if run_stage(6):
        logger.info("=" * 60)
        logger.info("Stage 6: Summary Report")
        generate_summary_report(all_xai_results, trained_models, config, save_path=f"{config.output.results_dir}/summary_report.txt")
        
    logger.info("=== Pipeline complete ===")

if __name__ == "__main__":
    main()

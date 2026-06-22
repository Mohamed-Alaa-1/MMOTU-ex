"""
evaluation/ensemble.py
Implements Model Ensemble for combining multiple backbone predictions.
Supports uniform and performance-weighted averaging.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.special import softmax

from models.factory import get_model
from utils.checkpoint import load_checkpoint
from evaluation.tta import TTAEvaluator

class ModelEnsemble:
    """
    ModelEnsemble loads multiple trained checkpoints and averages softmax probabilities.
    """
    def __init__(self, models: list[nn.Module], device: torch.device, weights: list[float] = None):
        self.models = models
        self.device = device
        for m in self.models:
            m.to(device)
            m.eval()
        
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights) / sum(weights)

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Weighted average of softmax outputs for a single image tensor."""
        all_probs = []
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        for model in self.models:
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            
        # Weighted average
        avg_probs = np.zeros_like(all_probs[0])
        for p, w in zip(all_probs, self.weights):
            avg_probs += p * w
            
        return avg_probs.squeeze(0)

    def predict_dataset(self, dataset):
        """Returns (probs [N,C], labels [N], paths [N]) for a full dataset."""
        all_probs = []
        all_labels = []
        all_paths = []
        
        for i in tqdm(range(len(dataset)), desc="Ensemble Inference"):
            # MMOTUDataset returns (image, mask, label, path)
            img, _, label, path = dataset[i]
            probs = self.predict(img)
            
            all_probs.append(probs)
            all_labels.append(label)
            all_paths.append(path)
            
        return np.stack(all_probs), np.array(all_labels), all_paths

    def predict_with_tta(self, tta_evaluators: list[TTAEvaluator], dataset):
        """Combines ensemble with TTA for maximum accuracy."""
        all_probs = []
        all_labels = []
        all_paths = []
        
        from PIL import Image
        
        for i in tqdm(range(len(dataset)), desc="Ensemble + TTA Inference"):
            if hasattr(dataset, 'df'):
                row = dataset.df.iloc[i]
                img_path = row['image_path']
                label = int(row['class_label'])
            else:
                sample = dataset[i]
                _, _, label, img_path = sample[:4]
                
            pil_image = Image.open(img_path).convert('RGB')
            
            model_probs = []
            for tta in tta_evaluators:
                probs = tta.predict(pil_image)
                model_probs.append(probs)
                
            # Weighted average of TTA results from each model
            avg_probs = np.zeros_like(model_probs[0])
            for p, w in zip(model_probs, self.weights):
                avg_probs += p * w
                
            all_probs.append(avg_probs)
            all_labels.append(label)
            all_paths.append(img_path)
            
        return np.stack(all_probs), np.array(all_labels), all_paths

def build_ensemble_from_checkpoints(checkpoint_paths: dict, model_names: list, config, device: torch.device) -> ModelEnsemble:
    """Loads each checkpoint via load_checkpoint(), builds ensemble with uniform weights."""
    loaded_models = []
    
    # If checkpoint_paths has more entries than model_names (e.g. folds), 
    # and we want to ensemble all of them, we should iterate over checkpoint_paths.
    # But usually we want to ensemble the models specified in model_names.
    
    for trained_key, ckpt_path in checkpoint_paths.items():
        # Check if this key belongs to one of the models we want to train
        base_model_name = trained_key
        if "_fold" in trained_key:
            base_model_name = trained_key.split("_fold")[0]
            
        if base_model_name not in model_names:
            continue
            
        print(f"Loading {trained_key} for ensemble...")
        model, _ = get_model(base_model_name, num_classes=config.training.num_classes)
        load_checkpoint(ckpt_path, model)
        loaded_models.append(model)
        
    return ModelEnsemble(loaded_models, device)

def compute_weighted_ensemble(val_f1_scores: dict) -> list[float]:
    """Softmax over F1 scores to produce performance-proportional weights."""
    models = list(val_f1_scores.keys())
    scores = np.array([val_f1_scores[m] for m in models])
    weights = softmax(scores)
    return weights.tolist()

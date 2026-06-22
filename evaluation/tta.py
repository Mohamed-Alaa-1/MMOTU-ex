"""
evaluation/tta.py
Implements Test-Time Augmentation (TTA) for robust inference.
Applies 8 deterministic augmentations and averages predictions.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

class TTAEvaluator:
    """
    TTAEvaluator applies 8 deterministic augmentations to a single PIL image,
    runs inference for each, and returns the mean probability vector.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Standard normalization used in val_transform
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        # Define 8 deterministic augmentations
        # 1. Original
        # 2. H-flip
        # 3. V-flip
        # 4. Rot90
        # 5. Rot180
        # 6. Rot270
        # 7. Center-crop 256 -> 224
        # 8. Resize 248 -> Center 224
        
        self.base_resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()
        
    def _get_aug_tensors(self, pil_image: Image.Image) -> list[torch.Tensor]:
        """Returns 8 augmented and normalized tensors."""
        img = self.base_resize(pil_image)
        
        augs = [
            img,                                      # 1. Original
            transforms.functional.hflip(img),          # 2. H-flip
            transforms.functional.vflip(img),          # 3. V-flip
            transforms.functional.rotate(img, 90),     # 4. Rot90
            transforms.functional.rotate(img, 180),    # 5. Rot180
            transforms.functional.rotate(img, 270),    # 6. Rot270
        ]
        
        # 7. Center-crop 256 -> 224
        img_256 = transforms.Resize((256, 256))(pil_image)
        augs.append(transforms.functional.center_crop(img_256, (224, 224)))
        
        # 8. Resize 248 -> Center 224
        img_248 = transforms.Resize((248, 248))(pil_image)
        augs.append(transforms.functional.center_crop(img_248, (224, 224)))
        
        # Convert to tensor and normalize
        return [self.normalize(self.to_tensor(a)) for a in augs]

    @torch.no_grad()
    def predict(self, pil_image: Image.Image) -> np.ndarray:
        """Runs TTA inference and returns mean probability vector."""
        self.model.eval()  # Guard: enforce eval mode on every predict call
        aug_tensors = self._get_aug_tensors(pil_image)
        batch = torch.stack(aug_tensors).to(self.device)
        
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)
        mean_probs = probs.mean(dim=0).cpu().numpy()
        
        return mean_probs

def compute_tta_predictions(model: nn.Module, dataset, device: torch.device):
    """
    Computes TTA predictions for a full dataset.
    Dataset must return image paths (MMOTUDataset with return_path=True).
    """
    evaluator = TTAEvaluator(model, device)
    all_probs = []
    all_labels = []
    all_paths = []
    
    # We iterate manually to access raw PIL images via paths
    for i in tqdm(range(len(dataset)), desc="TTA Inference"):
        # We need the path to re-open as PIL
        # MMOTUDataset.__getitem__ returns (image, mask, label, path, mask_path) if return_path=True
        # But image is already transformed. We want the raw image for TTA control.
        
        # Check if dataset has samples/metadata
        if hasattr(dataset, 'df'):
            row = dataset.df.iloc[i]
            img_path = row['image_path']
            label = int(row['class_label'])
        else:
            # Fallback to __getitem__ if possible, though less efficient for PIL re-read
            sample = dataset[i]
            _, _, label, img_path = sample[:4]
            
        pil_image = Image.open(img_path).convert('RGB')
        probs = evaluator.predict(pil_image)
        
        all_probs.append(probs)
        all_labels.append(label)
        all_paths.append(img_path)
        
    return np.stack(all_probs), all_labels, all_paths

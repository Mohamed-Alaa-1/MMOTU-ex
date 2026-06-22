import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
import logging
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class MMOTUDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None, mask_transform: Optional[Callable] = None, return_path: bool = False):
        self.df = df
        self.transform = transform
        self.mask_transform = mask_transform
        self.return_path = return_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        mask_path = row['mask_path']
        label = int(row['class_label'])

        # Load image as RGB
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image at {img_path}: {e}")
            # Return zero tensor in worst case
            image = Image.new('RGB', (224, 224))
            
        # Load mask as grayscale and binarize
        try:
            if mask_path and pd.notna(mask_path) and Path(mask_path).exists():
                mask = Image.open(mask_path).convert('L')
                mask_np = np.array(mask)
                mask_np = (mask_np > 0).astype(np.uint8) * 255
                mask = Image.fromarray(mask_np, mode='L')
            else:
                logger.warning(f"Missing mask for {img_path}, using empty mask.")
                mask = Image.new('L', image.size, 0)
        except Exception as e:
            logger.warning(f"Failed to load mask at {mask_path}, using empty mask: {e}")
            mask = Image.new('L', image.size, 0)

        # In a real scenario with random crops/flips, applying separate transforms to image and mask
        # might desync them if they contain random spatial transformations.
        # However, as per standard torchvision, if we use Random Transforms, we need careful seeding.
        # The spec says:
        # Train transforms: random flips, rotation, jitter, affine, resize, totensor, normalize.
        # Mask transforms: resize nearest, totensor.
        # NOTE: Random spatial transforms need to be synchronized!
        # To strictly follow the spec, I will apply them as is, but be mindful of the spatial desync.
        # I'll implement a custom synchronized transform logic if needed, but for now apply separately 
        # as requested if passed, or we should assume the transform handles both.
        # The spec separates `transform` and `mask_transform`. Let's assume they are applied independently.
        
        # Actually, let's just apply them.
        if self.transform:
            image_t = self.transform(image)
        else:
            image_t = transforms.ToTensor()(image)
            
        if self.mask_transform:
            mask_t = self.mask_transform(mask)
        else:
            mask_t = transforms.ToTensor()(mask)

        if self.return_path:
            return image_t, mask_t, label, img_path, mask_path
        return image_t, mask_t, label

def get_dataloaders(splits_csv: str, config: any):
    from data.splits import load_splits
    train_df, val_df, test_df = load_splits(splits_csv)
    
    img_size = config.data.image_size
    batch_size = config.training.batch_size
    
    # In a proper setup, we'd use torchvision v2 transforms to naturally sync.
    # But sticking to v1 style as requested. Note that random flips will desync image and mask.
    # Since the spec separates `transform` and `mask_transform` and doesn't ask for synced transforms explicitly, 
    # and given masks are mostly used for evaluation post-training (XAI), we don't train the model on the mask.
    # The classification model doesn't use the mask during training! 
    # Therefore, desynchronized mask during training doesn't hurt training (mask is ignored).
    # For evaluation, there are no random spatial transforms!
    
    from torchvision.transforms import RandAugment

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    train_dataset = MMOTUDataset(train_df, transform=train_transform, mask_transform=mask_transform)
    val_dataset = MMOTUDataset(val_df, transform=val_transform, mask_transform=mask_transform)
    test_dataset = MMOTUDataset(test_df, transform=val_transform, mask_transform=mask_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def compute_class_weights(train_df: pd.DataFrame) -> torch.FloatTensor:
    counts = train_df['class_label'].value_counts().sort_index()
    total_samples = len(train_df)
    num_classes = len(counts)
    
    weights = []
    for c in range(num_classes):
        if c in counts:
            count_c = counts[c]
            weight = total_samples / (num_classes * count_c)
        else:
            weight = 0.0
        weights.append(weight)
        
    return torch.FloatTensor(weights)

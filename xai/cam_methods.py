import torch
import numpy as np
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import logging

def reshape_transform_swin(tensor):
    """Required for Swin Transformer CAM computation."""
    if len(tensor.shape) == 3: # (B, L, C)
        B, L, C = tensor.shape
        H = W = int(np.sqrt(L))
        result = tensor.reshape(B, H, W, C)
    else: # (B, H, W, C)
        result = tensor
    
    # result is (B, H, W, C), grad-cam expects (B, C, H, W)
    result = result.permute(0, 3, 1, 2).clone()
    return result

def reshape_transform_vit(tensor):
    """Required for ViT CAM computation (has CLS token)."""
    # tensor shape is (B, L, C) where L = H*W + 1
    B, L, C = tensor.shape
    H = W = int(np.sqrt(L - 1))
    result = tensor[:, 1:, :].reshape(B, H, W, C)
    result = result.permute(0, 3, 1, 2).clone()
    return result

class CAMExplainer:
    def __init__(self, model, model_name: str, device: torch.device):
        self.model = model
        self.model_name = model_name.lower()
        self.device = device
        self.target_layers = self.get_target_layer()
        self.use_cuda = (device.type == 'cuda')
        
    def get_target_layer(self):
        """Return the correct last conv layer for each backbone."""
        if self.model_name == "densenet121":
            return [self.model.features.denseblock4.denselayer16.conv2]
        elif self.model_name == "resnet50":
            return [self.model.layer4[-1].conv3]
        elif self.model_name == "resnet101":
            return [self.model.layer4[-1].conv3]
        elif self.model_name == "efficientnet_b3":
            return [self.model.features[-1][0]]
        elif self.model_name == "mobilenet_v3_large":
            return [self.model.features[-1][0]]
        elif self.model_name == "swin_t":
            return [self.model.features[-1][-1].norm2]
        elif self.model_name == "vit_b_16":
            return [self.model.encoder.layers[-1].ln_1]
        else:
            logging.getLogger(__name__).warning(f"Unknown model for CAM: {self.model_name}")
            return []

    def compute_cam(self, image_tensor: torch.Tensor, class_idx: int, method: str = "gradcam") -> np.ndarray:
        method = method.lower()
        
        reshape_transform = None
        if self.model_name == "swin_t":
            reshape_transform = reshape_transform_swin
        elif self.model_name == "vit_b_16":
            reshape_transform = reshape_transform_vit
        
        cam_class = None
        if method == "gradcam":
            cam_class = GradCAM
        elif method == "gradcam_pp" or method == "gradcam++":
            cam_class = GradCAMPlusPlus
        elif method == "scorecam":
            cam_class = ScoreCAM
        elif method == "eigencam":
            cam_class = EigenCAM
        else:
            raise ValueError(f"Unknown CAM method: {method}")
            
        targets = [ClassifierOutputTarget(class_idx)]
        
        try:
            with cam_class(model=self.model, target_layers=self.target_layers, reshape_transform=reshape_transform) as cam:
                # ScoreCAM might be heavy, we can batch it but the lib usually handles it or we pass batch_size (if available).
                grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
                cam_res = grayscale_cam[0, :]
                
                # Normalize to [0, 1]
                cam_min, cam_max = cam_res.min(), cam_res.max()
                if cam_max > cam_min:
                    cam_res = (cam_res - cam_min) / (cam_max - cam_min)
                else:
                    cam_res = np.zeros_like(cam_res)
                    
                return cam_res
        except Exception as e:
            logging.getLogger(__name__).error(f"Error computing {method} for {self.model_name}: {e}")
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]), dtype=np.float32)

    def compute_all_cams(self, image_tensor: torch.Tensor, class_idx: int) -> dict:
        methods = ["gradcam", "gradcam_pp", "scorecam", "eigencam"]
        results = {}
        for m in methods:
            results[m] = self.compute_cam(image_tensor, class_idx, method=m)
        return results

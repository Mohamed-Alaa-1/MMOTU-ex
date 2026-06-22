import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import gc
from pathlib import Path
import json

from xai.cam_methods import CAMExplainer
from xai.gradient_methods import GradientExplainer
from xai.shap_methods import SHAPExplainer
from evaluation.alignment_metrics import compute_all_metrics
from evaluation.faithfulness import compute_insertion_deletion_auc

class XAIRunner:
    def __init__(self, model, model_name: str, dataset_with_paths, config: any, device: torch.device, logger):
        self.model = model
        self.model_name = model_name # This can be "densenet121_fold0"
        self.dataset = dataset_with_paths
        self.config = config
        self.device = device
        self.logger = logger
        
        # Extract base model name for CAMExplainer (e.g. "densenet121")
        base_model_name = model_name
        if "_fold" in model_name:
            base_model_name = model_name.split("_fold")[0]
            
        # Disable inplace operations to prevent backward hook errors
        self._disable_inplace(self.model)
        
        self.cam_explainer = CAMExplainer(model, base_model_name, device)
        self.grad_explainer = GradientExplainer(model, device)
        self.shap_explainer = None

    def _disable_inplace(self, model):
        for m in model.modules():
            if hasattr(m, 'inplace'):
                m.inplace = False
        
        # Determine global min/max WCIS
        # In a real scenario, this is passed or pre-computed. If not provided via config, we'll use [0, 1] fallback
        try:
            stats_file = Path(config.wcis_stats_path)
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                self.wcis_global_min = stats['wcis_global_min']
                self.wcis_global_max = stats['wcis_global_max']
            else:
                self.wcis_global_min = 0.0
                self.wcis_global_max = 1.0
        except:
            self.wcis_global_min = 0.0
            self.wcis_global_max = 1.0

    def init_shap(self, background_dataset):
        if self.config.run_shap:
            # Use a dedicated CPU copy for SHAP to avoid device mismatch and OOM
            import copy
            self.shap_model = copy.deepcopy(self.model).cpu().eval()
            shap_device = torch.device('cpu')
            self.shap_explainer = SHAPExplainer(self.shap_model, background_dataset, shap_device, n_background=self.config.shap_n_background)
            
    def run(self, output_dir: str) -> pd.DataFrame:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = []
        
        for idx in tqdm(range(len(self.dataset)), desc=f"Running XAI - {self.model_name}"):
            try:
                # Need to run individually to extract specific tensors, though we can batch if not for SHAP/Captum nuances
                image_t, mask_t, true_label, image_path, mask_path = self.dataset[idx]
                
                # Image tensor with batch dim
                image_batch = image_t.unsqueeze(0).to(self.device)
                
                # Get predicted class
                with torch.no_grad():
                    logits = self.model(image_batch)
                    probs = torch.softmax(logits, dim=1)[0]
                    confidence, predicted_class = torch.max(probs, dim=0)
                    predicted_class = predicted_class.item()
                    confidence = confidence.item()
                    is_correct = (predicted_class == true_label)
                    
                mask_np = mask_t.squeeze().cpu().numpy()
                
                # Process single image
                img_results = self._process_single_image(image_batch, mask_np, predicted_class, true_label, confidence, is_correct, image_path, mask_path)
                results.extend(img_results)
                
            except Exception as e:
                self.logger.error(f"Error processing image {image_path}: {e}")
                
            torch.cuda.empty_cache()
            gc.collect()
            
        df = pd.DataFrame(results)
        df.to_csv(Path(output_dir) / f"xai_results_{self.model_name}.csv", index=False)
        return df

    def _process_single_image(self, image_tensor, mask_np, predicted_class, true_label, confidence, is_correct, image_path, mask_path=None) -> list:
        img_results = []
        
        # 1. CAM Methods
        cams = self.cam_explainer.compute_all_cams(image_tensor, predicted_class)
        
        # 2. Gradient Methods
        grads = self.grad_explainer.compute_all_gradients(image_tensor, predicted_class)
        
        all_maps = {**cams, **grads}
        
        # 3. SHAP
        if self.config.run_shap and self.shap_explainer:
            try:
                shap_map = self.shap_explainer.compute_shap(image_tensor.cpu(), predicted_class)
                all_maps["shap"] = shap_map
            except Exception as e:
                self.logger.error(f"SHAP eval failed for {image_path}: {e}")
                
        # Base dict for this image
        import re
        patient_id_match = re.search(r'patient(\d+)', image_path)
        patient_id = patient_id_match.group(1) if patient_id_match else "unknown"
        
        base_info = {
            "image_path": image_path,
            "mask_path": mask_path,
            "patient_id": patient_id,
            "class_label": true_label,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_correct": is_correct
        }
        
        # 4. Compute metrics
        for method_name, heat_map in all_maps.items():
            metrics = compute_all_metrics(heat_map, mask_np, self.config.cam_thresholds, self.wcis_global_min, self.wcis_global_max)
            
            # 5. Faithfulness
            faith_metrics = {"insertion_auc": np.nan, "deletion_auc": np.nan}
            if self.config.run_faithfulness:
                try:
                    faith_metrics = compute_insertion_deletion_auc(self.model, image_tensor, heat_map, predicted_class, self.device, self.config.faithfulness_n_steps)
                except Exception as e:
                    self.logger.warning(f"Faithfulness failed for {method_name} on {image_path}: {e}")
                    
            for m in metrics:
                row = {**base_info, "xai_method": method_name, "cam_threshold": m["threshold"]}
                row.update({"sc": m["sc"], "cc": m["cc"], "wcis": m["wcis"], "exbale": m["exbale"]})
                row.update(faith_metrics)
                img_results.append(row)
                
        return img_results

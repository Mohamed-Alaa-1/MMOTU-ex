import torch
import torch.nn as nn
import pandas as pd
import json
import time
import logging
from pathlib import Path
import numpy as np

from utils.grad_monitor import GradientMonitor
from utils.checkpoint import save_checkpoint
from training.metrics import compute_classification_metrics, compute_top2_accuracy

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, metric_value: float) -> bool:
        if self.best_score is None:
            self.best_score = metric_value
        elif self.mode == "max" and metric_value < self.best_score + self.min_delta:
            self.counter += 1
        elif self.mode == "min" and metric_value > self.best_score - self.min_delta:
            self.counter += 1
        else:
            self.best_score = metric_value
            self.counter = 0
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device, logger: logging.Logger, checkpoint_dir: str, run_name: str, class_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        
        self.logs_dir = Path(checkpoint_dir).parent / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Segregate parameters for differential learning rates
        head_names = ['classifier', 'fc', 'head', 'heads']
        head_params = []
        backbone_params = []
        
        for name, param in self.model.named_parameters():
            if any(h in name for h in head_names):
                head_params.append(param)
            else:
                backbone_params.append(param)
                
        self.optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": config.lr_backbone, "weight_decay": config.weight_decay},
            {"params": head_params, "lr": config.lr_head, "weight_decay": config.weight_decay},
        ])
        
        # SWA Initialization
        from torch.optim.swa_utils import AveragedModel, SWALR
        self.swa_model = AveragedModel(self.model)
        self.swa_start_epoch = max(1, int(config.num_epochs * 0.75))
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=config.lr_backbone * 0.1, anneal_epochs=5)
        self.use_swa = getattr(config, 'use_swa', True)
        
        self.backbone_params = backbone_params
        
        # Loss function
        from training.losses import WeightedCrossEntropyLoss, FocalLoss
        cw = class_weights.to(device) if class_weights is not None else None
        if config.loss_fn == "weighted_ce":
            self.criterion = WeightedCrossEntropyLoss(weight=cw, label_smoothing=config.label_smoothing)
        else:
            self.criterion = FocalLoss(gamma=config.focal_gamma, alpha=cw, label_smoothing=config.label_smoothing)
            
        # Scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and self.device.type == 'cuda')
        
        # Monitors
        # We assume gradient config is passed via a unified config or config.gradient in main, but let's access it directly if passed.
        # Handling the possibility that `config.gradient` might not be directly available, we will pass it as kwargs or access.
        # For simplicity, assuming `config` is a dict-like or namespace that has `clip_value`, etc. or default to 1.0.
        try:
            clip_val = config.gradient.clip_value
            expl_thresh = config.gradient.explosion_threshold
            skip_thresh = config.gradient.skip_threshold
        except AttributeError:
            # Fallbacks if config structure is flat
            clip_val = getattr(config, 'clip_value', 1.0)
            expl_thresh = getattr(config, 'explosion_threshold', 10.0)
            skip_thresh = getattr(config, 'skip_threshold', 100.0)
            
        self.grad_monitor = GradientMonitor(self.model, self.logger, clip_val, expl_thresh, skip_thresh)
        
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode="max")
        
        self.best_val_metric = -1.0
        self.best_epoch = -1
        self.best_val_metrics_dict = {}
        
    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        
        # Freeze/unfreeze backbone
        if epoch < self.config.freeze_epochs:
            for param in self.backbone_params:
                param.requires_grad = False
        else:
            for param in self.backbone_params:
                param.requires_grad = True
                
        running_loss = 0.0
        skipped_batches = 0
        total_batches = len(self.train_loader)
        
        all_preds, all_targets, all_probs = [], [], []
        grad_norms = []
        
        from training.augmentation import mixup_data, cutmix_data, mixup_criterion
        use_mixup = getattr(self.config, 'use_mixup', True)
        
        for batch_idx, (images, _, labels) in enumerate(self.train_loader):
            if torch.isnan(images).any() or torch.isinf(images).any():
                self.logger.warning(f"NaN/Inf in input batch {batch_idx}, skipping")
                skipped_batches += 1
                continue
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # MixUp / CutMix Logic
            aug_active = False
            if use_mixup and epoch >= self.config.freeze_epochs:
                r = np.random.rand()
                if r < 0.5: # 50% MixUp
                    images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=getattr(self.config, 'mixup_alpha', 0.4), device=self.device)
                    aug_active = True
                elif r < 0.75: # 25% CutMix
                    images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=getattr(self.config, 'cutmix_alpha', 1.0), device=self.device)
                    aug_active = True
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                
                if torch.isnan(logits).any():
                    self.logger.warning(f"NaN in logits at batch {batch_idx}, skipping")
                    skipped_batches += 1
                    continue
                
                if aug_active:
                    loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(logits, labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"NaN/Inf loss={loss.item():.4f} at batch {batch_idx}, skipping")
                    skipped_batches += 1
                    continue
                    
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            should_skip = self.grad_monitor.check_and_clip(loss)
            grad_norms.append(self.grad_monitor.get_grad_norm())
            
            if should_skip:
                self.scaler.update()
                self.optimizer.zero_grad()
                skipped_batches += 1
                continue
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            
        if skipped_batches > 0:
            self.logger.warning(f"Epoch {epoch}: skipped {skipped_batches}/{total_batches} batches")
            
        metrics = compute_classification_metrics(np.array(all_preds), np.array(all_targets), np.array(all_probs), num_classes=self.config.num_classes)
        # We need logits for top-2, compute it approximately or via custom loop. 
        # But we only really care about val top-2. We'll skip train top-2 for brevity or calculate it if needed.
        metrics['loss'] = running_loss / max(1, (total_batches - skipped_batches))
        metrics['grad_norm_mean'] = np.mean(grad_norms) if grad_norms else 0.0
        metrics['grad_norm_max'] = np.max(grad_norms) if grad_norms else 0.0
        metrics['skipped_batches'] = skipped_batches
        return metrics

    def _validate(self, epoch: int) -> dict:
        self.model.eval()
        running_loss = 0.0
        
        all_preds, all_targets, all_probs, all_logits = [], [], [], []
        
        with torch.no_grad():
            for images, _, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    
                running_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
                
        metrics = compute_classification_metrics(np.array(all_preds), np.array(all_targets), np.array(all_probs), num_classes=self.config.num_classes)
        metrics['loss'] = running_loss / len(self.val_loader)
        metrics['top2_acc'] = compute_top2_accuracy(torch.tensor(np.array(all_logits)), torch.tensor(np.array(all_targets)))
        return metrics

    def train(self) -> str:
        best_ckpt_path = Path(self.checkpoint_dir) / f"{self.run_name}_best.pt"
        log_csv_path = self.logs_dir / f"{self.run_name}_training_log.csv"
        
        # Init CSV
        with open(log_csv_path, 'w') as f:
            f.write("epoch,train_loss,val_loss,train_top1_acc,val_top1_acc,train_macro_f1,val_macro_f1,val_balanced_acc,lr_backbone,lr_head,grad_norm_mean,grad_norm_max,skipped_batches,epoch_time_sec\n")
            
        # Schedulers
        def lr_lambda(current_step: int):
            if current_step < self.config.warmup_epochs:
                return float(current_step) / float(max(1, self.config.warmup_epochs))
            # Cosine annealing approximation after warmup
            progress = float(current_step - self.config.warmup_epochs) / float(max(1, self.config.num_epochs - self.config.warmup_epochs))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate(epoch)
            
            epoch_time = time.time() - start_time
            
            lr_bb = self.optimizer.param_groups[0]['lr']
            lr_h = self.optimizer.param_groups[1]['lr']
            
            scheduler.step()
            
            # SWA Step
            if self.use_swa and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            
            # Logging
            with open(log_csv_path, 'a') as f:
                f.write(f"{epoch},{train_metrics['loss']},{val_metrics['loss']},{train_metrics['top1_acc']},{val_metrics['top1_acc']},{train_metrics['macro_f1']},{val_metrics['macro_f1']},{val_metrics['balanced_acc']},{lr_bb},{lr_h},{train_metrics['grad_norm_mean']},{train_metrics['grad_norm_max']},{train_metrics['skipped_batches']},{epoch_time}\n")
                
            self.logger.info(f"Epoch {epoch}/{self.config.num_epochs} - Time: {epoch_time:.1f}s - Train Loss: {train_metrics['loss']:.4f} - Val Loss: {val_metrics['loss']:.4f} - Val F1: {val_metrics['macro_f1']:.4f}")
            
            # Save best
            monitored_metric = val_metrics[self.config.early_stopping_metric]
            is_best = monitored_metric > self.best_val_metric
            if is_best:
                self.best_val_metric = monitored_metric
                self.best_epoch = epoch
                self.best_val_metrics_dict = val_metrics
                
            state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'best_metric': self.best_val_metric,
                'config': self.config
            }
            save_checkpoint(state, is_best, self.checkpoint_dir, self.run_name)
            
            if is_best:
                best_metrics_path = self.logs_dir / f"{self.run_name}_best_metrics.json"
                with open(best_metrics_path, 'w') as f:
                    json.dump({
                        "best_epoch": epoch,
                        "best_val_macro_f1": val_metrics['macro_f1'],
                        "best_val_top1_acc": val_metrics['top1_acc'],
                        "best_val_top2_acc": val_metrics['top2_acc'],
                        "best_val_balanced_acc": val_metrics['balanced_acc'],
                        "run_name": self.run_name,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }, f, indent=4)
                    
            if self.early_stopping.step(monitored_metric):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        # Finalize SWA
        if self.use_swa:
            from torch.optim.swa_utils import update_bn
            self.logger.info("Updating Batch Norm for SWA...")
            update_bn(self.train_loader, self.swa_model, device=self.device)
            swa_path = Path(self.checkpoint_dir) / f"{self.run_name}_swa.pt"
            # AveragedModel wraps the model in .module
            torch.save({'model_state_dict': self.swa_model.module.state_dict()}, swa_path)
            self.logger.info(f"SWA model saved: {swa_path}")
            return str(swa_path)
            
        return str(best_ckpt_path)

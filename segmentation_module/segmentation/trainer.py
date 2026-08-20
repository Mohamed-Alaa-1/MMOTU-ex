"""
segmentation/trainer.py

Standalone training loop for LightweightAuraViT, separate from
training/trainer.py (classification). Two things this trainer must do
that the model itself does not, both documented in
segmentation_extension_plan.md Section 1, point 4:

1. LightweightAuraViT.forward raises ValueError on NaN detection (input,
   after any transformer layer, or output) rather than the soft
   skip-and-continue pattern used elsewhere in this project
   (training/trainer.py + utils/grad_monitor.py for the classification
   backbones). Left unhandled, a single NaN batch crashes the entire
   segmentation training run. This trainer wraps the forward call in a
   try/except that catches this ValueError, logs it the same way
   GradientMonitor already does for classification, and skips the batch.

2. Gradient clipping is applied the same way as the classification
   trainer (torch.nn.utils.clip_grad_norm_), since this model has no
   equivalent of GradientMonitor built in.

3. A learning rate scheduler is supported via the scheduler_type config
   key. Three modes:
     - "cosine_warmup" (default): linear warmup for warmup_epochs, then
       CosineAnnealingLR to eta_min over the remaining epochs.
     - "cosine": CosineAnnealingLR for the full training run, no warmup.
     - "plateau": ReduceLROnPlateau(mode='max') stepping on val_dice.
     - "none": constant LR (AdamW default, no scheduler).

4. Per-epoch metrics are written to a dedicated CSV at
   {log_dir}/{run_name}_history.csv for independent analysis of the
   segmentation training run, separate from the classification pipeline's
   logs in results/logs/.

setup_segmentation_logger() (module-level function) creates a Logger with
both a FileHandler (DEBUG+) and a StreamHandler (INFO+), writing to
{log_dir}/{run_name}.log. This produces a standalone segmentation log
file that can be tailed, grepped, or loaded into a DataFrame without
touching the classification logs.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)

from segmentation.losses import DiceBCELoss
from segmentation.metrics import dice_score, iou_score


# ---------------------------------------------------------------------------
# Logger factory (module-level so it can be called before constructing the
# trainer — useful in train_segmentation.py to log startup info)
# ---------------------------------------------------------------------------

def setup_segmentation_logger(
    log_dir: str,
    run_name: str,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """Create a named Logger that writes to {log_dir}/{run_name}.log and
    the console simultaneously.

    The file handler captures DEBUG and above (full training detail).
    The stream handler captures INFO and above (epoch summaries only),
    keeping the console readable during long runs.

    Args:
        log_dir:       Directory for the log file. Created if absent.
        run_name:      Stem of the log filename and logger name. Use a
                       descriptive name such as 'laura_small_run1'.
        console_level: Minimum log level for console output. Default INFO.

    Returns:
        logging.Logger instance. Can be passed directly to SegmentationTrainer.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger_name = f"segmentation.{run_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Clear any handlers added by a previous call (prevents duplication if
    # the function is called twice in an interactive session or notebook).
    logger.handlers.clear()

    # --- File handler ---
    fh = logging.FileHandler(log_dir / f"{run_name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # --- Console handler ---
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(levelname)-8s  %(message)s"))
    logger.addHandler(ch)

    # Prevent messages from propagating to the root logger so they do not
    # appear duplicated in environments that configure a root handler.
    logger.propagate = False

    return logger


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SegmentationTrainer:
    """Training loop for LightweightAuraViT with:
        - NaN-safe forward pass (ValueError catching + batch skipping)
        - Batch-size-1 guard for ASPP BatchNorm failure (train mode only)
        - Gradient clipping (clip_grad_norm_)
        - LR scheduling (cosine_warmup, cosine, plateau, or none)
        - Per-epoch CSV history for offline analysis
        - Separate log file via setup_segmentation_logger()

    All config values are read from config via getattr(..., default) so the
    config object can be a SimpleNamespace, argparse.Namespace, or any
    object with attribute access.

    Args:
        model:          LightweightAuraViT (or compatible) nn.Module.
        train_loader:   Training DataLoader (drop_last=True required).
        val_loader:     Validation DataLoader.
        config:         Config object with attributes (all optional with defaults):
                          lr (1e-4), weight_decay (1e-4), clip_value (1.0),
                          dice_smooth (1.0), bce_weight (0.5),
                          scheduler_type ("cosine_warmup"), warmup_epochs (5),
                          eta_min (1e-6), lr_patience (10 — plateau only).
        device:         torch.device.
        logger:         logging.Logger (from setup_segmentation_logger or any
                        other source — the trainer logs to it, does not own it).
        checkpoint_dir: Directory for best-model checkpoint files.
        run_name:       Stem used for checkpoint and CSV filenames.
        log_dir:        Directory for the per-epoch CSV history file.
                        Defaults to checkpoint_dir/../logs/segmentation if None.
    """

    _CSV_FIELDS = [
        "epoch", "train_loss", "val_loss", "val_dice", "val_iou",
        "lr", "skipped_batches_train", "skipped_batches_val", "elapsed_s",
    ]

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config,
        device: torch.device,
        logger: logging.Logger,
        checkpoint_dir: str,
        run_name: str,
        log_dir: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name

        # Log directory for the CSV history file. Kept separate from the
        # checkpoint directory so metrics files and model weights live in
        # different places (matching the classification pipeline layout).
        if log_dir is not None:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = self.checkpoint_dir.parent / "logs" / "segmentation"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = DiceBCELoss(
            smooth=getattr(config, "dice_smooth", 1.0),
            bce_weight=getattr(config, "bce_weight", 0.5),
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=getattr(config, "lr", 1e-4),
            weight_decay=getattr(config, "weight_decay", 1e-4),
        )
        self.clip_value = getattr(config, "clip_value", 1.0)
        self.best_val_dice = -1.0
        self.best_epoch = -1

        self._csv_path = self.log_dir / f"{run_name}_history.csv"
        self._csv_file = None
        self._csv_writer = None

    # ------------------------------------------------------------------
    # LR scheduling
    # ------------------------------------------------------------------

    def _build_scheduler(self, num_epochs: int):
        """Construct and return the LR scheduler configured in self.config.

        Supported scheduler_type values:

        "cosine_warmup" (default):
            Linear warmup from (lr * 0.1) to lr over warmup_epochs, then
            CosineAnnealingLR from lr to eta_min over the remaining epochs.
            Recommended for ViT-based models: warmup stabilises the large
            positional embedding gradient at the very start of training,
            while cosine gives a smooth, predictable decay over 100 epochs.

        "cosine":
            CosineAnnealingLR for the full num_epochs, no warmup.
            Use when warmup is not needed (e.g., fine-tuning from a
            checkpoint that already has stable embeddings).

        "plateau":
            ReduceLROnPlateau(mode='max', factor=0.5) stepping on val_dice.
            Halves the LR whenever val_dice does not improve for lr_patience
            (default 10) consecutive epochs. Adaptive but less predictable
            over a fixed 100-epoch budget.

        "none":
            No scheduler; AdamW runs at constant lr throughout. Only useful
            for debugging the optimizer behaviour in isolation.
        """
        scheduler_type = getattr(self.config, "scheduler_type", "cosine_warmup")
        warmup_epochs = getattr(self.config, "warmup_epochs", 5)
        eta_min = getattr(self.config, "eta_min", 1e-6)

        if scheduler_type == "cosine_warmup":
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, num_epochs - warmup_epochs),
                eta_min=eta_min,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )

        elif scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=eta_min
            )

        elif scheduler_type == "plateau":
            patience = getattr(self.config, "lr_patience", 10)
            return ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5,
                patience=patience, min_lr=eta_min,
            )

        elif scheduler_type == "none":
            return None

        else:
            raise ValueError(
                f"Unknown scheduler_type '{scheduler_type}'. "
                f"Choose from: cosine_warmup, cosine, plateau, none."
            )

    def _get_current_lr(self) -> float:
        """Return the current learning rate from the first param group."""
        return float(self.optimizer.param_groups[0]["lr"])

    # ------------------------------------------------------------------
    # CSV history
    # ------------------------------------------------------------------

    def _open_csv(self) -> None:
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._CSV_FIELDS)
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def _write_csv_row(self, row: dict) -> None:
        if self._csv_writer is not None:
            self._csv_writer.writerow(row)
            self._csv_file.flush()

    def _close_csv(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward_safe(self, images: torch.Tensor):
        """Wraps the model's forward call, catching the ValueError
        LightweightAuraViT raises on NaN detection. Returns None if the
        batch should be skipped, otherwise returns the logits.

        Also guards against a real, reproducible failure independent of
        the NaN handling: this architecture's ASPP global-average-pool
        branch collapses spatial dimensions to 1x1 before a BatchNorm2d
        layer, which raises ValueError when training on a batch of size 1
        (BatchNorm cannot compute batch statistics from a single value per
        channel). The train dataloader is built with drop_last=True
        (segmentation/dataset.py) specifically to avoid a trailing
        batch-of-1, but that alone does not protect against a dataset
        smaller than batch_size or a differently configured dataloader, so
        this is checked explicitly here rather than relied upon
        elsewhere. Only relevant in train mode; eval mode (used by
        MC-Dropout uncertainty estimation, which keeps BatchNorm in eval
        regardless of batch size) is unaffected."""
        if self.model.training and images.size(0) < 2:
            self.logger.warning(
                f"Skipping batch of size {images.size(0)} in train mode: "
                f"BatchNorm2d inside this model's ASPP cannot compute batch "
                f"statistics from a single sample. Set drop_last=True on "
                f"the train dataloader to avoid this in normal operation."
            )
            return None
        try:
            return self.model(images)
        except ValueError as e:
            self.logger.warning(f"NaN detected in forward pass, skipping batch: {e}")
            return None

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        running_loss = 0.0
        skipped_batches = 0
        total_batches = len(self.train_loader)

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self._forward_safe(images)
            if logits is None:
                skipped_batches += 1
                continue

            loss = self.criterion(logits, masks)
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(
                    f"NaN/Inf loss={loss.item():.4f} at batch {batch_idx}, skipping"
                )
                skipped_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            self.optimizer.step()

            running_loss += loss.item()

        denom = max(1, total_batches - skipped_batches)
        return {"loss": running_loss / denom, "skipped_batches": skipped_batches}

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        self.model.eval()
        running_loss = 0.0
        dice_scores_list, iou_scores_list = [], []
        skipped_batches = 0

        for images, masks in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            logits = self._forward_safe(images)
            if logits is None:
                skipped_batches += 1
                continue

            loss = self.criterion(logits, masks)
            running_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_bin = probs >= 0.5

            for i in range(preds_bin.shape[0]):
                dice_scores_list.append(dice_score(preds_bin[i, 0], masks_np[i, 0] >= 0.5))
                iou_scores_list.append(iou_score(preds_bin[i, 0], masks_np[i, 0] >= 0.5))

        denom = max(1, len(self.val_loader) - skipped_batches)
        return {
            "loss": running_loss / denom,
            "dice": float(np.mean(dice_scores_list)) if dice_scores_list else 0.0,
            "iou": float(np.mean(iou_scores_list)) if iou_scores_list else 0.0,
            "skipped_batches": skipped_batches,
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: int) -> str:
        """Run the training loop for num_epochs.

        Scheduler stepping:
            - cosine_warmup, cosine: scheduler.step() called once per epoch
              after the validation pass (so epoch 0 starts at the warmup LR,
              not the target LR — matches the SequentialLR convention).
            - plateau: scheduler.step(val_dice) called with the current
              validation Dice so the scheduler can track improvement.
            - none: no stepping.

        Returns:
            Absolute path string of the best checkpoint file.
        """
        best_ckpt_path = self.checkpoint_dir / f"{self.run_name}_best.pt"
        scheduler = self._build_scheduler(num_epochs)
        scheduler_type = getattr(self.config, "scheduler_type", "cosine_warmup")

        self.logger.info(
            f"Starting training: {num_epochs} epochs, "
            f"lr={self._get_current_lr():.2e}, "
            f"scheduler={scheduler_type}, "
            f"device={self.device}"
        )
        self.logger.info(f"CSV history → {self._csv_path}")
        self.logger.info(f"Best checkpoint → {best_ckpt_path}")

        self._open_csv()

        try:
            for epoch in range(num_epochs):
                start = time.time()
                current_lr = self._get_current_lr()

                train_metrics = self._train_epoch(epoch)
                val_metrics = self._validate(epoch)
                elapsed = time.time() - start

                # Step the scheduler (plateau needs the monitored metric)
                if scheduler is not None:
                    if scheduler_type == "plateau":
                        scheduler.step(val_metrics["dice"])
                    else:
                        scheduler.step()

                self.logger.info(
                    f"Epoch {epoch + 1:3d}/{num_epochs} "
                    f"[{elapsed:5.1f}s]  "
                    f"lr={current_lr:.2e}  "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"(skip={train_metrics['skipped_batches']})  "
                    f"val_loss={val_metrics['loss']:.4f}  "
                    f"val_dice={val_metrics['dice']:.4f}  "
                    f"val_iou={val_metrics['iou']:.4f} "
                    f"(skip={val_metrics['skipped_batches']})"
                )

                self._write_csv_row({
                    "epoch": epoch + 1,
                    "train_loss": round(train_metrics["loss"], 6),
                    "val_loss": round(val_metrics["loss"], 6),
                    "val_dice": round(val_metrics["dice"], 6),
                    "val_iou": round(val_metrics["iou"], 6),
                    "lr": f"{current_lr:.2e}",
                    "skipped_batches_train": train_metrics["skipped_batches"],
                    "skipped_batches_val": val_metrics["skipped_batches"],
                    "elapsed_s": round(elapsed, 1),
                })

                if val_metrics["dice"] > self.best_val_dice:
                    self.best_val_dice = val_metrics["dice"]
                    self.best_epoch = epoch + 1
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": (
                            scheduler.state_dict() if scheduler is not None else None
                        ),
                        "best_val_dice": self.best_val_dice,
                        "config": getattr(self.model, "cf", None),
                    }, best_ckpt_path)
                    self.logger.info(
                        f"  ✓ New best val_dice={self.best_val_dice:.4f} "
                        f"at epoch {self.best_epoch} → saved checkpoint"
                    )

        finally:
            # Always close the CSV even if training is interrupted by
            # KeyboardInterrupt or an unhandled exception, so the file
            # is not left in a half-written state.
            self._close_csv()

        self.logger.info(
            f"Training complete. Best val_dice={self.best_val_dice:.4f} "
            f"at epoch {self.best_epoch}/{num_epochs}."
        )
        return str(best_ckpt_path)

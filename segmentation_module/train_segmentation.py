"""
train_segmentation.py

Standalone entry point for training LightweightAuraViT (LAURA_SMALL) on
the MMOTU-ex ovarian tumour segmentation dataset.

Usage (from project root):

    python segmentation_module/train_segmentation.py

All arguments have sensible defaults for the LAURA_SMALL / 100-epoch run
agreed in the session. Override individual args as needed:

    # Ablation: disable overlay inpainting
    python segmentation_module/train_segmentation.py --no_inpainting

    # Use plateau scheduler instead of cosine warmup
    python segmentation_module/train_segmentation.py --scheduler_type plateau

    # Resume from a checkpoint (loads model weights + optimizer + scheduler)
    python segmentation_module/train_segmentation.py --resume results/checkpoints/segmentation/laura_small_best.pt

Output layout (all relative to project root):
    results/
        checkpoints/segmentation/
            laura_small_best.pt          ← best checkpoint by val_dice
        logs/segmentation/
            laura_small.log              ← full training log (DEBUG+)
            laura_small_history.csv      ← per-epoch metrics table
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

# ---------------------------------------------------------------------------
# Make segmentation_module packages importable when run from project root
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent           # segmentation_module/
_PROJECT_ROOT = _HERE.parent                       # f:\GitHub repos\MMOTU-ex\

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from segmentation.models.auravit_config import LAURA_BASE, LAURA_SMALL, LAURA_TINY
from segmentation.models.lightweight_auravit import LightweightAuraViT
from segmentation.dataset import get_segmentation_dataloaders
from segmentation.trainer import SegmentationTrainer, setup_segmentation_logger


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LightweightAuraViT for MMOTU segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument(
        "--splits", default="results/splits.csv",
        help="Path to splits CSV, relative to project root.",
    )
    p.add_argument(
        "--checkpoint_dir", default="results/checkpoints/segmentation",
        help="Directory for best-model checkpoint files.",
    )
    p.add_argument(
        "--log_dir", default="results/logs/segmentation",
        help="Directory for .log and _history.csv files.",
    )
    p.add_argument(
        "--run_name", default="laura_run",
        help="Stem used for checkpoint and log filenames.",
    )
    p.add_argument(
        "--architecture", default="small",
        choices=["base", "small", "tiny"],
        help="LAURA variant to train.",
    )
    p.add_argument(
        "--resume", default=None,
        help="Path to an existing checkpoint (.pt) to resume training from.",
    )

    # Training schedule
    p.add_argument("--epochs", type=int, default=100,
                   help="Total number of training epochs.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers. Use 0 on Windows to avoid multiprocessing issues.")

    # Optimiser
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Peak learning rate (after warmup if cosine_warmup).")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_value", type=float, default=1.0,
                   help="Gradient clipping norm (clip_grad_norm_).")

    # LR scheduler
    p.add_argument(
        "--scheduler_type", default="cosine_warmup",
        choices=["cosine_warmup", "cosine", "plateau", "none"],
        help=(
            "cosine_warmup: linear warmup then cosine decay (recommended). "
            "cosine: cosine decay from epoch 0. "
            "plateau: ReduceLROnPlateau on val_dice. "
            "none: constant LR."
        ),
    )
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="Linear warmup length (cosine_warmup only).")
    p.add_argument("--eta_min", type=float, default=1e-6,
                   help="Minimum LR at end of cosine decay.")
    p.add_argument("--lr_patience", type=int, default=10,
                   help="Epochs without improvement before LR halved (plateau only).")

    # Loss
    p.add_argument("--dice_smooth", type=float, default=1.0)
    p.add_argument("--bce_weight", type=float, default=0.5,
                   help="Weight of BCE term in DiceBCE loss (0=Dice only, 1=BCE only).")

    # Data preprocessing
    p.add_argument(
        "--no_inpainting", action="store_true",
        help="Disable caliper overlay inpainting (ablation flag).",
    )

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Seed for reproducibility (weight init, data shuffling)
    torch.manual_seed(args.seed)

    # Device: GPU if available, CPU otherwise (CPU-only dev machine is fine
    # for verifying the pipeline; GPU recommended for the full 100-epoch run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve all paths relative to project root so the script can be run
    # from any working directory.
    splits_csv    = str(_PROJECT_ROOT / args.splits)
    checkpoint_dir = str(_PROJECT_ROOT / args.checkpoint_dir)
    log_dir        = str(_PROJECT_ROOT / args.log_dir)

    # -----------------------------------------------------------------------
    # Logger — creates {log_dir}/laura_small.log before any other output
    # -----------------------------------------------------------------------
    logger = setup_segmentation_logger(log_dir, args.run_name)
    logger.info("=" * 70)
    logger.info(f"MMOTU Segmentation Training — LightweightAuraViT (LAURA_{args.architecture.upper()})")
    logger.info("=" * 70)
    logger.info(f"Device        : {device}")
    logger.info(f"Seed          : {args.seed}")
    logger.info(f"Epochs        : {args.epochs}")
    logger.info(f"Batch size    : {args.batch_size}")
    logger.info(f"LR            : {args.lr:.2e}")
    logger.info(f"Scheduler     : {args.scheduler_type}"
                + (f" (warmup={args.warmup_epochs}ep, eta_min={args.eta_min:.0e})"
                   if args.scheduler_type in ("cosine_warmup", "cosine") else ""))
    logger.info(f"Inpainting    : {'OFF (ablation)' if args.no_inpainting else 'ON'}")
    logger.info(f"Splits CSV    : {splits_csv}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Log dir       : {log_dir}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    arch_map = {
        "base": LAURA_BASE,
        "small": LAURA_SMALL,
        "tiny": LAURA_TINY,
    }
    model_cfg = arch_map[args.architecture]
    
    model = LightweightAuraViT(model_cfg)
    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model         : LAURA_{args.architecture.upper()}  "
        f"total={n_params_total/1e6:.2f}M  trainable={n_params_train/1e6:.2f}M"
    )

    # -----------------------------------------------------------------------
    # Resume from checkpoint (optional)
    # -----------------------------------------------------------------------
    start_epoch_offset = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = _PROJECT_ROOT / args.resume
        logger.info(f"Resuming from : {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        start_epoch_offset = ckpt.get("epoch", 0)
        logger.info(f"  Loaded weights from epoch {start_epoch_offset}, "
                    f"best_val_dice was {ckpt.get('best_val_dice', 'N/A'):.4f}")

    # -----------------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------------
    logger.info("Building dataloaders ...")
    train_loader, val_loader, test_loader = get_segmentation_dataloaders(
        splits_csv=splits_csv,
        image_size=256,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        apply_inpainting=not args.no_inpainting,
    )
    logger.info(
        f"  train={len(train_loader.dataset)} images "
        f"({len(train_loader)} batches, drop_last=True)  "
        f"val={len(val_loader.dataset)} images  "
        f"test={len(test_loader.dataset)} images"
    )

    # -----------------------------------------------------------------------
    # Config namespace passed to trainer
    # -----------------------------------------------------------------------
    config = SimpleNamespace(
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_value=args.clip_value,
        scheduler_type=args.scheduler_type,
        warmup_epochs=args.warmup_epochs,
        eta_min=args.eta_min,
        lr_patience=args.lr_patience,
        dice_smooth=args.dice_smooth,
        bce_weight=args.bce_weight,
    )

    # -----------------------------------------------------------------------
    # Trainer and training run
    # -----------------------------------------------------------------------
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        run_name=args.run_name,
        log_dir=log_dir,
    )

    # Restore optimizer + scheduler state if resuming
    if args.resume and "optimizer_state_dict" in ckpt:
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.best_val_dice = ckpt.get("best_val_dice", -1.0)
        trainer.best_epoch = start_epoch_offset
        logger.info("  Optimizer state restored.")

    wall_start = time.time()
    best_ckpt = trainer.train(num_epochs=args.epochs)
    wall_elapsed = time.time() - wall_start

    logger.info(
        f"Total wall time: {wall_elapsed / 3600:.2f}h  "
        f"({wall_elapsed:.0f}s)"
    )
    logger.info(f"Best checkpoint : {best_ckpt}")
    logger.info(
        f"CSV history     : "
        f"{str(_PROJECT_ROOT / args.log_dir / args.run_name)}_history.csv"
    )


if __name__ == "__main__":
    main()

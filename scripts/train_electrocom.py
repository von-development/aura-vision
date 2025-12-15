#!/usr/bin/env python3
"""
Train YOLO11n on ElectroCom-61 dataset.

This script trains a YOLO11n model on the full ElectroCom-61 dataset
(61 electronics component classes) with hyperparameters optimized for
FINE-TUNING from COCO pretrained weights.

Key optimizations:
- Lower learning rate (0.001 vs 0.01) to preserve pretrained features
- AdamW optimizer for better fine-tuning convergence
- Disabled vertical flip (electronics don't appear upside down)
- Extended warmup and patience for 61 classes

Usage:
    python scripts/train_electrocom.py --debug    # Quick validation (5 epochs)
    python scripts/train_electrocom.py            # Full training (100 epochs)
    python scripts/train_electrocom.py --device 0 # Train on GPU
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO


# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "dataset"
DATA_YAML = DATASET_PATH / "data.yaml"
YOLO_WEIGHTS = REPO_ROOT / "yolo" / "yolo11n.pt"
OUTPUT_DIR = REPO_ROOT / "runs"


@dataclass
class TrainingConfig:
    """
    Training configuration for ElectroCom-61 fine-tuning.
    
    Hyperparameters specifically optimized for:
    - YOLO11n (nano, 2.6M params)
    - Small dataset (~2000 images)
    - 61 classes (electronics components)
    - Fine-tuning from COCO pretrained weights
    
    Key differences from default YOLO training:
    - lr0=0.001 (not 0.01) - prevents "catastrophic forgetting"
    - AdamW optimizer - better for fine-tuning
    - flipud=0 - electronics don't appear upside down
    - Longer warmup and patience for many classes
    """

    # Model and data
    model_path: Path = YOLO_WEIGHTS
    data_yaml: Path = DATA_YAML

    # === Core Training Parameters ===
    epochs: int = 100           # Standard for small dataset, early stopping will kick in
    imgsz: int = 640            # Standard YOLO size, good for small components
    batch: int = 8              # Reduced for CPU stability & better gradients
    device: str = "cpu"
    fraction: float = 1.0       # Use 100% of data
    
    # === Learning Rate Schedule (CRITICAL for fine-tuning) ===
    # Using lower LR to preserve pretrained COCO features
    lr0: float = 0.001          # Initial LR (10x lower than default 0.01)
    lrf: float = 0.01           # Final LR = lr0 * lrf = 0.00001
    warmup_epochs: float = 5.0  # Longer warmup for 61 classes
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # === Optimizer ===
    # AdamW is better for fine-tuning (adaptive LR per parameter)
    optimizer: str = "AdamW"    # Changed from SGD for better fine-tuning
    momentum: float = 0.937     # Only used if optimizer=SGD
    weight_decay: float = 0.001 # Slight increase for regularization
    
    # === Augmentation (tuned for electronics components) ===
    hsv_h: float = 0.015        # Hue - minimal, electronics have specific colors
    hsv_s: float = 0.5          # Saturation - reduced from 0.7
    hsv_v: float = 0.4          # Value - good for lighting variation
    degrees: float = 15.0       # Rotation - increased, components at various angles
    translate: float = 0.2      # Translation - increased, components move in frame
    scale: float = 0.5          # Scale - good variation
    shear: float = 2.0          # Shear - appropriate
    flipud: float = 0.0         # DISABLED - electronics don't appear upside down
    fliplr: float = 0.5         # Horizontal flip OK
    mosaic: float = 1.0         # Essential for small dataset
    mixup: float = 0.15         # Slight increase for regularization
    copy_paste: float = 0.1     # Good for multi-object scenes
    
    # === Training Control ===
    close_mosaic: int = 15      # Disable mosaic in last 15 epochs for cleaner convergence
    patience: int = 25          # More patience for 61 classes
    save_period: int = 10       # Save checkpoint every 10 epochs
    workers: int = 0            # 0 for Windows compatibility (avoids multiprocessing issues)
    
    # === Layer Freezing (optional) ===
    # Freeze backbone to preserve pretrained features
    # freeze: int = 10          # Uncomment to freeze first 10 layers
    
    # Output
    project: Path = OUTPUT_DIR
    name: str = field(default_factory=lambda: f"electrocom61_{datetime.now().strftime('%m%d_%H%M')}")
    
    def to_train_args(self) -> dict[str, Any]:
        """Convert to dictionary for YOLO train()."""
        return {
            "data": str(self.data_yaml),
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device,
            "fraction": self.fraction,
            # Learning rate
            "lr0": self.lr0,
            "lrf": self.lrf,
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
            # Optimizer
            "optimizer": self.optimizer,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            # Augmentation
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            # Training control
            "close_mosaic": self.close_mosaic,
            "patience": self.patience,
            "save_period": self.save_period,
            "workers": self.workers,
            # Output
            "project": str(self.project),
            "name": self.name,
            # Extra
            "exist_ok": True,
            "pretrained": True,
            "verbose": True,
        }


def create_debug_config() -> TrainingConfig:
    """
    Create configuration for quick debug run.
    
    Purpose: Verify paths, data loading, and basic training works.
    Duration: ~10-15 minutes on CPU
    """
    return TrainingConfig(
        epochs=5,
        imgsz=640,
        batch=8,
        device="cpu",
        fraction=0.3,           # Only 30% of data
        warmup_epochs=2.0,      # Shorter warmup for debug
        patience=3,
        save_period=1,
        close_mosaic=2,
        name=f"debug_{datetime.now().strftime('%m%d_%H%M')}",
        # Reduced augmentation for faster debug
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
    )


def create_full_config(device: str = "cpu") -> TrainingConfig:
    """
    Create configuration for full production training.
    
    Purpose: Train high-quality model for deployment.
    Duration: ~4-6 hours on CPU, ~1-2 hours on GPU
    
    Expected results:
    - mAP50: 55-70%
    - mAP50-95: 35-50%
    - Precision: 65-80%
    - Recall: 55-70%
    """
    # Adjust batch size based on device
    batch = 8 if device == "cpu" else 16
    
    return TrainingConfig(
        epochs=100,
        imgsz=640,
        batch=batch,
        device=device,
        fraction=1.0,
        patience=25,
        save_period=10,
        name=f"electrocom61_{datetime.now().strftime('%m%d_%H%M')}",
    )


def validate_paths() -> bool:
    """Validate that all required paths exist."""
    errors = []
    
    if not YOLO_WEIGHTS.exists():
        errors.append(f"Model weights not found: {YOLO_WEIGHTS}")
    
    if not DATA_YAML.exists():
        errors.append(f"data.yaml not found: {DATA_YAML}")
    
    if not DATASET_PATH.exists():
        errors.append(f"Dataset folder not found: {DATASET_PATH}")
    
    # Check for train/valid splits
    for split in ["train", "valid"]:
        img_dir = DATASET_PATH / split / "images"
        lbl_dir = DATASET_PATH / split / "labels"
        
        if not img_dir.exists():
            errors.append(f"{split} images not found: {img_dir}")
        if not lbl_dir.exists():
            errors.append(f"{split} labels not found: {lbl_dir}")
    
    if errors:
        print("[ERROR] Validation failed:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    # Count images
    train_imgs = len(list((DATASET_PATH / "train" / "images").glob("*")))
    valid_imgs = len(list((DATASET_PATH / "valid" / "images").glob("*")))
    
    print(f"[OK] Dataset validated:")
    print(f"     Train: {train_imgs} images")
    print(f"     Valid: {valid_imgs} images")
    
    return True


def print_config(config: TrainingConfig, mode: str) -> None:
    """Print training configuration summary with justifications."""
    print("\n" + "=" * 70)
    print(f"YOLO11n FINE-TUNING CONFIGURATION ({mode.upper()})")
    print("=" * 70)
    print(f"Model:         {config.model_path.name} (2.6M params, pretrained on COCO)")
    print(f"Dataset:       {config.data_yaml}")
    print(f"Classes:       61 (ElectroCom electronics components)")
    print("-" * 70)
    
    print("\n[CORE TRAINING]")
    print(f"  epochs:        {config.epochs:>6}   | Early stopping will prevent overfitting")
    print(f"  imgsz:         {config.imgsz:>6}   | Standard size, good for small components")
    print(f"  batch:         {config.batch:>6}   | Small batch for CPU stability")
    print(f"  device:        {config.device:>6}")
    print(f"  data fraction: {config.fraction * 100:>5.0f}%")
    
    print("\n[LEARNING RATE] (Optimized for fine-tuning)")
    print(f"  lr0:           {config.lr0:>6}   | 10x lower than default (preserves pretrained features)")
    print(f"  lrf:           {config.lrf:>6}   | Final LR = {config.lr0 * config.lrf}")
    print(f"  warmup:        {config.warmup_epochs:>6}   | Longer warmup for 61 classes")
    print(f"  optimizer:     {config.optimizer:>6}   | AdamW better for fine-tuning")
    
    print("\n[AUGMENTATION] (Tuned for electronics)")
    print(f"  mosaic:        {config.mosaic:>6}   | Essential for small dataset")
    print(f"  mixup:         {config.mixup:>6}   | Regularization")
    print(f"  flipud:        {config.flipud:>6}   | DISABLED - electronics don't flip vertically")
    print(f"  degrees:       {config.degrees:>6}   | Rotation range")
    
    print("\n[TRAINING CONTROL]")
    print(f"  patience:      {config.patience:>6}   | Early stopping epochs")
    print(f"  close_mosaic:  {config.close_mosaic:>6}   | Disable mosaic in last N epochs")
    print(f"  save_period:   {config.save_period:>6}   | Checkpoint frequency")
    
    print("\n[OUTPUT]")
    print(f"  {config.project / config.name}")
    print("=" * 70 + "\n")


def estimate_duration(config: TrainingConfig) -> str:
    """Estimate training duration based on config."""
    if config.device == "cpu":
        # ~3 min per epoch on CPU with batch=8
        mins_per_epoch = 3.0 * (1.0 / config.fraction)
    else:
        # ~0.5 min per epoch on GPU
        mins_per_epoch = 0.5 * (1.0 / config.fraction)
    
    total_mins = mins_per_epoch * config.epochs
    hours = total_mins / 60
    
    if hours < 1:
        return f"~{int(total_mins)} minutes"
    else:
        return f"~{hours:.1f} hours"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train YOLO11n on ElectroCom-61 dataset (fine-tuning optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_electrocom.py --debug    # Quick validation (5 epochs, ~15 min)
  python scripts/train_electrocom.py            # Full training (100 epochs, ~5 hours)
  python scripts/train_electrocom.py --device 0 # Train on GPU (~1-2 hours)

Key Optimizations:
  - Lower learning rate (0.001) for fine-tuning
  - AdamW optimizer for better convergence
  - Disabled vertical flip (electronics don't flip)
  - Extended warmup and patience for 61 classes
        """,
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Quick debug run (5 epochs, 30%% data)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: 'cpu' or GPU index like '0'",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    print("\n" + "=" * 70)
    print("VALIDATING DATASET")
    print("=" * 70)
    if not validate_paths():
        return 1
    
    # Create configuration
    if args.debug:
        config = create_debug_config()
        mode = "debug"
    else:
        config = create_full_config(args.device)
        mode = "full"
    
    # Override with CLI arguments
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch is not None:
        config.batch = args.batch
    if args.imgsz is not None:
        config.imgsz = args.imgsz
    if args.device:
        config.device = args.device
    
    print_config(config, mode)
    
    # Estimate duration
    duration = estimate_duration(config)
    print(f"Estimated duration: {duration}")
    print()
    
    # Confirm before long training
    if not args.debug and config.epochs >= 10 and not args.yes:
        try:
            response = input("Start training? [Y/n]: ").strip().lower()
            if response == "n":
                print("Aborted.")
                return 0
        except EOFError:
            pass  # Non-interactive mode
    
    # Load model
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    
    if args.resume:
        model = YOLO(str(args.resume))
        print(f"Resuming from: {args.resume}")
    else:
        model = YOLO(str(config.model_path))
        print(f"Starting from: {config.model_path}")
        print("Pretrained on COCO (80 classes) - will adapt to 61 ElectroCom classes")
    
    # Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    
    try:
        results = model.train(**config.to_train_args())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training stopped by user.")
        print("You can resume with: --resume runs/<run_name>/weights/last.pt")
        return 1
    
    # Training complete
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    weights_dir = config.project / config.name / "weights"
    best_pt = weights_dir / "best.pt"
    
    if best_pt.exists():
        print(f"\nBest weights: {best_pt}")
        print(f"Last weights: {weights_dir / 'last.pt'}")
        
        # Copy best weights to yolo/ folder
        import shutil
        dst = REPO_ROOT / "yolo" / "electrocom61_best.pt"
        shutil.copy2(best_pt, dst)
        print(f"\n[OK] Copied best weights to: {dst}")
        print("\nTo use the trained model:")
        print(f"  aura webcam --model yolo/electrocom61_best.pt")
        print(f"  aura webcam --model yolo/electrocom61_best.pt --depth")
    else:
        print(f"\n[WARNING] Best weights not found at {best_pt}")
    
    return 0


if __name__ == "__main__":
    exit(main())

"""
RIVA Part B - Configuration Module for Swin-RTDETR
RT-DETR with Swin Transformer backbone for improved cell detection.

RT-DETR advantages over YOLO:
  - End-to-end detection (no NMS post-processing)
  - Transformer encoder-decoder → better global context
  - Hybrid encoder fuses multi-scale features more effectively
  - Better at detecting small/overlapping objects
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os
import random
import numpy as np


@dataclass
class ModelConfig:
    """Model selection configuration."""
    # RT-DETR base model (used as architecture template)
    primary_model: str = 'rtdetr-l.pt'
    # Swin variant from timm
    swin_variant: str = 'swin_base_patch4_window12_384_in22k'
    swin_pretrained: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters - OPTIMIZED FOR Swin-RTDETR.

    Key differences from Swin-YOLO:
      - RT-DETR uses its own transformer decoder (no NMS needed)
      - Lower LR since both backbone AND decoder are transformers
      - Different loss weights (Hungarian matching based)
      - Larger effective batch via gradient accumulation
    """

    # === IMAGE AND BATCH ===
    imgsz: int = 1280
    epochs: int = 150
    batch: int = 2           # RT-DETR + Swin is memory-heavy
    workers: int = 8
    patience: int = 50

    # === OPTIMIZER - Tuned for dual-transformer architecture ===
    optimizer: str = 'AdamW'
    lr0: float = 0.0001      # Even lower LR (two transformers in pipeline)
    lrf: float = 0.01        # Cosine decay to lr0 * 0.01
    weight_decay: float = 0.05  # Standard transformer weight decay
    momentum: float = 0.937

    # === SCHEDULER - Longer warmup for transformers ===
    cos_lr: bool = True
    warmup_epochs: int = 10   # Transformers benefit from longer warmup
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.0  # RT-DETR default: no bias LR warmup

    # === AUGMENTATION ===
    hsv_h: float = 0.005     # Minimal hue (staining is meaningful)
    hsv_s: float = 0.4       # Moderate saturation
    hsv_v: float = 0.3       # Moderate brightness

    # Geometric augmentations
    degrees: float = 15.0     # Cells can be any orientation
    translate: float = 0.15
    scale: float = 0.4        # More scale variation for detection
    shear: float = 2.0
    perspective: float = 0.0
    flipud: float = 0.5       # Cells are rotation-invariant
    fliplr: float = 0.5
    bgr: float = 0.0

    # Mosaic & mixing
    mosaic: float = 0.8       # Mosaic helps with dense small objects
    mixup: float = 0.15       # Mild mixup
    copy_paste: float = 0.15  # Copy-paste for object detection
    close_mosaic: int = 20    # Disable mosaic in last 20 epochs
    auto_augment: Optional[str] = None
    erasing: float = 0.1      # Random erasing
    crop_fraction: float = 1.0

    # === REGULARIZATION ===
    label_smoothing: float = 0.0
    dropout: float = 0.0
    nbs: int = 64

    # === DETECTION SETTINGS ===
    single_cls: bool = True
    overlap_mask: bool = True
    mask_ratio: int = 4

    # === VALIDATION ===
    val: bool = True
    save_period: int = 10
    plots: bool = True
    save: bool = True
    cache: bool = False

    # === REPRODUCIBILITY ===
    deterministic: bool = True
    seed: int = 42
    device: int = 0
    verbose: bool = True


@dataclass
class InferenceConfig:
    """Inference and ensemble configuration.

    RT-DETR inference differences:
      - No NMS needed (end-to-end)
      - Generally higher confidence predictions
      - TTA still beneficial for multi-scale
    """
    # Multi-scale inference
    scales: List[int] = field(default_factory=lambda: [1024, 1280, 1536])

    # Detection thresholds - RT-DETR outputs cleaner predictions
    conf_threshold: float = 0.001   # Can be slightly higher than YOLO
    iou_threshold: float = 0.3
    max_detections: int = 2000

    # Test-time augmentation
    augment: bool = True

    # Weighted Boxes Fusion parameters
    wbf_iou_threshold: float = 0.4
    wbf_skip_threshold: float = 0.001
    wbf_conf_type: str = 'avg'

    # Post-processing thresholds
    final_thresholds: List[float] = field(
        default_factory=lambda: [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15]
    )


@dataclass
class DataConfig:
    """Dataset paths and fold configuration."""
    root_dir: str = "riva-partb-dataset"
    work_dir: str = "swin-rtdetr-exp"
    n_folds: int = 5
    seed: int = 42

    @property
    def img_dir(self) -> str:
        return os.path.join(self.root_dir, "images")

    @property
    def ann_dir(self) -> str:
        return os.path.join(self.root_dir, "annotations")

    @property
    def train_csv(self) -> str:
        return os.path.join(self.ann_dir, "train.csv")

    @property
    def val_csv(self) -> str:
        return os.path.join(self.ann_dir, "val.csv")


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def print_summary(self):
        """Print configuration summary."""
        print("=" * 80)
        print("RIVA PIPELINE - Swin-RTDETR (Dual Transformer Architecture)")
        print("=" * 80)
        print(f"\nDetector: RT-DETR (end-to-end, no NMS)")
        print(f"Backbone: Swin Transformer ({self.model.swin_variant})")
        print(f"Image Size: {self.training.imgsz}px")
        print(f"Epochs: {self.training.epochs}")
        print(f"Batch Size: {self.training.batch}")
        print(f"Initial LR: {self.training.lr0}")
        print(f"Weight Decay: {self.training.weight_decay}")
        print(f"Warmup Epochs: {self.training.warmup_epochs}")
        print(f"K-Folds: {self.data.n_folds}")

        print(f"\nKEY ADVANTAGES OF SWIN-RTDETR:")
        print(f"  1. End-to-end detection (no NMS post-processing)")
        print(f"  2. Swin backbone → strong multi-scale features")
        print(f"  3. RT-DETR decoder → global context via cross-attention")
        print(f"  4. Better small/overlapping object detection")
        print(f"  5. Cleaner confidence scores")

        print(f"\nTRAINING SETTINGS:")
        print(f"  LR: {self.training.lr0} (very low for dual-transformer)")
        print(f"  Weight decay: {self.training.weight_decay}")
        print(f"  Warmup: {self.training.warmup_epochs} epochs")
        print(f"  Mosaic: {self.training.mosaic}")
        print(f"  Mixup: {self.training.mixup}")
        print(f"  Copy-paste: {self.training.copy_paste}")

        print(f"\nINFERENCE SETTINGS:")
        print(f"  Multi-scale: {self.inference.scales}")
        print(f"  Conf threshold: {self.inference.conf_threshold}")
        print(f"  Max detections: {self.inference.max_detections}")
        print(f"  TTA: {'ENABLED' if self.inference.augment else 'DISABLED'}")
        print("=" * 80)


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    return PipelineConfig()


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

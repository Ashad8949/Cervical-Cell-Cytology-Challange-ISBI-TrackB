"""
RIVA Part B - IMPROVED Configuration Module
Based on successful YOLOv8x settings with optimizations for YOLOv11x
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os
import random
import numpy as np


@dataclass
class ModelConfig:
    """Model selection configuration."""
    models_to_try: List[str] = field(default_factory=lambda: ['yolo11x.pt', 'yolo8x.pt'])
    primary_model: str = 'yolo11x.pt'


@dataclass
class TrainingConfig:
    """Training hyperparameters - OPTIMIZED FROM YOLOv8x SUCCESS."""
    
    # === CRITICAL CHANGES FROM YOLOv8x ===
    # Image size: Match YOLOv8x's successful 1280px (not 1920px)
    imgsz: int = 1280
    
    # Epochs: Moderate length like YOLOv8x (not excessive 300)
    epochs: int = 150
    
    # Batch: Keep reasonable for 1280px images
    batch: int = 4
    workers: int = 8
    patience: int = 50
    
    # === OPTIMIZER - Match YOLOv8x's working settings ===
    optimizer: str = 'AdamW'
    lr0: float = 0.0005  # YOLOv8x value (was 0.001)
    lrf: float = 0.01    # Keep this - works well
    weight_decay: float = 0.0005  # YOLOv8x value
    momentum: float = 0.937
    
    # === SCHEDULER - Keep YOLOv8x's warmup ===
    cos_lr: bool = True
    warmup_epochs: int = 5
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # === AUGMENTATION - KEY INSIGHT: ENABLE MOSAIC ===
    # YOLOv8x uses mosaic=0.3 successfully - medical images CAN handle it
    hsv_h: float = 0.01   # Keep conservative
    hsv_s: float = 0.4    # YOLOv8x value (increase from 0.3)
    hsv_v: float = 0.3    # Keep YOLOv8x value
    
    # Geometric augmentations - more conservative like YOLOv8x
    degrees: float = 10.0  # Much less rotation (was 90!)
    translate: float = 0.1  # YOLOv8x value
    scale: float = 0.3     # YOLOv8x value (increase from 0.2)
    shear: float = 2.0     # YOLOv8x value
    perspective: float = 0.0  # Keep disabled
    flipud: float = 0.0    # YOLOv8x: no vertical flip
    fliplr: float = 0.5    # YOLOv8x: horizontal flip OK
    bgr: float = 0.0
    
    # === CRITICAL: ENABLE MOSAIC (YOLOv8x uses 0.3) ===
    mosaic: float = 0.3    # Re-enable! (was 0.0)
    mixup: float = 0.0     # Keep disabled
    copy_paste: float = 0.0  # Keep disabled
    close_mosaic: int = 20  # YOLOv8x value - disable mosaic in last 20 epochs
    auto_augment: Optional[str] = None
    erasing: float = 0.0
    crop_fraction: float = 1.0
    
    # === REGULARIZATION - Keep minimal ===
    label_smoothing: float = 0.0
    dropout: float = 0.0
    nbs: int = 64
    
    # === LOSS WEIGHTS - Standard YOLO ===
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    
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
    """Inference and ensemble configuration."""
    # Multi-scale inference - adjust for 1280 base size
    scales: List[int] = field(default_factory=lambda: [1024, 1280, 1536])
    
    # Detection thresholds - match YOLOv8x's ultra-low threshold
    conf_threshold: float = 0.0001  # YOLOv8x value (was 0.05)
    iou_threshold: float = 0.3
    max_detections: int = 2000  # YOLOv8x value (was 1000)
    
    # Test-time augmentation
    augment: bool = True
    
    # Weighted Boxes Fusion parameters
    wbf_iou_threshold: float = 0.4
    wbf_skip_threshold: float = 0.0001  # Match YOLOv8x ultra-low threshold
    wbf_conf_type: str = 'avg'
    
    # Post-processing thresholds
    final_thresholds: List[float] = field(
        default_factory=lambda: [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15]
    )


@dataclass
class DataConfig:
    """Dataset paths and fold configuration."""
    root_dir: str = "riva-dataset-partb"
    work_dir: str = "yolov11x"
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
        print("RIVA IMPROVED PIPELINE - Based on YOLOv8x Success")
        print("=" * 80)
        print(f"\nModel: {self.model.primary_model}")
        print(f"Image Size: {self.training.imgsz}px (matched to YOLOv8x)")
        print(f"Epochs: {self.training.epochs} (reduced from 300)")
        print(f"Batch Size: {self.training.batch}")
        print(f"Initial LR: {self.training.lr0} (matched to YOLOv8x)")
        print(f"K-Folds: {self.data.n_folds}")
        
        print(f"\nðŸ”‘ KEY CHANGES FROM PREVIOUS CONFIG:")
        print(f"  âœ“ Image size: 1920 â†’ 1280px (match YOLOv8x)")
        print(f"  âœ“ Learning rate: 0.001 â†’ 0.0005 (match YOLOv8x)")
        print(f"  âœ“ Mosaic: ENABLED at 0.3 (was disabled!)")
        print(f"  âœ“ Rotation: Â±90Â° â†’ Â±10Â° (much more conservative)")
        print(f"  âœ“ Scale aug: 0.2 â†’ 0.3 (match YOLOv8x)")
        print(f"  âœ“ Epochs: 300 â†’ 150 (faster iteration)")
        print(f"  âœ“ Conf threshold: 0.05 â†’ 0.0001 (match YOLOv8x)")
        print(f"  âœ“ Max detections: 1000 â†’ 2000 (match YOLOv8x)")
        
        print(f"\nAugmentation Settings:")
        print(f"  - Mosaic: {self.training.mosaic} (ENABLED - close at epoch {self.training.close_mosaic})")
        print(f"  - Rotation: Â±{self.training.degrees}Â° (conservative)")
        print(f"  - Scale: {self.training.scale}")
        print(f"  - HSV: h={self.training.hsv_h}, s={self.training.hsv_s}, v={self.training.hsv_v}")
        print(f"  - Flips: LR={self.training.fliplr}, UD={self.training.flipud}")
        
        print(f"\nInference Settings:")
        print(f"  - Multi-scale: {self.inference.scales}")
        print(f"  - Conf threshold: {self.inference.conf_threshold}")
        print(f"  - Max detections: {self.inference.max_detections}")
        print(f"  - WBF skip threshold: {self.inference.wbf_skip_threshold}")
        print(f"  - TTA: {'ENABLED' if self.inference.augment else 'DISABLED'}")
        
        print("\nðŸ’¡ RATIONALE:")
        print("  YOLOv8x succeeded with mosaic augmentation, moderate image size,")
        print("  and conservative geometric augmentations. The key mistakes were:")
        print("  1) Disabling mosaic entirely")
        print("  2) Using excessive rotation (90Â° vs 10Â°)")
        print("  3) Using too large images (1920 vs 1280)")
        print("  4) Training too long (300 vs 150 epochs)")
        
        print("\nðŸ“Š Expected Improvements:")
        print("  - Faster training (150 vs 300 epochs)")
        print("  - Better generalization (mosaic augmentation)")
        print("  - More stable training (lower LR, moderate image size)")
        print("  - Higher recall (ultra-low confidence thresholds)")
        print("=" * 80)


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    return PipelineConfig()


def get_yolov8x_proven_config() -> PipelineConfig:
    """
    Get configuration that matches proven YOLOv8x settings.
    Use this as a safe baseline.
    """
    config = PipelineConfig()
    # Already set to YOLOv8x values by default
    return config


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
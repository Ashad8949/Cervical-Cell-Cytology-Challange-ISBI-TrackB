# Cervical Cell Cytology Detection - ISBI 2025 Track B (RIVA)

An advanced cervical cytology cell detection system for the **ISBI Challenge Track B**. This project implements multiple detection architectures — hybrid CNN-Transformer models, RT-DETR, Swin-based variants, and YOLO — with ensemble methods and test-time augmentation (TTA) for accurate cell localization in cytological images.

**Objective**: Detect and localize individual cells in cervical cytology images, predicting center coordinates (x, y) and bounding box dimensions (width, height).

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Pipeline Diagram](#pipeline-diagram)
- [Dataset Setup](#dataset-setup)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Generating Submissions](#generating-submissions)
  - [Alternative Models](#alternative-models)
- [Configurations](#configurations)
- [Key Design Decisions](#key-design-decisions)
- [Results](#results)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                                   │
│                   (Cervical Cytology Slide)                          │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Preprocessing   │  Stain normalization, CLAHE,
                    │  & Augmentation  │  geometric + color augmentations
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼────────┐    │    ┌─────────▼─────────┐
     │  CNN Backbone    │    │    │  ViT Backbone      │
     │  (ConvNeXt V2)   │    │    │  (PathDINO/Swin)   │
     │  Local Features  │    │    │  Global Features   │
     └────────┬────────┘    │    └─────────┬─────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Feature Fusion  │  Cross-Attention / Gating / Concat
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Feature Pyramid │  Multi-scale features
                    │  Network (FPN)   │  (4 levels)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Detection Head  │  DINO-DETR / Deformable DETR
                    │  (Transformer    │  Hungarian matching
                    │   Decoder)       │  300-500 object queries
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Post-Processing │  NMS / Soft-NMS / WBF
                    │  & TTA Ensemble  │  Multi-scale + flip + rotation
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Predictions    │  (x, y, width, height, conf)
                    └─────────────────┘
```

---

## Project Structure

```
├── main.py                          # CLI entry point (train / inference / submit)
├── requirements.txt                 # Python dependencies
├── configs/
│   ├── training_config.yaml         # Full production training config (200 epochs, 5-fold CV)
│   ├── run_config.yaml              # Standard training config (30 epochs)
│   ├── test_run_config.yaml         # Quick test config (1 epoch, small subset)
│   ├── model_config.yaml            # Model architecture config
│   └── inference_config.yaml        # Inference + TTA + ensemble config
│
├── data/
│   ├── dataset.py                   # RIVADataset - data loading & annotation parsing
│   ├── augmentations.py             # Train/Val/Test augmentation pipelines
│   ├── preprocessing.py             # Stain normalization, artifact removal, CLAHE
│   └── folds.py                     # Cross-validation fold generation
│
├── models/
│   ├── backbones.py                 # PathDINO, Swin, ConvNeXt V2, MaxViT, CaiT
│   ├── detection_heads.py           # DINO-DETR, Deformable DETR, Conditional DETR
│   ├── hybrid_models.py             # HybridCellDetector (main model), MultiScaleDetector
│   ├── losses.py                    # DETRLoss, FocalLoss, GIoU, CellDetectionLoss
│   └── ensemble.py                  # Model ensemble utilities
│
├── training/
│   ├── trainer.py                   # Training loop (AMP, gradient accumulation, checkpointing)
│   ├── callbacks.py                 # EarlyStopping, ModelCheckpoint, LR scheduler, TensorBoard
│   ├── curriculum.py                # Curriculum learning (progressive difficulty & resolution)
│   └── optimizers.py                # Optimizer & scheduler factories with layer-wise LR
│
├── inference/
│   ├── tta.py                       # Test-Time Augmentation (multi-scale, flip, rotation)
│   ├── ensemble_inference.py        # Multi-model ensemble with WBF
│   ├── postprocessing.py            # NMS, Soft-NMS, WBF, box format conversions
│   └── multi_scale.py               # Multi-scale inference utilities
│
├── scripts/
│   ├── train.py                     # Standalone training script with fold-based CV
│   ├── inference.py                 # Standalone inference script
│   ├── inference_tta.py             # Inference with TTA
│   ├── submit.py                    # Submission CSV generation
│   ├── convert_to_yolo.py           # Convert annotations to YOLO format
│   ├── ensemble_wbf.py              # Post-hoc WBF ensembling of CSVs
│   ├── generate_all_submissions.py  # Batch submission generation
│   └── train_yolo11.py              # YOLO11 training script
│
├── rtdetr/
│   ├── train_rtdetr.py              # RT-DETR training (Ultralytics)
│   ├── train_rtdetr_v2.py           # RT-DETR v2 training with optimized settings
│   └── inference_rtdetr.py          # RT-DETR inference
│
├── swin-rtdetr/
│   ├── config.py                    # Swin-RT-DETR configuration
│   ├── swin_backbone.py             # Swin Transformer backbone for RT-DETR
│   ├── training.py                  # Swin-RT-DETR training pipeline
│   ├── inference.py                 # Swin-RT-DETR inference
│   ├── run_inference.py             # Inference runner
│   └── data_preparation.py          # Data preparation for Swin-RT-DETR
│
├── swin-yolo/
│   ├── config.py                    # Swin-YOLO configuration
│   ├── swin_backbone.py             # Swin backbone for YOLO
│   ├── training.py                  # Swin-YOLO training
│   ├── inference.py                 # Swin-YOLO inference
│   └── data_peparation.py           # Data preparation
│
├── yolo11x/
│   ├── config.py                    # YOLO11-XLarge configuration
│   ├── training.py                  # YOLO11 training
│   ├── inference.py                 # YOLO11 inference
│   └── data_preparation.py          # Data preparation
│
├── utils/
│   ├── io_utils.py                  # Config I/O, checkpoint save/load, submission generation
│   ├── metrics.py                   # mAP, AP@IoU, precision/recall computation
│   ├── logging.py                   # Console/file logger, W&B logger, TensorBoard logger
│   └── visualization.py             # Box drawing, prediction visualization, training curves
│
├── yolo_dataset/
│   └── dataset.yaml                 # YOLO format dataset definition
│
├── riva-partb-dataset/              # ⚠️ NOT included — download separately (see below)
│   ├── annotations/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── train_sample.csv
│   │   └── sample_submission.csv
│   └── images/
│       ├── train/
│       ├── val/
│       └── test/
│
├── experiments/                     # ⚠️ NOT included — generated during training
└── submissions/                     # ⚠️ NOT included — generated during inference
```

---

## Model Architectures

### 1. HybridCellDetector (Primary Model)

The main model fuses **CNN** and **Vision Transformer** features for robust cell detection:

| Component | Architecture | Role |
|-----------|-------------|------|
| CNN Backbone | ConvNeXt V2 Base | Local feature extraction (edges, textures) |
| Transformer Backbone | PathDINO (ViT-B/14, DINOv2) | Global context & long-range dependencies |
| Feature Fusion | Cross-Attention / Gating | Combine local + global features |
| Neck | FPN (4 levels, 256 channels) | Multi-scale feature aggregation |
| Detection Head | DINO-DETR (6 enc + 6 dec layers) | Set prediction with Hungarian matching |

**Why hybrid?** Cervical cytology images have both fine-grained cell morphology (benefits from CNN) and complex spatial arrangements (benefits from Transformer attention).

### 2. RT-DETR (Alternative)

Real-Time Detection Transformer via Ultralytics — a production-ready DETR variant with:
- Deformable attention for efficiency
- 2-stage refinement
- IoU-aware classification scoring

### 3. Swin-RT-DETR (Alternative)

Replaces the RT-DETR backbone with **Swin Transformer** (hierarchical vision transformer):
- `swin_base_patch4_window12_384_in22k` pretrained weights
- Multi-scale outputs at strides 8, 16, 32
- RT-DETR hybrid encoder + transformer decoder preserved

### 4. YOLO11-XLarge (Alternative)

Ultralytics YOLO11 extra-large model for single-stage detection as a fast baseline.

---

## Pipeline Diagram

### Training Pipeline

```
Dataset (CSV + Images)
    │
    ├── Augmentation ──────────────────────────────┐
    │   ├── Geometric: flip, rotate, affine, crop  │
    │   ├── Color: CLAHE, sharpen, jitter          │
    │   ├── Medical: stain normalization, blur sim  │
    │   └── Strong: CoarseDropout, PixelDropout    │
    │                                               │
    ▼                                               │
 HybridCellDetector                                │
    │                                               │
    ├── DETRLoss                                    │
    │   ├── Hungarian Matching (optimal assignment) │
    │   ├── Focal Loss (classification, α=0.25, γ=2)│
    │   ├── L1 Loss (box regression)               │
    │   └── GIoU Loss (box quality)                │
    │                                               │
    ├── CellDetectionLoss (domain-specific)         │
    │   ├── Size constraints (cell size distribution)│
    │   ├── Density constraints (cells per image)   │
    │   └── Overlap constraints                     │
    │                                               │
    ├── AdamW Optimizer                             │
    │   ├── Backbone LR: 1e-5                      │
    │   ├── Detection Head LR: 1e-4                │
    │   └── Weight Decay: 1e-4                     │
    │                                               │
    ├── LR Schedule                                 │
    │   ├── Linear Warmup (3 epochs)               │
    │   └── Cosine Annealing w/ Warm Restarts      │
    │                                               │
    └── Training Features                           │
        ├── Mixed Precision (AMP)                   │
        ├── Gradient Accumulation (2 steps)         │
        ├── Gradient Clipping (0.1)                 │
        └── Curriculum Learning (optional)          │
```

### Inference Pipeline

```
Test Image
    │
    ▼
 Test-Time Augmentation (TTA)
    ├── Scales: [896, 1024, 1152, 1280]
    ├── Flips: [none, horizontal, vertical, diagonal]
    ├── Rotations: [0°, 90°, 180°, 270°]
    │
    ▼ (for each augmentation)
 Model Forward Pass → Reverse Transform → Collect Predictions
    │
    ▼
 Ensemble (Multiple Models)
    ├── Model 1: HybridCellDetector (fold 1)
    ├── Model 2: HybridCellDetector (fold 2)
    ├── ...up to 5 models
    │
    ▼
 Weighted Box Fusion (WBF)
    │
    ▼
 Post-Processing
    ├── NMS / Soft-NMS (IoU threshold)
    ├── Confidence filtering (threshold: 0.05)
    ├── Size filtering (min/max box size)
    └── Max detections cap (300)
    │
    ▼
 Submission CSV (image, x, y, w, h, conf, class)
```

---

## Dataset Setup

The dataset is **not included** in this repository. You must download it separately from the ISBI challenge.

### Required Directory: `riva-partb-dataset/`

Create the following structure in the project root:

```
riva-partb-dataset/
├── annotations/
│   ├── train.csv              # Training annotations
│   ├── val.csv                # Validation annotations
│   ├── train_sample.csv       # Small subset for quick testing
│   └── sample_submission.csv  # Submission template
└── images/
    ├── train/                 # Training images (.png)
    ├── val/                   # Validation images (.png)
    └── test/                  # Test images (.png)
```

### Annotation Format

CSV files use center-based bounding boxes:

```csv
image_filename,x,y,width,height,class_name,class
LSIL_45_3.png,1000.55,526.33,100,100,CELL,0
LSIL_45_3.png,842.12,433.78,100,100,CELL,0
...
```

| Column | Description |
|--------|-------------|
| `image_filename` | Image file name in `images/` directory |
| `x` | Bounding box center x-coordinate |
| `y` | Bounding box center y-coordinate |
| `width` | Bounding box width |
| `height` | Bounding box height |
| `class_name` | Class label (always `CELL`) |
| `class` | Class index (always `0`) |

### YOLO Format Dataset (Auto-generated)

To convert to YOLO format for YOLO11/Swin-YOLO training:

```bash
python scripts/convert_to_yolo.py
```

This creates `yolo_dataset/` with images, labels, and `dataset.yaml`.

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- CUDA 11.8+ / 12.x

### Setup

```bash
# Clone the repository
git clone https://github.com/Ashad8949/Cervical-Cell-Cytology-Challange-ISBI-TrackB.git
cd Cervical-Cell-Cytology-Challange-ISBI-TrackB

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch >= 2.0` | Deep learning framework |
| `torchvision >= 0.15` | Vision utilities |
| `timm >= 0.9` | Pretrained vision backbones (PathDINO, Swin, ConvNeXt) |
| `transformers >= 4.30` | HuggingFace transformer models |
| `albumentations >= 1.3` | Image augmentation pipeline |
| `opencv-python >= 4.7` | Image processing |
| `scipy >= 1.10` | Hungarian matching for DETR |
| `pandas >= 2.0` | Data handling |
| `wandb >= 0.15` | Experiment tracking (optional) |
| `matplotlib >= 3.7` | Visualization |
| `PyYAML >= 6.0` | Config parsing |

---

## Usage

### Quick Test Run (Verify Pipeline)

Run a single epoch on a small data subset to verify everything works:

```bash
python main.py train --config configs/test_run_config.yaml
```

Then test inference:

```bash
python main.py submit \
    --checkpoint experiments/riva_test_run_one_epoch/checkpoints/best.pth \
    --test-dir riva-partb-dataset/images/test \
    --output submission_test.csv
```

### Training

#### Standard Training (30 epochs)

```bash
python main.py train --config configs/run_config.yaml
```

#### Full Production Training (200 epochs, 5-fold CV)

```bash
python main.py train --config configs/training_config.yaml
```

#### Resume Training from Checkpoint

```bash
python main.py train --config configs/run_config.yaml \
    --resume experiments/riva_cell_detection_v2/checkpoints/checkpoint_epoch10.pth
```

#### RT-DETR Training

```bash
python rtdetr/train_rtdetr.py
# or the v2 variant:
python rtdetr/train_rtdetr_v2.py
```

#### YOLO11 Training

```bash
python scripts/train_yolo11.py
```

### Inference

#### Basic Inference

```bash
python main.py inference \
    --checkpoint experiments/riva_cell_detection_v2/checkpoints/best.pth \
    --test-dir riva-partb-dataset/images/test
```

#### Inference with Test-Time Augmentation

```bash
python main.py inference \
    --checkpoint experiments/riva_cell_detection_v2/checkpoints/best.pth \
    --test-dir riva-partb-dataset/images/test \
    --tta \
    --confidence 0.05
```

#### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | required | Path to model checkpoint (.pth) |
| `--test-dir` | required | Directory containing test images |
| `--tta` | off | Enable test-time augmentation |
| `--confidence` | 0.1 | Confidence threshold for detections |
| `--nms-threshold` | 0.5 | NMS IoU threshold |
| `--max-detections` | 300 | Maximum detections per image |

### Generating Submissions

```bash
python main.py submit \
    --checkpoint experiments/riva_cell_detection_v2/checkpoints/best.pth \
    --test-dir riva-partb-dataset/images/test \
    --output submission.csv \
    --tta \
    --confidence 0.05
```

#### Batch Submission Generation (Multiple Configs)

```bash
python scripts/generate_all_submissions.py
```

#### Ensemble Multiple Submission CSVs via WBF

```bash
python scripts/ensemble_wbf.py
```

---

## Configurations

All configs are in `configs/`. Key parameters:

### Training Config ([`training_config.yaml`](configs/training_config.yaml))

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 200 | Full training epochs |
| `image_size` | 1024 | Input resolution |
| `batch_size` | 2 | Batch size per GPU |
| `lr` | 1e-4 | Learning rate (detection head) |
| `backbone_lr` | 1e-5 | Learning rate (backbone) |
| `mixed_precision` | true | FP16 training |
| `accumulation_steps` | 2 | Gradient accumulation |
| `grad_clip_norm` | 0.1 | Gradient clipping |
| `warmup_epochs` | 3 | Linear LR warmup |
| `num_queries` | 300 | DETR object queries |
| `num_classes` | 1 | Single class (CELL) |

### Inference Config ([`inference_config.yaml`](configs/inference_config.yaml))

| Parameter | Value | Description |
|-----------|-------|-------------|
| `tta_scales` | [896, 1024, 1152, 1280] | TTA multi-scale |
| `tta_flips` | 4 variants | Horizontal, vertical, diagonal |
| `tta_rotations` | [0, 90, 180, 270] | Rotation augmentations |
| `ensemble_models` | 5 | Number of models in ensemble |
| `merge_method` | weighted_box_fusion | WBF for ensemble |
| `confidence_threshold` | 0.05 | Min detection confidence |
| `max_detections` | 300 | Max detections per image |

---

## Key Design Decisions

1. **Hybrid CNN + Transformer Backbone**: Combines ConvNeXt V2 (strong local features for cell morphology) with PathDINO ViT (global context for spatial relationships). This outperforms single-backbone approaches on cytology data.

2. **DINO-DETR Head**: Set-based detection avoids NMS during training, and Hungarian matching provides optimal box-to-GT assignment. DINO improvements (denoising, contrastive learning) improve convergence.

3. **Domain-Specific Losses**: Beyond standard DETR losses, `CellDetectionLoss` adds constraints on cell size distribution, density, and overlap — encoding prior knowledge about cytology images.

4. **Aggressive TTA**: 4 scales x 4 flips x 4 rotations = 64 augmented views per image at test time. Combined via Weighted Box Fusion for maximum recall.

5. **Multi-Model Ensemble**: 5 models from different folds / architectures fused via WBF to reduce variance and improve robustness.

6. **Curriculum Learning**: Starts training on easier images (fewer cells, larger cells) and progressively introduces harder cases, improving convergence on challenging dense regions.

7. **Layer-wise Learning Rates**: Backbone at 0.1x, transformer body at 1x, detection head at 5x — preserves pretrained features while rapidly adapting task-specific layers.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| mAP@0.50:0.95 | COCO-style mean Average Precision (primary metric) |
| mAP@0.50 | AP at IoU threshold 0.50 |
| mAP@0.75 | AP at IoU threshold 0.75 |
| Precision / Recall | Per IoU threshold |

---

## License

This project is developed for the ISBI Cervical Cell Cytology Challenge (Track B).

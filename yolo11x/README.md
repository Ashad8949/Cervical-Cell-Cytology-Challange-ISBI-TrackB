# YOLOv11x: YOLO11 Extra-Large for Cell Detection

Standard **YOLOv11x** (extra-large) model for cervical cell detection, with hyperparameters optimized from successful YOLOv8x runs. This serves as a strong CNN baseline without custom backbone modifications.

---

## Architecture

```
Input Image (1280×1280)
    │
    ▼
┌───────────────────────────────┐
│  YOLOv11x CNN Backbone        │  Pretrained on COCO
│  (Extra-Large variant)        │  Deep feature extraction
│                               │
│  P3 (stride 8)               │
│  P4 (stride 16)              │
│  P5 (stride 32)              │
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│  YOLO Neck (FPN + PANet)      │  SPPF + C2PSA modules
│  Multi-scale feature fusion   │  Top-down + bottom-up pathways
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│  YOLO Detection Head          │  Anchor-free detection
│  (3 detection scales)         │  Box + Cls + DFL losses
└───────────┬───────────────────┘
            │
            ▼
      Predictions → NMS → (x, y, w, h, conf)
```

---

## Files

| File | Description |
|------|-------------|
| `config.py` | All configuration dataclasses — model, training, inference, data |
| `training.py` | `ImprovedModelTrainer` + `ImprovedTrainingPipeline` — K-fold training |
| `inference.py` | Multi-scale TTA + WBF ensemble inference + submission generation |
| `data_preparation.py` | Dataset loading, stratified K-fold splits, YOLO-format labels |

---

## Training Configuration

Based on proven YOLOv8x settings:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | `yolo11x.pt` | Extra-large, pretrained on COCO |
| Image Size | 1280px | Optimal for ~100px cell boxes |
| Epochs | 150 | Moderate length |
| Batch Size | 4 | Higher than Swin variants (less VRAM) |
| Optimizer | AdamW | |
| Learning Rate | 5e-4 | Standard for CNN backbone |
| Weight Decay | 5e-4 | Standard CNN regularization |
| Warmup | 5 epochs | |
| Cosine LR | Yes | Decay to lr0 × 0.01 |
| Mosaic | 0.3 | Proven from YOLOv8x |
| Close Mosaic | 20 | Disable in last 20 epochs |
| Horizontal Flip | 0.5 | |
| Rotation | 10° | Conservative for medical images |
| Scale Variation | 0.3 | |
| K-Folds | 5 | Stratified by cell count |
| Early Stopping | 50 | Patience epochs |
| Loss Weights | box=7.5, cls=0.5, dfl=1.5 | Standard YOLO |

---

## Inference Configuration

| Parameter | Value |
|-----------|-------|
| Scales | [1024, 1280, 1536] |
| Confidence threshold | 0.0001 (ultra-low — WBF filters) |
| IoU threshold | 0.3 |
| Max detections | 2000 |
| WBF IoU | 0.4 |
| WBF skip threshold | 0.0001 |
| TTA | Multi-scale + horizontal flip |
| Final thresholds | [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15] |

---

## Usage

### 1. Prepare Data

```bash
cd yolo11x
python data_preparation.py --root_dir ../riva-partb-dataset --work_dir yolov11x-exp --n_folds 5
```

Creates stratified K-fold directory structure:
```
yolov11x-exp/
├── fold_0/
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   ├── labels/val/
│   └── data.yaml
├── fold_1/
│   └── ...
└── fold_info.csv
```

### 2. Train

```bash
cd yolo11x
python training.py
```

Trains all K folds sequentially. Each fold saves:
- `runs/fold_N/weights/best.pt` — best mAP checkpoint
- `runs/fold_N/weights/last.pt` — last epoch checkpoint

### 3. Inference

#### Single model
```bash
cd yolo11x
python inference.py --test_dir ../riva-partb-dataset/images/test --models best_f0.pt
```

#### Ensemble (all folds)
```bash
cd yolo11x
python inference.py --test_dir ../riva-partb-dataset/images/test --work_dir yolov11x-exp
```

Auto-discovers `best.pt` from each fold and runs ensemble with WBF. Generates submission CSVs at multiple confidence thresholds.

---

## Dependencies

Requires everything in the root `requirements.txt` plus:
- `ultralytics` — YOLO11 implementation
- `ensemble_boxes` — Weighted Box Fusion

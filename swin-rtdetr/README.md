# Swin-RT-DETR: Swin Transformer Backbone for RT-DETR

A dual-transformer cell detection architecture that replaces RT-DETR's default CNN backbone (HGNetV2/ResNet) with a **Swin Transformer**, while preserving RT-DETR's hybrid encoder and transformer decoder for end-to-end detection.

---

## Architecture

```
Input Image (1280×1280)
    │
    ▼
┌───────────────────────────────┐
│  Swin Transformer Backbone    │  swin_base_patch4_window12_384_in22k
│  (Pretrained, from timm)      │  Hierarchical multi-scale features
│                               │
│  Stage 1 → P3 (stride 8)     │  256 channels
│  Stage 2 → P4 (stride 16)    │  512 channels
│  Stage 3 → P5 (stride 32)    │  1024 channels
└───────────┬───────────────────┘
            │  Channel Adapters (1×1 Conv)
            ▼
┌───────────────────────────────┐
│  RT-DETR Hybrid Encoder       │  AIFI (Attention-based Intra-scale
│  (AIFI + CCFM)                │  Feature Interaction) +
│                               │  CCFM (Cross-scale Feature Merge)
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│  RT-DETR Transformer Decoder  │  End-to-end detection (NO NMS)
│  (Cross-attention queries)    │  Hungarian matching loss
└───────────┬───────────────────┘
            │
            ▼
      Predictions (x, y, w, h, conf)
```

**Key advantage**: Both the backbone and decoder are transformers, providing global context at every stage. RT-DETR's end-to-end design eliminates NMS post-processing.

---

## Files

| File | Description |
|------|-------------|
| `config.py` | All configuration dataclasses (model, training, inference, data, pipeline) |
| `swin_backbone.py` | `SwinRTDETRModel` — injects Swin into RT-DETR with channel adapters |
| `training.py` | `SwinRTDETRModelTrainer` — K-fold training pipeline |
| `inference.py` | Multi-scale TTA + WBF ensemble inference |
| `run_inference.py` | Standalone runner for fold-1, fold-2, and ensemble inference |
| `data_preparation.py` | Dataset loading, stratified K-fold splits, YOLO-format label generation |
| `apply_fixbox.py` | Force all predicted boxes to 100×100 (matching GT format) |

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Image Size | 1280px | High resolution for small cell detection |
| Epochs | 150 | Sufficient for dual-transformer convergence |
| Batch Size | 2 | Memory-heavy (Swin + RT-DETR decoder) |
| Optimizer | AdamW | Standard for transformers |
| Learning Rate | 1e-4 | Very low — two transformers in pipeline |
| Weight Decay | 0.05 | Standard transformer regularization |
| Warmup | 10 epochs | Longer warmup for transformer stability |
| Mosaic | 0.8 | Aggressive — helps dense small objects |
| Mixup | 0.15 | Mild mixing |
| Copy-paste | 0.15 | Object augmentation |
| Close Mosaic | 20 | Disable mosaic in last 20 epochs for clean convergence |
| K-Folds | 5 | Stratified cross-validation |

---

## Inference Configuration

| Parameter | Value |
|-----------|-------|
| Scales | [1024, 1280, 1536] |
| Confidence threshold | 0.001 (WBF handles filtering) |
| IoU threshold | 0.3 |
| Max detections | 2000 |
| WBF IoU | 0.4 |
| TTA | Multi-scale + horizontal flip |
| Final thresholds | [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15] |

---

## Usage

### 1. Prepare Data

```bash
cd swin-rtdetr
python data_preparation.py --root_dir ../riva-partb-dataset --work_dir swin-rtdetr-exp --n_folds 5
```

This creates stratified K-fold splits with YOLO-format labels:
```
swin-rtdetr-exp/
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
cd swin-rtdetr
python training.py
```

Trains all K folds sequentially. Each fold saves:
- `runs/fold_N/weights/best.pt`
- `runs/fold_N/weights/last.pt`

### 3. Inference

#### Single model
```bash
cd swin-rtdetr
python inference.py --test_dir ../riva-partb-dataset/images/test --models ../best_f1.pt
```

#### Ensemble (recommended)
```bash
cd swin-rtdetr
python run_inference.py
```

Runs 3 passes: fold 1 only, fold 2 only, and ensemble — generates multiple submission CSVs at different confidence thresholds.

### 4. Fix Box Size

```bash
cd swin-rtdetr
python apply_fixbox.py --box_size 100
```

Forces all predicted boxes to 100×100 pixels (matching RIVA GT format) for improved IoU at stricter thresholds.

---

## Dependencies

Requires everything in the root `requirements.txt` plus:
- `ultralytics` — RT-DETR implementation
- `timm` — Swin Transformer pretrained weights
- `ensemble_boxes` — Weighted Box Fusion

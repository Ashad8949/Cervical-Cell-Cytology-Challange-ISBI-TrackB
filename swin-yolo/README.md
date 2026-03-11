# Swin-YOLO: Swin Transformer Backbone for YOLOv11x

Replaces YOLOv11x's CNN backbone with a **Swin Transformer** while keeping YOLO's FPN/PANet neck and detection head. Combines the strong multi-scale features of Swin with YOLO's fast detection pipeline.

---

## Architecture

```
Input Image (1280×1280)
    │
    ▼
┌───────────────────────────────┐
│  Swin Transformer Backbone    │  swin_base_patch4_window12_384_in22k
│  (Pretrained, from timm)      │  Replaces YOLO CNN backbone (layers 0-8)
│                               │
│  Stage 1 → P3 (stride 8)     │  256 channels
│  Stage 2 → P4 (stride 16)    │  512 channels
│  Stage 3 → P5 (stride 32)    │  1024 channels
└───────────┬───────────────────┘
            │  Channel Adapters (1×1 Conv → 512ch)
            ▼
┌───────────────────────────────┐
│  YOLO Neck (FPN/PANet)        │  SPPF (layer 9) + C2PSA (layer 10)
│  Multi-scale feature fusion   │  Top-down + bottom-up pathways
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│  YOLO Detection Head          │  Anchor-free detection
│  (Standard YOLO head)         │  Box + Cls + DFL losses
└───────────┬───────────────────┘
            │
            ▼
      Predictions → NMS → (x, y, w, h, conf)
```

**Why Swin + YOLO?** Swin provides stronger multi-scale features than YOLO's CNN backbone (better for small/overlapping cells), while YOLO's head and neck are battle-tested for fast detection.

---

## Files

| File | Description |
|------|-------------|
| `config.py` | Configuration dataclasses optimized for Swin backbone (lower LR, higher weight decay) |
| `swin_backbone.py` | `SwinDetectionModel` — injects Swin into YOLO with channel adapters, freezes CNN layers |
| `training.py` | `ImprovedModelTrainer` — K-fold training with Swin-specific settings |
| `inference.py` | Multi-scale TTA + WBF ensemble inference |
| `data_peparation.py` | Dataset loading, stratified K-fold splits, YOLO-format label generation |

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Image Size | 1280px | Matches YOLOv8x's proven setting |
| Epochs | 150 | Moderate for Swin convergence |
| Batch Size | 2 | Reduced — Swin backbone uses more VRAM |
| Optimizer | AdamW | Best for transformers |
| Learning Rate | 2e-4 | Lower than CNN YOLO (transformer sensitivity) |
| Weight Decay | 0.05 | Higher — standard transformer regularization |
| Warmup | 10 epochs | Longer for transformer stability |
| Mosaic | 0.3 | Proven from YOLOv8x success |
| Close Mosaic | 20 | Disable in last 20 epochs |
| Horizontal Flip | 0.5 | Cell orientation invariance |
| Loss Weights | box=7.5, cls=0.5, dfl=1.5 | Standard YOLO loss balance |

**Key differences from standard YOLO training:**
- Lower LR (2e-4 vs 5e-4) — transformers need gentler optimization
- Higher weight decay (0.05 vs 5e-4) — transformer regularization
- Longer warmup (10 vs 5 epochs) — Swin needs gradual warmup
- Gentler warmup bias LR (0.01 vs 0.1)

---

## Inference Configuration

| Parameter | Value |
|-----------|-------|
| Scales | [1024, 1280, 1536] |
| Confidence threshold | 0.0001 (ultra-low, matches YOLOv8x strategy) |
| IoU threshold | 0.3 |
| Max detections | 2000 |
| WBF IoU | 0.4 |
| TTA | Multi-scale + horizontal flip |
| Final thresholds | [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15] |

---

## Usage

### 1. Prepare Data

```bash
cd swin-yolo
python data_peparation.py --root_dir ../riva-partb-dataset --work_dir swin-yolo-exp --n_folds 5
```

Creates K-fold splits with YOLO-format labels.

### 2. Train

```bash
cd swin-yolo
python training.py
```

Trains all K folds. Outputs saved to:
- `runs/fold_N/weights/best.pt`
- `runs/fold_N/weights/last.pt`

### 3. Inference

```bash
cd swin-yolo
python inference.py --test_dir ../riva-partb-dataset/images/test --models best_f0.pt best_f1.pt
```

Runs multi-scale TTA with WBF ensemble across models.

---

## Dependencies

Requires everything in the root `requirements.txt` plus:
- `ultralytics` — YOLO implementation
- `timm` — Swin Transformer pretrained weights
- `ensemble_boxes` — Weighted Box Fusion

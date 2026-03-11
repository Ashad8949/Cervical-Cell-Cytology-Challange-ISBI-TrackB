# RT-DETR: Real-Time Detection Transformer for Cell Detection

Ultralytics **RT-DETR** (Real-Time Detection Transformer) for cervical cell detection. Uses the production-ready RT-DETR implementation with deformable attention, two-stage refinement, and IoU-aware classification — providing an end-to-end detection pipeline that requires no NMS.

---

## Architecture

```
Input Image
    │
    ▼
┌───────────────────────────────┐
│  HGNetV2 Backbone             │  Pretrained CNN backbone
│  (RT-DETR-L or RT-DETR-X)    │  Multi-scale feature extraction
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│  RT-DETR Hybrid Encoder       │  AIFI: Intra-scale feature interaction
│  (AIFI + CCFM)                │  CCFM: Cross-scale feature merge
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│  Transformer Decoder          │  Deformable cross-attention
│  (End-to-end detection)       │  Two-stage refinement
│                               │  IoU-aware classification
│                               │  Hungarian matching (no NMS)
└───────────┬───────────────────┘
            │
            ▼
      Predictions (x, y, w, h, conf)
```

**Advantages over YOLO**: End-to-end (no NMS), better global context via transformer decoder, cleaner confidence scores, superior on small/overlapping objects.

---

## Files

| File | Description |
|------|-------------|
| `train_rtdetr.py` | V1 training — RT-DETR-L/X at 640px, 80 epochs, basic augmentation |
| `train_rtdetr_v2.py` | V2 training — targets 0.55+ mAP, 1024px native, 150 epochs, tuned augmentation |
| `inference_rtdetr.py` | Inference → Kaggle CSV submission, optional fixed 100×100 boxes |

---

## Training Configurations

### V1 (Basic — `train_rtdetr.py`)

| Parameter | Value |
|-----------|-------|
| Model | `rtdetr-l.pt` (or `rtdetr-x.pt`) |
| Image Size | 640px |
| Epochs | 80 |
| Batch Size | 4 |
| LR | 1e-4 |
| Mosaic | 0.5 |
| Mixup | 0.1 |
| Copy-paste | 0.1 |
| Patience | 20 |

### V2 (Improved — `train_rtdetr_v2.py`)

| Parameter | Value | Improvement |
|-----------|-------|-------------|
| Model | `rtdetr-l.pt` / `rtdetr-x.pt` | Same |
| Image Size | **1024px** | Native resolution — cells are ~100px |
| Epochs | **150** | More training time |
| Batch Size | 4 | Same |
| LR | 1e-4 | Same |
| Mosaic | **0.8** | More aggressive |
| Mixup | **0.15** | Slightly increased |
| Copy-paste | **0.15** | Slightly increased |
| Close Mosaic | **20** | Disable mosaic for clean convergence |
| Erasing | **0.1** | Added random erasing |
| Rotation | **15°** | More geometric variation |
| FlipUD | **0.5** | Cells are rotation-invariant |
| Patience | **30** | More patience |

---

## Usage

### Prerequisites

Convert dataset to YOLO format first (required for Ultralytics RT-DETR):

```bash
python scripts/convert_to_yolo.py
```

### Train

#### V1 (Quick baseline)
```bash
# RT-DETR-L at 640px
python rtdetr/train_rtdetr.py

# RT-DETR-X (larger model)
python rtdetr/train_rtdetr.py --model rtdetr-x.pt

# Resume training
python rtdetr/train_rtdetr.py --resume
```

#### V2 (Recommended)
```bash
# RT-DETR-L at 1024px (recommended)
python rtdetr/train_rtdetr_v2.py --model rtdetr-l.pt --imgsz 1024 --batch 4 --name rtdetr_l_1024

# RT-DETR-X at 1024px (larger model, batch 2)
python rtdetr/train_rtdetr_v2.py --model rtdetr-x.pt --imgsz 1024 --batch 2 --name rtdetr_x_1024

# Resume
python rtdetr/train_rtdetr_v2.py --resume --name rtdetr_l_1024
```

### Inference

```bash
# Basic inference
python rtdetr/inference_rtdetr.py \
    --weights experiments/rtdetr_riva/weights/best.pt \
    --test-dir riva-partb-dataset/images/test \
    --output submissions/rtdetr_submission.csv \
    --imgsz 1024 --conf 0.1

# With TTA
python rtdetr/inference_rtdetr.py \
    --weights experiments/rtdetr_riva/weights/best.pt \
    --test-dir riva-partb-dataset/images/test \
    --output submissions/rtdetr_tta.csv \
    --imgsz 1024 --conf 0.05 --augment

# With fixed 100×100 boxes (matches GT format)
python rtdetr/inference_rtdetr.py \
    --weights experiments/rtdetr_riva/weights/best.pt \
    --test-dir riva-partb-dataset/images/test \
    --output submissions/rtdetr_fixbox.csv \
    --fix-box-size 100
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--weights` | required | Path to trained weights (.pt) |
| `--test-dir` | `riva-partb-dataset/images/test` | Test images directory |
| `--output` | `submissions/rtdetr_submission.csv` | Output CSV path |
| `--imgsz` | 640 | Inference image size |
| `--conf` | 0.1 | Confidence threshold |
| `--iou` | 0.5 | NMS IoU threshold |
| `--batch` | 8 | Batch size |
| `--augment` | off | Enable TTA |
| `--fix-box-size` | 100 | Force boxes to fixed size (0 = disable) |

---

## Output Format

Kaggle-format CSV:

```csv
id,image_filename,class,x,y,width,height,conf
0,LSIL_45_3.png,0,1000.55,526.33,100,100,0.95
1,LSIL_45_3.png,0,842.12,433.78,100,100,0.87
```

---

## Dependencies

Requires everything in the root `requirements.txt` plus:
- `ultralytics` — RT-DETR implementation

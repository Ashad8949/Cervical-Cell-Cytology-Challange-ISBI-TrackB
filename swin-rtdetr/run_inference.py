"""
RIVA Part B - Run All Swin-RTDETR Inference
Runs 3 inference passes:
  1. Fold 1 only (best_f1.pt)
  2. Fold 2 only (best_f2.pt)
  3. Ensemble (both folds combined)

Usage:
  python run_inference.py
"""

import os
import sys
import glob
import tempfile
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory for imports if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import InferenceConfig

# Try importing from ultralytics
from ultralytics import RTDETR
from ensemble_boxes import weighted_boxes_fusion

# Import swin_backbone to register custom model classes for checkpoint loading
import swin_backbone


# ======================================================================
# Configuration
# ======================================================================

# Paths — adjust if your layout differs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)  # TrackB/

MODEL_F1 = os.path.join(PROJECT_DIR, "best_f1.pt")
MODEL_F2 = os.path.join(PROJECT_DIR, "best_f2.pt")
TEST_DIR = os.path.join(PROJECT_DIR, "riva-partb-dataset", "images", "test")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "submissions", "swin-rtdetr")

# Inference settings
SCALES = [1024, 1280, 1536]
CONF_THRESHOLD = 0.001       # Low — WBF handles filtering
IOU_THRESHOLD = 0.3
MAX_DET = 2000
WBF_IOU = 0.4
WBF_SKIP = 0.001
AUGMENT = True               # TTA within ultralytics

# Thresholds for submission variants
FINAL_THRESHOLDS = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15]


# ======================================================================
# Inference helpers
# ======================================================================

def predict_single_scale(model, img_path: str, scale: int) -> Tuple[List, List]:
    """Run prediction at one scale, return normalized xyxy boxes + scores."""
    results = model.predict(
        source=img_path,
        imgsz=scale,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        augment=AUGMENT,
        max_det=MAX_DET,
        verbose=False,
    )
    boxes, scores = [], []
    if results[0].boxes is not None and len(results[0].boxes):
        xywhn = results[0].boxes.xywhn.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        for (xc, yc, w, h), c in zip(xywhn, confs):
            boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
            scores.append(float(c))
    return boxes, scores


def predict_flipped(model, img_path: str, scale: int) -> Tuple[List, List]:
    """Predict on horizontally-flipped image and mirror boxes back."""
    img = cv2.imread(img_path)
    img_flip = cv2.flip(img, 1)

    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(tmp, img_flip)

    results = model.predict(
        source=tmp,
        imgsz=scale,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        augment=AUGMENT,
        max_det=MAX_DET,
        verbose=False,
    )

    boxes, scores = [], []
    if results[0].boxes is not None and len(results[0].boxes):
        xywhn = results[0].boxes.xywhn.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        for (xc, yc, w, h), c in zip(xywhn, confs):
            xc_orig = 1.0 - xc
            boxes.append([xc_orig - w / 2, yc - h / 2, xc_orig + w / 2, yc + h / 2])
            scores.append(float(c))

    if os.path.exists(tmp):
        os.remove(tmp)

    return boxes, scores


def multi_scale_tta(model, img_path: str) -> Tuple[List, List]:
    """Run multi-scale TTA (original + flipped) at all scales."""
    all_boxes, all_scores = [], []
    for scale in SCALES:
        b, s = predict_single_scale(model, img_path, scale)
        all_boxes.extend(b)
        all_scores.extend(s)
        b, s = predict_flipped(model, img_path, scale)
        all_boxes.extend(b)
        all_scores.extend(s)
    return all_boxes, all_scores


def run_inference_for_models(
    model_paths: List[str],
    test_dir: str,
    tag: str,
    output_dir: str,
):
    """
    Run full multi-scale TTA inference with one or more models,
    fuse with WBF, and write submission CSVs.
    """
    print(f"\n{'='*80}")
    print(f"  INFERENCE: {tag}")
    print(f"{'='*80}")
    print(f"  Models : {model_paths}")
    print(f"  Scales : {SCALES}")
    print(f"  TTA    : original + horizontal flip")
    print(f"  WBF iou: {WBF_IOU}   skip: {WBF_SKIP}")
    print(f"{'='*80}")

    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])
    print(f"  Test images: {len(test_images)}")

    # --- Collect predictions from every model ---
    per_image: Dict[str, Dict[str, List]] = {}
    for model_path in model_paths:
        print(f"\n  Loading model: {model_path}")
        model = RTDETR(model_path)

        for img_name in tqdm(test_images, desc=f"  {os.path.basename(model_path)}"):
            img_path = os.path.join(test_dir, img_name)
            boxes, scores = multi_scale_tta(model, img_path)

            if img_name not in per_image:
                per_image[img_name] = {"boxes": [], "scores": []}
            per_image[img_name]["boxes"].extend(boxes)
            per_image[img_name]["scores"].extend(scores)

    # --- WBF fusion ---
    print("\n  Applying WBF fusion...")
    rows = []
    pred_id = 0
    for img_name in tqdm(sorted(per_image.keys()), desc="  WBF"):
        data = per_image[img_name]
        if not data["boxes"]:
            continue
        boxes_arr = np.array(data["boxes"])
        scores_arr = np.array(data["scores"])
        labels_arr = np.zeros(len(boxes_arr))

        boxes_fused, scores_fused, _ = weighted_boxes_fusion(
            [boxes_arr], [scores_arr], [labels_arr],
            weights=None,
            iou_thr=WBF_IOU,
            skip_box_thr=WBF_SKIP,
            conf_type="avg",
        )

        # Read image dimensions for absolute coordinate conversion
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img_h, img_w = img.shape[:2]
        else:
            img_h, img_w = 1024, 1024

        for box, conf in zip(boxes_fused, scores_fused):
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2) * img_w
            cy = ((y1 + y2) / 2) * img_h
            bw = (x2 - x1) * img_w
            bh = (y2 - y1) * img_h

            rows.append({
                "id": pred_id,
                "image_filename": img_name,
                "class": 0,
                "x": cx,
                "y": cy,
                "width": bw,
                "height": bh,
                "conf": float(conf),
            })
            pred_id += 1

    df = pd.DataFrame(rows)

    # --- Write submissions ---
    os.makedirs(output_dir, exist_ok=True)

    # Full (all detections)
    full_path = os.path.join(output_dir, f"{tag}_all.csv")
    df.to_csv(full_path, index=False, float_format="%.6f")

    n_images = df["image_filename"].nunique() if len(df) else 0

    print(f"\n  Total detections: {len(df):,}  |  Images: {n_images}")
    if len(df):
        print(f"  Conf range: {df['conf'].min():.4f} – {df['conf'].max():.4f}")

    # Threshold variants
    print(f"\n  Threshold variants:")
    for thr in FINAL_THRESHOLDS:
        sub = df[df["conf"] >= thr].copy()
        sub["id"] = range(len(sub))
        out_path = os.path.join(output_dir, f"{tag}_conf_{thr:.4f}.csv")
        sub.to_csv(out_path, index=False, float_format="%.6f")
        avg = len(sub) / n_images if n_images else 0
        print(f"    {thr:.4f}: {len(sub):>6,} dets  ({avg:>6.1f}/img)  → {out_path}")

    print(f"\n  ✓ {tag} done.\n")
    return df


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 80)
    print("  SWIN-RTDETR INFERENCE PIPELINE")
    print("  3 runs: fold1 | fold2 | ensemble")
    print("=" * 80)

    # Verify paths
    for path, label in [(MODEL_F1, "Model F1"), (MODEL_F2, "Model F2"), (TEST_DIR, "Test dir")]:
        exists = os.path.exists(path)
        print(f"  [{('OK' if exists else 'MISSING'):>7s}] {label}: {path}")
        if not exists:
            print(f"\n  ERROR: {label} not found. Aborting.")
            return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Run 1: Fold 1 only ---
    run_inference_for_models(
        model_paths=[MODEL_F1],
        test_dir=TEST_DIR,
        tag="swin_rtdetr_fold1",
        output_dir=OUTPUT_DIR,
    )

    # --- Run 2: Fold 2 only ---
    run_inference_for_models(
        model_paths=[MODEL_F2],
        test_dir=TEST_DIR,
        tag="swin_rtdetr_fold2",
        output_dir=OUTPUT_DIR,
    )

    # --- Run 3: Ensemble (both folds) ---
    run_inference_for_models(
        model_paths=[MODEL_F1, MODEL_F2],
        test_dir=TEST_DIR,
        tag="swin_rtdetr_ensemble",
        output_dir=OUTPUT_DIR,
    )

    # Summary
    print("\n" + "=" * 80)
    print("  ALL 3 INFERENCE RUNS COMPLETE")
    print("=" * 80)
    print(f"\n  Outputs in: {OUTPUT_DIR}")
    print(f"\n  Submissions generated:")
    for csv_file in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))):
        size = os.path.getsize(csv_file)
        print(f"    {os.path.basename(csv_file):>50s}  ({size:>10,} bytes)")

    print("\n  Recommended submission order:")
    print("    1. swin_rtdetr_ensemble_conf_0.0500.csv")
    print("    2. swin_rtdetr_fold1_conf_0.0500.csv")
    print("    3. swin_rtdetr_fold2_conf_0.0500.csv")
    print("    4. Try other thresholds (0.01, 0.10, 0.15)")
    print("=" * 80)


if __name__ == "__main__":
    main()

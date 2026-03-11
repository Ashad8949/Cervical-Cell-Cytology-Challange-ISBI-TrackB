"""
Visualization utilities for RIVA Cell Detection
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
    box_format: str = 'xyxy'
) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image (H, W, C) in BGR format
        boxes: Bounding boxes, shape (N, 4)
        scores: Confidence scores, shape (N,)
        labels: Class labels, shape (N,)
        color: Box color in BGR
        thickness: Line thickness
        font_scale: Font scale for labels
        box_format: 'xyxy' or 'cxcywh'
    
    Returns:
        Image with drawn boxes
    """
    image = image.copy()
    h, w = image.shape[:2]
    
    if len(boxes) == 0:
        return image
    
    boxes = np.atleast_2d(boxes)
    
    # Convert to xyxy if needed
    if box_format == 'cxcywh':
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        # If normalized, scale to image size
        if np.max(boxes) <= 1.0:
            cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
        x1 = (cx - bw / 2).astype(int)
        y1 = (cy - bh / 2).astype(int)
        x2 = (cx + bw / 2).astype(int)
        y2 = (cy + bh / 2).astype(int)
    else:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        # If normalized, scale to image size
        if np.max(boxes) <= 1.0:
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
        x1, y1, x2, y2 = x1.astype(int), y1.astype(int), x2.astype(int), y2.astype(int)
    
    for i in range(len(boxes)):
        # Draw rectangle
        cv2.rectangle(image, (x1[i], y1[i]), (x2[i], y2[i]), color, thickness)
        
        # Draw label
        if scores is not None:
            label = f'{scores[i]:.2f}'
            if labels is not None:
                label = f'{int(labels[i])}: {label}'
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(image, (x1[i], y1[i] - text_h - 5), (x1[i] + text_w, y1[i]), color, -1)
            cv2.putText(image, label, (x1[i], y1[i] - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    
    return image


def visualize_predictions(
    image: np.ndarray,
    predictions: Dict[str, np.ndarray],
    targets: Optional[Dict[str, np.ndarray]] = None,
    pred_color: Tuple[int, int, int] = (0, 255, 0),
    gt_color: Tuple[int, int, int] = (0, 0, 255),
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[np.ndarray]:
    """
    Visualize predictions and optionally ground truth
    
    Args:
        image: Input image (H, W, C) in RGB format
        predictions: Dict with 'boxes', 'scores', 'labels'
        targets: Optional dict with 'boxes', 'labels'
        pred_color: Color for predictions (RGB)
        gt_color: Color for ground truth (RGB)
        save_path: Path to save figure
        show: Whether to display figure
        figsize: Figure size
    
    Returns:
        Annotated image if show=False
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Display image
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    ax.imshow(image)
    
    h, w = image.shape[:2]
    
    # Draw predictions
    if 'boxes' in predictions and len(predictions['boxes']) > 0:
        boxes = predictions['boxes']
        scores = predictions.get('scores', np.ones(len(boxes)))
        
        for i, box in enumerate(boxes):
            # Convert to xyxy if cxcywh
            if np.max(box) <= 1.0:
                cx, cy, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
                x1, y1 = cx - bw / 2, cy - bh / 2
                x2, y2 = cx + bw / 2, cy + bh / 2
            else:
                x1, y1, x2, y2 = box
            
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=[c/255 for c in pred_color], facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add score
            ax.text(x1, y1 - 5, f'{scores[i]:.2f}', color=[c/255 for c in pred_color], fontsize=8)
    
    # Draw ground truth
    if targets is not None and 'boxes' in targets and len(targets['boxes']) > 0:
        boxes = targets['boxes']
        
        for box in boxes:
            # Convert to xyxy if cxcywh
            if np.max(box) <= 1.0:
                cx, cy, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
                x1, y1 = cx - bw / 2, cy - bh / 2
                x2, y2 = cx + bw / 2, cy + bh / 2
            else:
                x1, y1, x2, y2 = box
            
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=[c/255 for c in gt_color], facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
    
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')
    
    # Legend
    pred_patch = patches.Patch(color=[c/255 for c in pred_color], label='Predictions')
    if targets is not None:
        gt_patch = patches.Patch(color=[c/255 for c in gt_color], label='Ground Truth')
        ax.legend(handles=[pred_patch, gt_patch], loc='upper right')
    else:
        ax.legend(handles=[pred_patch], loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    else:
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot training curves
    
    Args:
        history: Dict with keys like 'train_loss', 'val_loss', 'val_mAP'
        save_path: Path to save figure
        show: Whether to display figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss curves
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train Loss', color='blue')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Val Loss', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # mAP curves
    ax = axes[1]
    if 'val_mAP' in history:
        ax.plot(history['val_mAP'], label='mAP@0.50:0.95', color='green')
    if 'val_mAP@0.50' in history:
        ax.plot(history['val_mAP@0.50'], label='mAP@0.50', color='lime', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('mAP Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision/Recall
    ax = axes[2]
    if 'val_precision' in history:
        ax.plot(history['val_precision'], label='Precision', color='purple')
    if 'val_recall' in history:
        ax.plot(history['val_recall'], label='Recall', color='red')
    if 'val_f1' in history:
        ax.plot(history['val_f1'], label='F1', color='cyan')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision/Recall Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def create_comparison_grid(
    images: List[np.ndarray],
    predictions_list: List[Dict],
    targets_list: Optional[List[Dict]] = None,
    cols: int = 4,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize_per_image: Tuple[float, float] = (4, 4)
):
    """
    Create a grid of images with predictions
    
    Args:
        images: List of images
        predictions_list: List of prediction dicts
        targets_list: Optional list of target dicts
        cols: Number of columns in grid
        save_path: Path to save figure
        show: Whether to display
        figsize_per_image: Size per image in grid
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_image[0], rows * figsize_per_image[1]))
    
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    for i, (img, pred) in enumerate(zip(images, predictions_list)):
        row, col = i // cols, i % cols
        ax = axes[row][col]
        
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # Draw boxes
        target = targets_list[i] if targets_list else None
        annotated = draw_boxes(img.copy(), pred.get('boxes', []), pred.get('scores', None), box_format='cxcywh')
        
        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f'Image {i+1}: {len(pred.get("boxes", []))} detections')
    
    # Hide empty subplots
    for i in range(n, rows * cols):
        row, col = i // cols, i % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)

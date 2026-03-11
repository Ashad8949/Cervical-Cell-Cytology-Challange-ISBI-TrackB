"""
Detection metrics for RIVA Cell Detection
Computes mAP@0.50:0.95 following COCO evaluation protocol
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import torch


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes
    
    Args:
        box1: Boxes in [x1, y1, x2, y2] format, shape (N, 4)
        box2: Boxes in [x1, y1, x2, y2] format, shape (M, 4)
    
    Returns:
        IoU matrix of shape (N, M)
    """
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)))
    
    # Ensure 2D
    box1 = np.atleast_2d(box1)
    box2 = np.atleast_2d(box2)
    
    # Intersection
    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - intersection
    
    return intersection / (union + 1e-8)


def box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format"""
    if len(boxes) == 0:
        return boxes
    boxes = np.atleast_2d(boxes)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using 101-point interpolation (COCO style)
    """
    # Add sentinel values
    recalls = np.concatenate([[0.], recalls, [1.]])
    precisions = np.concatenate([[1.], precisions, [0.]])
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    ap = 0
    for r in recall_points:
        precision_at_r = precisions[recalls >= r]
        ap += precision_at_r[0] if len(precision_at_r) > 0 else 0
    
    return ap / 101


def compute_map(
    predictions: List[Dict],
    targets: List[Dict],
    iou_thresholds: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute mAP at multiple IoU thresholds
    
    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        targets: List of target dicts with 'boxes', 'labels'
        iou_thresholds: List of IoU thresholds (default: 0.50:0.95:0.05)
    
    Returns:
        Dictionary with mAP values
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    
    aps_per_threshold = []
    
    for iou_thresh in iou_thresholds:
        ap = compute_ap_at_threshold(predictions, targets, iou_thresh)
        aps_per_threshold.append(ap)
    
    return {
        'mAP': np.mean(aps_per_threshold),
        'mAP@0.50': aps_per_threshold[0] if len(aps_per_threshold) > 0 else 0,
        'mAP@0.75': aps_per_threshold[5] if len(aps_per_threshold) > 5 else 0,
    }


def compute_ap_at_threshold(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float
) -> float:
    """
    Compute AP at a single IoU threshold
    """
    all_scores = []
    all_matches = []
    total_gt = 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred.get('boxes', np.array([]))
        pred_scores = pred.get('scores', np.array([]))
        gt_boxes = target.get('boxes', np.array([]))
        
        # Convert to numpy if tensor
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().numpy()
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
        
        # Convert box format if needed (cxcywh to xyxy)
        if len(pred_boxes) > 0 and pred_boxes.shape[1] == 4:
            # Check if likely cxcywh format (w, h should be positive and smaller than cx, cy)
            if np.all(pred_boxes[:, 2] < 1) and np.all(pred_boxes[:, 3] < 1):
                pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
        
        if len(gt_boxes) > 0 and gt_boxes.shape[1] == 4:
            if np.all(gt_boxes[:, 2] < 1) and np.all(gt_boxes[:, 3] < 1):
                gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
        
        total_gt += len(gt_boxes)
        
        if len(pred_boxes) == 0:
            continue
        
        if len(gt_boxes) == 0:
            all_scores.extend(pred_scores.tolist())
            all_matches.extend([False] * len(pred_scores))
            continue
        
        # Compute IoU matrix
        iou_matrix = compute_iou(pred_boxes, gt_boxes)
        
        # Match predictions to ground truth
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        
        # Sort predictions by score
        sorted_indices = np.argsort(pred_scores)[::-1]
        
        for idx in sorted_indices:
            all_scores.append(pred_scores[idx])
            
            # Find best matching GT
            ious = iou_matrix[idx]
            best_gt = np.argmax(ious)
            best_iou = ious[best_gt]
            
            if best_iou >= iou_threshold and not gt_matched[best_gt]:
                all_matches.append(True)
                gt_matched[best_gt] = True
            else:
                all_matches.append(False)
    
    if total_gt == 0:
        return 0.0
    
    if len(all_scores) == 0:
        return 0.0
    
    # Sort by score
    sorted_indices = np.argsort(all_scores)[::-1]
    all_matches = np.array(all_matches)[sorted_indices]
    
    # Compute precision and recall
    tp = np.cumsum(all_matches)
    fp = np.cumsum(~all_matches)
    
    precision = tp / (tp + fp)
    recall = tp / total_gt
    
    return compute_ap(recall, precision)


class DetectionMetrics:
    """
    Detection metrics calculator for RIVA competition
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        iou_thresholds: Optional[List[float]] = None
    ):
        self.iou_threshold = iou_threshold
        self.iou_thresholds = iou_thresholds or list(np.arange(0.5, 1.0, 0.05))
        
        # Accumulators
        self.reset()
    
    def reset(self):
        """Reset accumulators"""
        self.predictions = []
        self.targets = []
    
    def update(
        self,
        predictions: List[Dict],
        targets: List[Dict]
    ):
        """Add batch of predictions and targets"""
        self.predictions.extend(predictions)
        self.targets.extend(targets)
    
    def compute(
        self,
        predictions: Optional[List[Dict]] = None,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics
        """
        if predictions is None:
            predictions = self.predictions
        if targets is None:
            targets = self.targets
        
        if len(predictions) == 0 or len(targets) == 0:
            return {
                'mAP': 0.0,
                'mAP@0.50': 0.0,
                'mAP@0.75': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        # Compute mAP
        map_results = compute_map(predictions, targets, self.iou_thresholds)
        
        # Compute precision, recall at default IoU threshold
        pr_metrics = self._compute_pr_metrics(predictions, targets)
        
        return {
            **map_results,
            **pr_metrics
        }
    
    def _compute_pr_metrics(
        self,
        predictions: List[Dict],
        targets: List[Dict]
    ) -> Dict[str, float]:
        """Compute precision, recall, F1 at default IoU threshold"""
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for pred, target in zip(predictions, targets):
            pred_boxes = pred.get('boxes', np.array([]))
            pred_scores = pred.get('scores', np.array([]))
            gt_boxes = target.get('boxes', np.array([]))
            
            # Convert to numpy
            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.cpu().numpy()
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.cpu().numpy()
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
            
            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue
            
            if len(pred_boxes) == 0:
                total_fn += len(gt_boxes)
                continue
            
            if len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue
            
            # Convert format if needed
            if len(pred_boxes) > 0 and np.all(pred_boxes[:, 2] < 1):
                pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
            if len(gt_boxes) > 0 and np.all(gt_boxes[:, 2] < 1):
                gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
            
            # Compute IoU
            iou_matrix = compute_iou(pred_boxes, gt_boxes)
            
            # Match
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)
            
            sorted_indices = np.argsort(pred_scores)[::-1]
            
            for idx in sorted_indices:
                ious = iou_matrix[idx]
                best_gt = np.argmax(ious)
                best_iou = ious[best_gt]
                
                if best_iou >= self.iou_threshold and not gt_matched[best_gt]:
                    total_tp += 1
                    gt_matched[best_gt] = True
                else:
                    total_fp += 1
            
            total_fn += np.sum(~gt_matched)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

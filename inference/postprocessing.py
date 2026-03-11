"""
Post-processing utilities for RIVA Cell Detection
Includes NMS, Soft-NMS, confidence calibration, and size filtering
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict


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


def box_xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h] format"""
    if len(boxes) == 0:
        return boxes
    boxes = np.atleast_2d(boxes)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.stack([cx, cy, w, h], axis=1)


def compute_iou(box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between one box and multiple boxes
    
    Args:
        box1: Single box [x1, y1, x2, y2]
        boxes2: Multiple boxes (N, 4) [x1, y1, x2, y2]
    
    Returns:
        IoU values (N,)
    """
    x1 = np.maximum(box1[0], boxes2[:, 0])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y2 = np.minimum(box1[3], boxes2[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-8)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Non-Maximum Suppression
    
    Args:
        boxes: Boxes in [cx, cy, w, h] or [x1, y1, x2, y2] format (N, 4)
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Indices of kept boxes
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    boxes = np.atleast_2d(boxes)
    
    # Convert to xyxy if needed
    if np.max(boxes) <= 1.0 or (boxes[:, 2] < boxes[:, 0]).any():
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    else:
        boxes_xyxy = boxes
    
    # Sort by score
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        ious = compute_iou(boxes_xyxy[i], boxes_xyxy[order[1:]])
        
        # Keep boxes with IoU <= threshold
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int32)


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.01,
    method: str = 'gaussian'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soft Non-Maximum Suppression
    
    Args:
        boxes: Boxes in [cx, cy, w, h] format (N, 4)
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for hard suppression (linear method)
        sigma: Gaussian decay parameter
        score_threshold: Minimum score to keep
        method: 'gaussian' or 'linear'
    
    Returns:
        Tuple of (kept indices, updated scores)
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32), np.array([])
    
    boxes = np.atleast_2d(boxes)
    scores = scores.copy()
    
    # Convert to xyxy
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    
    N = len(boxes)
    indices = np.arange(N)
    
    for i in range(N):
        # Find max scoring box
        max_idx = np.argmax(scores[i:]) + i
        
        # Swap
        boxes_xyxy[[i, max_idx]] = boxes_xyxy[[max_idx, i]]
        scores[[i, max_idx]] = scores[[max_idx, i]]
        indices[[i, max_idx]] = indices[[max_idx, i]]
        
        # Compute IoU with remaining boxes
        if i < N - 1:
            ious = compute_iou(boxes_xyxy[i], boxes_xyxy[i+1:])
            
            if method == 'gaussian':
                # Gaussian decay
                decay = np.exp(-(ious ** 2) / sigma)
            else:
                # Linear decay
                decay = np.where(ious > iou_threshold, 1 - ious, 1)
            
            scores[i+1:] *= decay
    
    # Filter by score threshold
    keep_mask = scores >= score_threshold
    keep_indices = indices[keep_mask]
    kept_scores = scores[keep_mask]
    
    return keep_indices, kept_scores


def weighted_box_fusion(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
    iou_threshold: float = 0.5,
    skip_empty: bool = True
) -> Dict[str, np.ndarray]:
    """
    Weighted Boxes Fusion for ensemble predictions
    
    Args:
        boxes_list: List of box arrays from different models
        scores_list: List of score arrays
        labels_list: List of label arrays
        weights: Optional model weights
        iou_threshold: IoU threshold for fusion
        skip_empty: Skip empty predictions
    
    Returns:
        Dict with fused 'boxes', 'scores', 'labels'
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Convert all to xyxy format
    all_boxes = []
    all_scores = []
    all_labels = []
    all_model_indices = []
    
    for model_idx, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        if skip_empty and len(boxes) == 0:
            continue
        
        boxes = np.atleast_2d(boxes)
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        
        all_boxes.append(boxes_xyxy)
        all_scores.append(scores * weights[model_idx])
        all_labels.append(labels)
        all_model_indices.append(np.full(len(boxes), model_idx))
    
    if len(all_boxes) == 0:
        return {
            'boxes': np.zeros((0, 4), dtype=np.float32),
            'scores': np.zeros((0,), dtype=np.float32),
            'labels': np.zeros((0,), dtype=np.int32)
        }
    
    # Concatenate
    all_boxes = np.vstack(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    all_model_indices = np.concatenate(all_model_indices)
    
    # Group by label
    unique_labels = np.unique(all_labels)
    
    fused_boxes = []
    fused_scores = []
    fused_labels = []
    
    for label in unique_labels:
        label_mask = all_labels == label
        label_boxes = all_boxes[label_mask]
        label_scores = all_scores[label_mask]
        label_model_indices = all_model_indices[label_mask]
        
        # Cluster by IoU
        clusters = []
        used = np.zeros(len(label_boxes), dtype=bool)
        
        # Sort by score
        order = np.argsort(label_scores)[::-1]
        
        for idx in order:
            if used[idx]:
                continue
            
            # Start new cluster
            cluster_boxes = [label_boxes[idx]]
            cluster_scores = [label_scores[idx]]
            cluster_weights = [weights[int(label_model_indices[idx])]]
            used[idx] = True
            
            # Find matching boxes
            for other_idx in order:
                if used[other_idx]:
                    continue
                
                iou = compute_iou(label_boxes[idx], label_boxes[other_idx:other_idx+1])[0]
                
                if iou >= iou_threshold:
                    cluster_boxes.append(label_boxes[other_idx])
                    cluster_scores.append(label_scores[other_idx])
                    cluster_weights.append(weights[int(label_model_indices[other_idx])])
                    used[other_idx] = True
            
            clusters.append((cluster_boxes, cluster_scores, cluster_weights))
        
        # Fuse clusters
        for cluster_boxes, cluster_scores, cluster_weights in clusters:
            # Weighted average
            total_weight = sum(cluster_weights)
            fused_box = np.sum(
                [box * w for box, w in zip(cluster_boxes, cluster_weights)],
                axis=0
            ) / total_weight
            
            fused_score = sum(cluster_scores) / total_weight
            
            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
            fused_labels.append(label)
    
    # Convert back to cxcywh
    if len(fused_boxes) > 0:
        fused_boxes = np.array(fused_boxes)
        fused_boxes = box_xyxy_to_cxcywh(fused_boxes)
    else:
        fused_boxes = np.zeros((0, 4), dtype=np.float32)
    
    return {
        'boxes': fused_boxes.astype(np.float32),
        'scores': np.array(fused_scores, dtype=np.float32),
        'labels': np.array(fused_labels, dtype=np.int32)
    }


def size_filter(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    min_size: float = 0.001,
    max_size: float = 0.3,
    image_size: Optional[Tuple[int, int]] = None
) -> Dict[str, np.ndarray]:
    """
    Filter boxes by size (biological plausibility)
    
    Args:
        boxes: Boxes in [cx, cy, w, h] format (normalized)
        scores: Confidence scores
        labels: Class labels
        min_size: Minimum relative size (w*h)
        max_size: Maximum relative size (w*h)
        image_size: Optional (H, W) for denormalization
    
    Returns:
        Filtered predictions dict
    """
    if len(boxes) == 0:
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }
    
    boxes = np.atleast_2d(boxes)
    
    # Compute relative areas
    if image_size is not None:
        h, w = image_size
        areas = (boxes[:, 2] * w) * (boxes[:, 3] * h) / (w * h)
    else:
        areas = boxes[:, 2] * boxes[:, 3]
    
    # Filter
    keep = (areas >= min_size) & (areas <= max_size)
    
    return {
        'boxes': boxes[keep],
        'scores': scores[keep],
        'labels': labels[keep]
    }


def confidence_calibration(
    scores: np.ndarray,
    method: str = 'temperature',
    temperature: float = 1.5,
    scaling_factor: float = 1.0,
    offset: float = 0.0
) -> np.ndarray:
    """
    Calibrate confidence scores
    
    Args:
        scores: Raw confidence scores
        method: 'temperature', 'linear', or 'isotonic'
        temperature: Temperature for temperature scaling
        scaling_factor: Scale factor for linear method
        offset: Offset for linear method
    
    Returns:
        Calibrated scores
    """
    if len(scores) == 0:
        return scores
    
    if method == 'temperature':
        # Temperature scaling (softmax-like adjustment)
        calibrated = scores ** (1 / temperature)
        # Renormalize to [0, 1]
        calibrated = calibrated / (calibrated.max() + 1e-8)
    elif method == 'linear':
        calibrated = scores * scaling_factor + offset
        calibrated = np.clip(calibrated, 0, 1)
    else:
        calibrated = scores
    
    return calibrated


def cluster_based_filtering(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int = 1,
    max_density: float = 0.5,
    radius: float = 0.05
) -> Dict[str, np.ndarray]:
    """
    Filter predictions based on spatial clustering
    
    Args:
        boxes: Boxes in [cx, cy, w, h] format
        scores: Confidence scores
        labels: Class labels
        min_cluster_size: Minimum detections in cluster
        max_density: Maximum local density
        radius: Radius for density computation
    
    Returns:
        Filtered predictions dict
    """
    if len(boxes) < 2:
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }
    
    boxes = np.atleast_2d(boxes)
    centers = boxes[:, :2]  # cx, cy
    
    # Compute pairwise distances
    dist_matrix = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    
    # Count neighbors within radius
    neighbor_counts = np.sum(dist_matrix < radius, axis=1) - 1  # Exclude self
    
    # Compute local density
    areas = boxes[:, 2] * boxes[:, 3]
    local_density = neighbor_counts * areas
    
    # Filter by density
    keep = local_density <= max_density
    
    return {
        'boxes': boxes[keep],
        'scores': scores[keep],
        'labels': labels[keep]
    }


class PostProcessor:
    """
    Complete post-processing pipeline for cell detection
    """
    
    def __init__(
        self,
        nms_threshold: float = 0.5,
        confidence_threshold: float = 0.1,
        max_detections: int = 300,
        min_size: float = 0.001,
        max_size: float = 0.3,
        use_soft_nms: bool = False,
        soft_nms_sigma: float = 0.5,
        calibration_method: str = 'none',
        calibration_temperature: float = 1.5
    ):
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.min_size = min_size
        self.max_size = max_size
        self.use_soft_nms = use_soft_nms
        self.soft_nms_sigma = soft_nms_sigma
        self.calibration_method = calibration_method
        self.calibration_temperature = calibration_temperature
    
    def __call__(
        self,
        predictions: Dict[str, Union[np.ndarray, torch.Tensor]],
        image_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Apply full post-processing pipeline
        
        Args:
            predictions: Dict with 'boxes', 'scores', 'labels'
            image_size: Optional (H, W) tuple
        
        Returns:
            Processed predictions dict
        """
        # Convert to numpy if needed
        boxes = predictions.get('boxes', np.array([]))
        scores = predictions.get('scores', np.array([]))
        labels = predictions.get('labels', np.array([]))
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        if len(boxes) == 0:
            return {
                'boxes': np.zeros((0, 4), dtype=np.float32),
                'scores': np.zeros((0,), dtype=np.float32),
                'labels': np.zeros((0,), dtype=np.int32)
            }
        
        # 1. Filter by confidence
        keep = scores >= self.confidence_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        if len(boxes) == 0:
            return {
                'boxes': np.zeros((0, 4), dtype=np.float32),
                'scores': np.zeros((0,), dtype=np.float32),
                'labels': np.zeros((0,), dtype=np.int32)
            }
        
        # 2. Apply NMS
        if self.use_soft_nms:
            keep_indices, new_scores = soft_nms(
                boxes, scores,
                iou_threshold=self.nms_threshold,
                sigma=self.soft_nms_sigma
            )
            boxes = boxes[keep_indices]
            scores = new_scores
            labels = labels[keep_indices]
        else:
            keep_indices = nms(boxes, scores, self.nms_threshold)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
        
        # 3. Size filtering
        result = size_filter(
            boxes, scores, labels,
            self.min_size, self.max_size, image_size
        )
        boxes = result['boxes']
        scores = result['scores']
        labels = result['labels']
        
        # 4. Confidence calibration
        if self.calibration_method != 'none':
            scores = confidence_calibration(
                scores,
                method=self.calibration_method,
                temperature=self.calibration_temperature
            )
        
        # 5. Limit detections
        if len(scores) > self.max_detections:
            top_indices = np.argsort(scores)[::-1][:self.max_detections]
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            labels = labels[top_indices]
        
        return {
            'boxes': boxes.astype(np.float32),
            'scores': scores.astype(np.float32),
            'labels': labels.astype(np.int32)
        }

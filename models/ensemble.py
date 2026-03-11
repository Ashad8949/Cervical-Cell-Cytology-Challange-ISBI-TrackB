import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import itertools
from scipy.optimize import linear_sum_assignment

class ModelEnsemble:
    """
    Ensemble multiple models with weighted fusion
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        fusion_method: str = 'weighted_box_fusion',
        device: str = 'cuda'
    ):
        """
        Args:
            models: List of trained models
            weights: Weight for each model (sum to 1)
            fusion_method: 'weighted_box_fusion', 'nms', 'soft_nms', 'nmw'
            device: Device to run inference on
        """
        self.models = models
        self.device = device
        
        # Set weights
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]
        
        # Fusion method
        self.fusion_method = fusion_method
        if fusion_method == 'weighted_box_fusion':
            self.fuser = WeightedBoxesFusion()
        elif fusion_method == 'nms':
            self.fuser = NonMaximumSuppression()
        elif fusion_method == 'soft_nms':
            self.fuser = SoftNMS()
        elif fusion_method == 'nmw':
            self.fuser = NonMaximumWeighted()
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Move models to device
        for model in self.models:
            model.to(device)
            model.eval()
    
    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        confidence_threshold: float = 0.1,
        iou_threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Ensemble prediction
        
        Args:
            image: Input image tensor [1, 3, H, W]
            confidence_threshold: Minimum confidence score
            iou_threshold: IoU threshold for NMS
        
        Returns:
            Dictionary with predictions
        """
        all_predictions = []
        
        # Get predictions from all models
        for model, weight in zip(self.models, self.weights):
            # Forward pass
            outputs = model(image.to(self.device))
            
            # Post-process predictions
            predictions = self._post_process(
                outputs,
                confidence_threshold=confidence_threshold,
                weight=weight
            )
            
            all_predictions.append(predictions)
        
        # Fuse predictions
        if len(all_predictions) == 1:
            fused = all_predictions[0]
        else:
            fused = self.fuser(
                all_predictions,
                iou_threshold=iou_threshold
            )
        
        return fused
    
    def _post_process(
        self,
        outputs: Dict[str, torch.Tensor],
        confidence_threshold: float = 0.1,
        weight: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Post-process model outputs
        """
        # Get predictions
        logits = outputs['pred_logits'].sigmoid()
        boxes = outputs['pred_boxes']
        
        # Filter by confidence
        scores = logits[..., 0].cpu().numpy()
        boxes = boxes.cpu().numpy()
        
        # Apply weight to scores
        scores = scores * weight
        
        # Filter
        keep = scores > confidence_threshold
        filtered_scores = scores[keep]
        filtered_boxes = boxes[keep]
        
        # Convert to [x, y, w, h] format
        filtered_boxes = self._xyxy_to_xywh(filtered_boxes)
        
        return {
            'boxes': filtered_boxes,
            'scores': filtered_scores,
            'weight': weight
        }
    
    def _xyxy_to_xywh(self, boxes: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [x, y, w, h]"""
        xywh = np.zeros_like(boxes)
        xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x center
        xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y center
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]        # width
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]        # height
        return xywh
    
    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return xyxy


class WeightedBoxesFusion:
    """
    Weighted Boxes Fusion from ensemble-boxes
    """
    
    def __init__(
        self,
        iou_thr: float = 0.55,
        skip_box_thr: float = 0.0001,
        conf_type: str = 'avg',
        allows_overflow: bool = False
    ):
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        self.conf_type = conf_type
        self.allows_overflow = allows_overflow
    
    def __call__(
        self,
        predictions: List[Dict[str, np.ndarray]],
        iou_threshold: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Apply Weighted Boxes Fusion
        
        Args:
            predictions: List of predictions from different models
            iou_threshold: IoU threshold for matching
        
        Returns:
            Fused predictions
        """
        if iou_threshold is not None:
            self.iou_thr = iou_threshold
        
        # Combine all predictions
        all_boxes = []
        all_scores = []
        all_weights = []
        all_labels = []
        
        for pred in predictions:
            boxes = pred['boxes']
            scores = pred['scores']
            weight = pred.get('weight', 1.0)
            
            # Convert to xyxy format
            boxes_xyxy = self._xywh_to_xyxy(boxes)
            
            all_boxes.append(boxes_xyxy)
            all_scores.append(scores)
            all_weights.append(np.full(len(scores), weight))
            all_labels.append(np.zeros(len(scores), dtype=np.int32))
        
        # Fuse boxes
        boxes, scores, labels = self.weighted_boxes_fusion(
            all_boxes, all_scores, all_labels, all_weights
        )
        
        # Convert back to xywh format
        boxes = self._xyxy_to_xywh(boxes)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }
    
    def weighted_boxes_fusion(
        self,
        boxes_list: List[np.ndarray],
        scores_list: List[np.ndarray],
        labels_list: List[np.ndarray],
        weights: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Implementation of Weighted Boxes Fusion algorithm
        """
        # Implementation based on ensemble-boxes library
        # https://github.com/ZFTurbo/Weighted-Boxes-Fusion
        
        # Convert to list of lists
        boxes_list = [b.tolist() for b in boxes_list]
        scores_list = [s.tolist() for s in scores_list]
        labels_list = [l.tolist() for l in labels_list]
        
        if weights is None:
            weights = [np.ones(len(s)).tolist() for s in scores_list]
        else:
            weights = [w.tolist() for w in weights]
        
        # Create clusters
        clusters = []
        used_boxes = [set() for _ in range(len(boxes_list))]
        
        # Sort boxes by confidence
        all_boxes = []
        for model_idx, (boxes, scores, labels, ws) in enumerate(
            zip(boxes_list, scores_list, labels_list, weights)
        ):
            for box_idx, (box, score, label, weight) in enumerate(
                zip(boxes, scores, labels, ws)
            ):
                all_boxes.append({
                    'model_idx': model_idx,
                    'box_idx': box_idx,
                    'box': box,
                    'score': score,
                    'label': label,
                    'weight': weight
                })
        
        # Sort by score descending
        all_boxes.sort(key=lambda x: x['score'], reverse=True)
        
        # Cluster boxes
        for box_info in all_boxes:
            model_idx = box_info['model_idx']
            box_idx = box_info['box_idx']
            
            if box_idx in used_boxes[model_idx]:
                continue
            
            # Find matching boxes from other models
            cluster = [box_info]
            used_boxes[model_idx].add(box_idx)
            
            for other_box_info in all_boxes:
                other_model_idx = other_box_info['model_idx']
                other_box_idx = other_box_info['box_idx']
                
                if other_model_idx == model_idx:
                    continue
                
                if other_box_idx in used_boxes[other_model_idx]:
                    continue
                
                # Check IoU
                iou = self._compute_iou(box_info['box'], other_box_info['box'])
                
                if iou >= self.iou_thr and box_info['label'] == other_box_info['label']:
                    cluster.append(other_box_info)
                    used_boxes[other_model_idx].add(other_box_idx)
            
            clusters.append(cluster)
        
        # Fuse clusters
        fused_boxes = []
        fused_scores = []
        fused_labels = []
        
        for cluster in clusters:
            if not cluster:
                continue
            
            # Compute fused box
            cluster_boxes = np.array([c['box'] for c in cluster])
            cluster_scores = np.array([c['score'] for c in cluster])
            cluster_weights = np.array([c['weight'] for c in cluster])
            cluster_label = cluster[0]['label']
            
            # Weighted average
            total_weight = cluster_weights.sum()
            weighted_boxes = cluster_boxes * cluster_weights[:, None]
            fused_box = weighted_boxes.sum(axis=0) / total_weight
            
            # Fused score
            if self.conf_type == 'avg':
                fused_score = (cluster_scores * cluster_weights).sum() / total_weight
            elif self.conf_type == 'max':
                fused_score = cluster_scores.max()
            elif self.conf_type == 'box_and_model_avg':
                # Implementation specific
                pass
            
            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
            fused_labels.append(cluster_label)
        
        return (
            np.array(fused_boxes),
            np.array(fused_scores),
            np.array(fused_labels)
        )
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes in xyxy format"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert xywh to xyxy"""
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return xyxy
    
    def _xyxy_to_xywh(self, boxes: np.ndarray) -> np.ndarray:
        """Convert xyxy to xywh"""
        xywh = np.zeros_like(boxes)
        xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
        return xywh


class NonMaximumSuppression:
    """
    Non-Maximum Suppression for detections
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.05
    ):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
    
    def __call__(
        self,
        predictions: List[Dict[str, np.ndarray]],
        iou_threshold: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Apply NMS to combined predictions
        """
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        
        # Combine all predictions
        all_boxes = []
        all_scores = []
        
        for pred in predictions:
            boxes = pred['boxes']
            scores = pred['scores']
            
            # Convert to xyxy
            boxes_xyxy = self._xywh_to_xyxy(boxes)
            
            all_boxes.append(boxes_xyxy)
            all_scores.append(scores)
        
        # Concatenate
        boxes = np.vstack(all_boxes)
        scores = np.concatenate(all_scores)
        
        # Apply NMS
        keep = self.nms(boxes, scores)
        
        boxes = boxes[keep]
        scores = scores[keep]
        
        # Convert back to xywh
        boxes = self._xyxy_to_xywh(boxes)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': np.zeros(len(scores), dtype=np.int32)
        }
    
    def nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray
    ) -> np.ndarray:
        """
        Non-Maximum Suppression implementation
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
        
        # Sort by score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Compute IoU with remaining boxes
            ious = self._batch_iou(boxes[i], boxes[order[1:]])
            
            # Keep boxes with IoU <= threshold
            inds = np.where(ious <= self.iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep, dtype=np.int32)
    
    def _batch_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and multiple boxes"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - intersection
        
        return intersection / (union + 1e-8)
    
    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert xywh to xyxy"""
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return xyxy
    
    def _xyxy_to_xywh(self, boxes: np.ndarray) -> np.ndarray:
        """Convert xyxy to xywh"""
        xywh = np.zeros_like(boxes)
        xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
        return xywh


class SoftNMS:
    """
    Soft Non-Maximum Suppression
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        sigma: float = 0.5,
        score_threshold: float = 0.001,
        method: str = 'linear'
    ):
        self.iou_threshold = iou_threshold
        self.sigma = sigma
        self.score_threshold = score_threshold
        self.method = method
    
    def __call__(
        self,
        predictions: List[Dict[str, np.ndarray]],
        iou_threshold: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Apply Soft NMS
        """
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        
        # Combine predictions
        all_boxes = []
        all_scores = []
        
        for pred in predictions:
            boxes = pred['boxes']
            scores = pred['scores']
            
            boxes_xyxy = self._xywh_to_xyxy(boxes)
            
            all_boxes.append(boxes_xyxy)
            all_scores.append(scores)
        
        boxes = np.vstack(all_boxes)
        scores = np.concatenate(all_scores)
        
        # Apply Soft NMS
        keep = self.soft_nms(boxes, scores)
        
        boxes = boxes[keep]
        scores = scores[keep]
        
        # Convert back to xywh
        boxes = self._xyxy_to_xywh(boxes)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': np.zeros(len(scores), dtype=np.int32)
        }
    
    def soft_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray
    ) -> np.ndarray:
        """
        Soft NMS implementation
        """
        N = len(boxes)
        if N == 0:
            return np.array([], dtype=np.int32)
        
        # Initialize
        indexes = np.arange(N)
        soft_scores = scores.copy()
        
        for i in range(N):
            # Find max score
            max_pos = i + np.argmax(soft_scores[i:])
            
            # Swap
            boxes[[i, max_pos]] = boxes[[max_pos, i]]
            soft_scores[[i, max_pos]] = soft_scores[[max_pos, i]]
            indexes[[i, max_pos]] = indexes[[max_pos, i]]
            
            # IoU with remaining boxes
            ious = self._batch_iou(boxes[i], boxes[i+1:])
            
            # Decay scores
            if self.method == 'linear':
                # Linear decay
                decay = np.ones_like(ious)
                decay[ious > self.iou_threshold] = 1 - ious[ious > self.iou_threshold]
            elif self.method == 'gaussian':
                # Gaussian decay
                decay = np.exp(-(ious ** 2) / self.sigma)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            soft_scores[i+1:] *= decay
        
        # Filter by score threshold
        keep = indexes[soft_scores >= self.score_threshold]
        
        return keep
    
    def _batch_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and multiple boxes"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - intersection
        
        return intersection / (union + 1e-8)
    
    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert xywh to xyxy"""
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return xyxy
    
    def _xyxy_to_xywh(self, boxes: np.ndarray) -> np.ndarray:
        """Convert xyxy to xywh"""
        xywh = np.zeros_like(boxes)
        xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
        return xywh


class NonMaximumWeighted:
    """
    Non-Maximum Weighted (NMW) for better suppression
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.05,
        weight_threshold: float = 0.5
    ):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.weight_threshold = weight_threshold
    
    def __call__(
        self,
        predictions: List[Dict[str, np.ndarray]],
        iou_threshold: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Apply Non-Maximum Weighted
        """
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        
        # Implementation similar to NMS but with weighted averaging
        # Omitted for brevity
        pass
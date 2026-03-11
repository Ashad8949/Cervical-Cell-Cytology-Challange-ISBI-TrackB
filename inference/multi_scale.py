import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import cv2
from tqdm.auto import tqdm

class MultiScaleInference:
    """
    Multi-scale inference for cell detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        scales: List[int] = [768, 896, 1024, 1152, 1280],
        fusion_method: str = 'weighted_box_fusion',
        device: str = 'cuda',
        max_detections: int = 300,
        confidence_threshold: float = 0.1
    ):
        self.model = model
        self.scales = scales
        self.fusion_method = fusion_method
        self.device = device
        self.max_detections = max_detections
        self.confidence_threshold = confidence_threshold
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Multi-scale inference initialized with scales: {scales}")
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        original_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict at multiple scales and fuse results
        """
        if original_size is None:
            original_size = image.shape[:2]
        
        all_predictions = []
        
        for scale in self.scales:
            # Resize image
            resized = self._resize_image(image, scale)
            
            # Convert to tensor
            img_tensor = self._preprocess(resized).unsqueeze(0).to(self.device)
            
            # Inference
            outputs = self.model(img_tensor)
            
            # Post-process
            predictions = self._post_process(outputs)
            
            if len(predictions['boxes']) > 0:
                # Scale boxes back to original size
                predictions = self._scale_predictions(
                    predictions,
                    resized.shape[:2],
                    original_size
                )
                
                all_predictions.append(predictions)
        
        # Merge predictions from different scales
        if len(all_predictions) == 0:
            return {
                'boxes': np.zeros((0, 4), dtype=np.float32),
                'scores': np.zeros((0,), dtype=np.float32),
                'labels': np.zeros((0,), dtype=np.int32)
            }
        
        merged = self._merge_predictions(all_predictions)
        
        # Filter by confidence
        keep = merged['scores'] > self.confidence_threshold
        merged['boxes'] = merged['boxes'][keep]
        merged['scores'] = merged['scores'][keep]
        merged['labels'] = merged['labels'][keep]
        
        # Limit number of detections
        if len(merged['scores']) > self.max_detections:
            top_indices = np.argsort(merged['scores'])[::-1][:self.max_detections]
            merged['boxes'] = merged['boxes'][top_indices]
            merged['scores'] = merged['scores'][top_indices]
            merged['labels'] = merged['labels'][top_indices]
        
        return merged
    
    def _resize_image(
        self,
        image: np.ndarray,
        target_size: int
    ) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        # Resize longer side to target_size
        if h > w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target_size x target_size
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        return padded
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image = (image - mean) / std
        
        # Convert to tensor and reorder dimensions
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image
    
    def _post_process(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Post-process model outputs"""
        logits = outputs['pred_logits'].softmax(-1)
        boxes = outputs['pred_boxes']
        
        # Filter by confidence
        scores, labels = logits[0][:, :-1].max(dim=-1)
        keep = scores > 0.05  # Lower threshold for multi-scale
        
        return {
            'boxes': boxes[0][keep].cpu().numpy(),
            'scores': scores[keep].cpu().numpy(),
            'labels': labels[keep].cpu().numpy()
        }
    
    def _scale_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        resized_size: Tuple[int, int],
        original_size: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """Scale predictions from resized image back to original size"""
        boxes = predictions['boxes'].copy()  # [cx, cy, w, h] normalized
        
        resized_h, resized_w = resized_size
        orig_h, orig_w = original_size
        
        # Account for padding in resized image
        scale_factor = min(resized_h / orig_h, resized_w / orig_w)
        
        # Calculate padding
        pad_h = max(0, resized_h - int(orig_h * scale_factor)) // 2
        pad_w = max(0, resized_w - int(orig_w * scale_factor)) // 2
        
        # Convert to absolute coordinates in resized image
        boxes[:, 0] *= resized_w  # cx
        boxes[:, 1] *= resized_h  # cy
        boxes[:, 2] *= resized_w  # w
        boxes[:, 3] *= resized_h  # h
        
        # Convert to xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # Remove padding and scale
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_w) / scale_factor
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_h) / scale_factor
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_w) / scale_factor
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_h) / scale_factor
        
        # Clip to image bounds
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)
        
        # Convert back to cxcywh and normalize
        boxes_cxcywh = np.zeros_like(boxes_xyxy)
        boxes_cxcywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2 / orig_w  # cx
        boxes_cxcywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2 / orig_h  # cy
        boxes_cxcywh[:, 2] = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / orig_w  # w
        boxes_cxcywh[:, 3] = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / orig_h  # h
        
        return {
            'boxes': boxes_cxcywh,
            'scores': predictions['scores'],
            'labels': predictions['labels']
        }
    
    def _merge_predictions(
        self,
        predictions: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Merge predictions from different scales"""
        if self.fusion_method == 'weighted_box_fusion':
            return self._weighted_box_fusion(predictions)
        elif self.fusion_method == 'nms':
            return self._nms_merge(predictions)
        elif self.fusion_method == 'average':
            return self._average_merge(predictions)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _weighted_box_fusion(
        self,
        predictions: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Weighted Boxes Fusion"""
        # Implementation similar to TTA class
        # Omitted for brevity
        pass
    
    def _nms_merge(
        self,
        predictions: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Non-Maximum Suppression merge"""
        # Combine all predictions
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for pred in predictions:
            all_boxes.append(pred['boxes'])
            all_scores.append(pred['scores'])
            all_labels.append(pred['labels'])
        
        # Convert to list
        boxes_list = np.vstack(all_boxes)
        scores_list = np.concatenate(all_scores)
        labels_list = np.concatenate(all_labels)
        
        # Apply NMS
        keep = self._nms(boxes_list, scores_list)
        
        return {
            'boxes': boxes_list[keep],
            'scores': scores_list[keep],
            'labels': labels_list[keep]
        }
    
    def _average_merge(
        self,
        predictions: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Simple average merge"""
        # Average all predictions
        avg_boxes = np.mean([p['boxes'] for p in predictions], axis=0)
        avg_scores = np.mean([p['scores'] for p in predictions], axis=0)
        
        # Take labels from first prediction
        labels = predictions[0]['labels']
        
        return {
            'boxes': avg_boxes,
            'scores': avg_scores,
            'labels': labels
        }
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
        
        # Convert to xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        
        # Sort by score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Compute IoU with remaining boxes
            ious = self._compute_iou(boxes_xyxy[i], boxes_xyxy[order[1:]])
            
            # Keep boxes with IoU <= threshold
            inds = np.where(ious <= iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep, dtype=np.int32)
    
    def _compute_iou(self, box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and multiple boxes"""
        x1 = np.maximum(box1[0], boxes2[:, 0])
        y1 = np.maximum(box1[1], boxes2[:, 1])
        x2 = np.minimum(box1[2], boxes2[:, 2])
        y2 = np.minimum(box1[3], boxes2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-8)


class ScaleSelector:
    """
    Adaptive scale selection based on image characteristics
    """
    
    def __init__(
        self,
        min_scale: int = 768,
        max_scale: int = 1280,
        num_scales: int = 3,
        complexity_thresholds: List[float] = [0.3, 0.6]
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_scales = num_scales
        self.complexity_thresholds = complexity_thresholds
        
        # Generate scale options
        self.scales = np.linspace(min_scale, max_scale, num_scales).astype(int).tolist()
        
        print(f"Scale selector initialized with scales: {self.scales}")
    
    def select_scales(self, image: np.ndarray) -> List[int]:
        """
        Select appropriate scales based on image complexity
        """
        complexity = self._compute_complexity(image)
        
        if complexity < self.complexity_thresholds[0]:
            # Simple image - use smaller scales
            return self.scales[:1]
        elif complexity < self.complexity_thresholds[1]:
            # Medium complexity - use medium scales
            return self.scales[1:2]
        else:
            # Complex image - use all scales
            return self.scales
    
    def _compute_complexity(self, image: np.ndarray) -> float:
        """
        Compute image complexity score
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 1. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 2. Texture complexity (using variance)
        texture_complexity = np.var(gray)
        
        # 3. Color complexity (if color image)
        if len(image.shape) == 3:
            color_complexity = np.mean([np.var(image[:, :, i]) for i in range(3)])
        else:
            color_complexity = 0
        
        # Normalize
        edge_density_norm = min(edge_density / 0.1, 1.0)  # Assuming max 10% edge density
        texture_complexity_norm = min(texture_complexity / 1000, 1.0)
        color_complexity_norm = min(color_complexity / 1000, 1.0)
        
        # Combined complexity
        complexity = (
            0.5 * edge_density_norm +
            0.3 * texture_complexity_norm +
            0.2 * color_complexity_norm
        )
        
        return complexity
    
    def get_recommended_scales(
        self,
        image_size: Tuple[int, int],
        memory_constraints: Optional[Dict] = None
    ) -> List[int]:
        """
        Get recommended scales based on image size and memory constraints
        """
        h, w = image_size
        
        # Base scales on image size
        if max(h, w) < 800:
            # Small image - use smaller scales
            return [self.min_scale, (self.min_scale + self.max_scale) // 2]
        elif max(h, w) < 1200:
            # Medium image - use medium scales
            return [(self.min_scale + self.max_scale) // 2, self.max_scale]
        else:
            # Large image - use all scales
            return self.scales
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class TTAConfig:
    """
    Configuration for Test Time Augmentation
    """
    
    def __init__(
        self,
        scales: List[int] = [896, 1024, 1152, 1280],
        flips: List[str] = ['none', 'horizontal', 'vertical'],
        rotations: List[int] = [0, 90, 180, 270],
        merge_method: str = 'weighted_box_fusion',
        max_detections: int = 300,
        confidence_threshold: float = 0.1
    ):
        self.scales = scales
        self.flips = flips
        self.rotations = rotations
        self.merge_method = merge_method
        self.max_detections = max_detections
        self.confidence_threshold = confidence_threshold
        
        # Validate inputs
        valid_flips = ['none', 'horizontal', 'vertical', 'diagonal']
        for flip in flips:
            if flip not in valid_flips:
                raise ValueError(f"Invalid flip: {flip}. Must be one of {valid_flips}")
        
        valid_merge_methods = ['weighted_box_fusion', 'nms', 'soft_nms', 'average']
        if merge_method not in valid_merge_methods:
            raise ValueError(f"Invalid merge method: {merge_method}. Must be one of {valid_merge_methods}")


class TestTimeAugmentation:
    """
    Test Time Augmentation for cell detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TTAConfig,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device
        
        self.model.to(self.device)
        self.model.eval()
        
        # Create augmentation pipelines
        self.augmentations = self._create_augmentations()
        
        print(f"TTA initialized with:")
        print(f"  Scales: {config.scales}")
        print(f"  Flips: {config.flips}")
        print(f"  Rotations: {config.rotations}")
        print(f"  Merge method: {config.merge_method}")
    
    def _create_augmentations(self) -> Dict[str, List[Callable]]:
        """Create augmentation pipelines for each scale"""
        augmentations = {}
        
        for scale in self.config.scales:
            scale_augs = []
            
            for flip in self.config.flips:
                for rotation in self.config.rotations:
                    # Create transform
                    transform = self._create_transform(scale, flip, rotation)
                    scale_augs.append({
                        'transform': transform,
                        'flip': flip,
                        'rotation': rotation,
                        'scale': scale
                    })
            
            augmentations[scale] = scale_augs
        
        return augmentations
    
    def _create_transform(
        self,
        scale: int,
        flip: str,
        rotation: int
    ) -> A.Compose:
        """Create albumentations transform"""
        transforms = []
        
        # Resize
        transforms.append(A.LongestMaxSize(max_size=scale))
        
        # Pad
        transforms.append(A.PadIfNeeded(
            min_height=scale,
            min_width=scale,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ))
        
        # Flip
        if flip == 'horizontal':
            transforms.append(A.HorizontalFlip(p=1.0))
        elif flip == 'vertical':
            transforms.append(A.VerticalFlip(p=1.0))
        elif flip == 'diagonal':
            transforms.append(A.HorizontalFlip(p=1.0))
            transforms.append(A.VerticalFlip(p=1.0))
        
        # Rotate
        if rotation != 0:
            transforms.append(A.Rotate(
                limit=(rotation, rotation),
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0
            ))
        
        # Normalize and convert to tensor
        transforms.extend([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        original_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict with TTA
        
        Args:
            image: Input image (H, W, C) in RGB format
            original_size: Original image size (H, W) for scaling back
        
        Returns:
            Dictionary with predictions
        """
        if original_size is None:
            original_size = image.shape[:2]
        
        all_predictions = []
        
        # Run inference for each augmentation
        for scale, aug_list in self.augmentations.items():
            for aug_config in aug_list:
                # Apply augmentation
                augmented = aug_config['transform'](image=image)
                img_tensor = augmented['image'].unsqueeze(0).to(self.device)
                
                # Inference
                outputs = self.model(img_tensor)
                
                # Post-process
                predictions = self._post_process(outputs)
                
                if len(predictions['boxes']) > 0:
                    # Scale boxes back to original image
                    predictions = self._scale_predictions(
                        predictions,
                        aug_config,
                        original_size,
                        image.shape[:2]
                    )
                    
                    all_predictions.append(predictions)
        
        # Merge predictions
        if len(all_predictions) == 0:
            return {
                'boxes': np.zeros((0, 4), dtype=np.float32),
                'scores': np.zeros((0,), dtype=np.float32),
                'labels': np.zeros((0,), dtype=np.int32)
            }
        
        merged = self._merge_predictions(all_predictions)
        
        # Filter by confidence
        keep = merged['scores'] > self.config.confidence_threshold
        merged['boxes'] = merged['boxes'][keep]
        merged['scores'] = merged['scores'][keep]
        merged['labels'] = merged['labels'][keep]
        
        # Limit number of detections
        if len(merged['scores']) > self.config.max_detections:
            top_indices = np.argsort(merged['scores'])[::-1][:self.config.max_detections]
            merged['boxes'] = merged['boxes'][top_indices]
            merged['scores'] = merged['scores'][top_indices]
            merged['labels'] = merged['labels'][top_indices]
        
        return merged
    
    def _post_process(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Post-process model outputs"""
        logits = outputs['pred_logits'].softmax(-1)
        boxes = outputs['pred_boxes']
        
        # Filter by confidence
        scores, labels = logits[0][:, :-1].max(dim=-1)
        keep = scores > 0.05  # Lower threshold for TTA
        
        return {
            'boxes': boxes[0][keep].cpu().numpy(),
            'scores': scores[keep].cpu().numpy(),
            'labels': labels[keep].cpu().numpy()
        }
    
    def _scale_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        aug_config: Dict,
        original_size: Tuple[int, int],
        augmented_size: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """Scale predictions back to original image coordinates"""
        boxes = predictions['boxes'].copy()  # [cx, cy, w, h] in normalized coordinates
        
        # Convert to absolute coordinates in augmented image
        aug_h, aug_w = augmented_size
        boxes[:, 0] *= aug_w  # cx
        boxes[:, 1] *= aug_h  # cy
        boxes[:, 2] *= aug_w  # w
        boxes[:, 3] *= aug_h  # h
        
        # Convert to xyxy format
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # Reverse rotation
        if aug_config['rotation'] != 0:
            boxes_xyxy = self._reverse_rotation(
                boxes_xyxy,
                aug_config['rotation'],
                (aug_h, aug_w)
            )
        
        # Reverse flip
        if aug_config['flip'] != 'none':
            boxes_xyxy = self._reverse_flip(
                boxes_xyxy,
                aug_config['flip'],
                (aug_h, aug_w)
            )
        
        # Scale back to original size
        # Assuming LongestMaxSize was used
        orig_h, orig_w = original_size
        scale_factor = min(aug_h / orig_h, aug_w / orig_w)
        
        # Adjust for padding
        pad_h = max(0, aug_h - int(orig_h * scale_factor)) // 2
        pad_w = max(0, aug_w - int(orig_w * scale_factor)) // 2
        
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
        
        # Convert back to cxcywh format
        boxes_cxcywh = np.zeros_like(boxes_xyxy)
        boxes_cxcywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2  # cx
        boxes_cxcywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2  # cy
        boxes_cxcywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]  # w
        boxes_cxcywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]  # h
        
        # Normalize to [0, 1]
        boxes_cxcywh[:, 0] /= orig_w
        boxes_cxcywh[:, 1] /= orig_h
        boxes_cxcywh[:, 2] /= orig_w
        boxes_cxcywh[:, 3] /= orig_h
        
        return {
            'boxes': boxes_cxcywh,
            'scores': predictions['scores'],
            'labels': predictions['labels']
        }
    
    def _reverse_rotation(
        self,
        boxes: np.ndarray,
        rotation: int,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """Reverse rotation transformation"""
        h, w = image_size
        rotated = boxes.copy()
        
        if rotation == 90:
            # 90° clockwise -> reverse: 270° clockwise
            rotated[:, 0], rotated[:, 1] = h - boxes[:, 1], boxes[:, 0]
            rotated[:, 2], rotated[:, 3] = h - boxes[:, 3], boxes[:, 2]
        elif rotation == 180:
            # 180° -> reverse: 180°
            rotated[:, 0] = w - boxes[:, 0]
            rotated[:, 1] = h - boxes[:, 1]
            rotated[:, 2] = w - boxes[:, 2]
            rotated[:, 3] = h - boxes[:, 3]
        elif rotation == 270:
            # 270° clockwise -> reverse: 90° clockwise
            rotated[:, 0], rotated[:, 1] = boxes[:, 1], w - boxes[:, 0]
            rotated[:, 2], rotated[:, 3] = boxes[:, 3], w - boxes[:, 2]
        
        # Ensure x1 < x2, y1 < y2
        rotated[:, [0, 2]] = np.sort(rotated[:, [0, 2]], axis=1)
        rotated[:, [1, 3]] = np.sort(rotated[:, [1, 3]], axis=1)
        
        return rotated
    
    def _reverse_flip(
        self,
        boxes: np.ndarray,
        flip: str,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """Reverse flip transformation"""
        h, w = image_size
        flipped = boxes.copy()
        
        if flip == 'horizontal':
            flipped[:, 0] = w - boxes[:, 0]
            flipped[:, 2] = w - boxes[:, 2]
        elif flip == 'vertical':
            flipped[:, 1] = h - boxes[:, 1]
            flipped[:, 3] = h - boxes[:, 3]
        elif flip == 'diagonal':
            flipped[:, 0] = w - boxes[:, 0]
            flipped[:, 2] = w - boxes[:, 2]
            flipped[:, 1] = h - boxes[:, 1]
            flipped[:, 3] = h - boxes[:, 3]
        
        # Ensure x1 < x2, y1 < y2
        flipped[:, [0, 2]] = np.sort(flipped[:, [0, 2]], axis=1)
        flipped[:, [1, 3]] = np.sort(flipped[:, [1, 3]], axis=1)
        
        return flipped
    
    def _merge_predictions(
        self,
        predictions: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Merge predictions from different augmentations"""
        if self.config.merge_method == 'weighted_box_fusion':
            return self._weighted_box_fusion(predictions)
        elif self.config.merge_method == 'nms':
            return self._nms_merge(predictions)
        elif self.config.merge_method == 'soft_nms':
            return self._soft_nms_merge(predictions)
        elif self.config.merge_method == 'average':
            return self._average_merge(predictions)
        else:
            raise ValueError(f"Unknown merge method: {self.config.merge_method}")
    
    def _weighted_box_fusion(
        self,
        predictions: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Weighted Boxes Fusion"""
        # Combine all predictions
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for pred in predictions:
            all_boxes.append(pred['boxes'])
            all_scores.append(pred['scores'])
            all_labels.append(pred['labels'])
        
        # Convert to xyxy format for fusion
        all_boxes_xyxy = []
        for boxes in all_boxes:
            boxes_xyxy = np.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
            all_boxes_xyxy.append(boxes_xyxy)
        
        # Simple fusion: average boxes with similar IoU
        fused_boxes = []
        fused_scores = []
        fused_labels = []
        
        # Group by IoU
        iou_threshold = 0.5
        
        while len(all_boxes_xyxy) > 0:
            # Take first box as reference
            ref_box = all_boxes_xyxy[0][0]
            ref_score = all_scores[0][0]
            ref_label = all_labels[0][0]
            
            # Find matching boxes
            matching_boxes = [ref_box]
            matching_scores = [ref_score]
            
            # Remove from lists
            all_boxes_xyxy[0] = all_boxes_xyxy[0][1:]
            all_scores[0] = all_scores[0][1:]
            all_labels[0] = all_labels[0][1:]
            
            # Remove empty lists
            if len(all_boxes_xyxy[0]) == 0:
                all_boxes_xyxy.pop(0)
                all_scores.pop(0)
                all_labels.pop(0)
            
            # Search in other predictions
            for i in range(len(all_boxes_xyxy)):
                if len(all_boxes_xyxy[i]) == 0:
                    continue
                
                # Compute IoU with all boxes in this prediction
                ious = self._compute_iou(ref_box, all_boxes_xyxy[i])
                max_iou_idx = np.argmax(ious)
                max_iou = ious[max_iou_idx]
                
                if max_iou >= iou_threshold:
                    matching_boxes.append(all_boxes_xyxy[i][max_iou_idx])
                    matching_scores.append(all_scores[i][max_iou_idx])
                    
                    # Remove matched box
                    all_boxes_xyxy[i] = np.delete(all_boxes_xyxy[i], max_iou_idx, axis=0)
                    all_scores[i] = np.delete(all_scores[i], max_iou_idx)
                    all_labels[i] = np.delete(all_labels[i], max_iou_idx)
            
            # Fuse matching boxes
            if matching_boxes:
                fused_box = np.mean(matching_boxes, axis=0)
                fused_score = np.mean(matching_scores)
                
                fused_boxes.append(fused_box)
                fused_scores.append(fused_score)
                fused_labels.append(ref_label)
        
        # Convert back to cxcywh format
        fused_boxes_cxcywh = np.zeros((len(fused_boxes), 4))
        for i, box in enumerate(fused_boxes):
            fused_boxes_cxcywh[i, 0] = (box[0] + box[2]) / 2  # cx
            fused_boxes_cxcywh[i, 1] = (box[1] + box[3]) / 2  # cy
            fused_boxes_cxcywh[i, 2] = box[2] - box[0]  # w
            fused_boxes_cxcywh[i, 3] = box[3] - box[1]  # h
        
        return {
            'boxes': fused_boxes_cxcywh,
            'scores': np.array(fused_scores),
            'labels': np.array(fused_labels, dtype=np.int32)
        }
    
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
    
    def _soft_nms_merge(
        self,
        predictions: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Soft NMS merge"""
        # Similar to NMS but with soft suppression
        # Implementation omitted for brevity
        return self._nms_merge(predictions)
    
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
    
    def _compute_iou(self, box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and multiple boxes"""
        # box1: [x1, y1, x2, y2]
        # boxes2: N x [x1, y1, x2, y2]
        
        x1 = np.maximum(box1[0], boxes2[:, 0])
        y1 = np.maximum(box1[1], boxes2[:, 1])
        x2 = np.minimum(box1[2], boxes2[:, 2])
        y2 = np.minimum(box1[3], boxes2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-8)
    
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
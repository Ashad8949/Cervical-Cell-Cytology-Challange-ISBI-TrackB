import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import os
import json
from tqdm.auto import tqdm
import cv2

class ModelLoader:
    """
    Load multiple models for ensemble
    """
    
    def __init__(
        self,
        model_configs: List[Dict],
        device: str = 'cuda'
    ):
        self.model_configs = model_configs
        self.device = device
        
        self.models = []
        self.model_names = []
        
        print(f"Loading {len(model_configs)} models for ensemble...")
        
        for config in model_configs:
            model = self._load_model(config)
            self.models.append(model)
            self.model_names.append(config.get('name', f'model_{len(self.models)}'))
            
            print(f"  Loaded {config['name']} from {config['checkpoint']}")
    
    def _load_model(self, config: Dict) -> nn.Module:
        """Load a single model"""
        # Import model class based on config
        model_type = config.get('type', 'HybridCellDetector')
        
        if model_type == 'HybridCellDetector':
            from models.hybrid_models import HybridCellDetector
            model = HybridCellDetector(
                cnn_backbone_name=config.get('cnn_backbone', 'convnextv2_base'),
                transformer_backbone_name=config.get('transformer_backbone', 'path_dino'),
                detection_head_name=config.get('detection_head', 'dino_detr'),
                num_classes=config.get('num_classes', 1),
                num_queries=config.get('num_queries', 500),
                img_size=config.get('img_size', 1024),
                pretrained=False
            )
        elif model_type == 'PathDINO':
            from models.backbones import PathDINOBackbone
            from models.detection_heads import DINO_DETRHead
            
            # Create model
            backbone = PathDINOBackbone(
                model_name=config.get('backbone_name', 'vit_base_patch14_dinov2.lvd142m'),
                pretrained=False,
                img_size=config.get('img_size', 1024)
            )
            
            # Need to create full model - simplified
            model = nn.Sequential(
                backbone,
                DINO_DETRHead(
                    in_channels=256,
                    hidden_dim=256,
                    num_classes=config.get('num_classes', 1),
                    num_queries=config.get('num_queries', 300)
                )
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(config['checkpoint'], map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        
        # Move to device
        model.to(self.device)
        model.eval()
        
        return model
    
    def get_models(self) -> List[nn.Module]:
        """Get loaded models"""
        return self.models
    
    def get_model_names(self) -> List[str]:
        """Get model names"""
        return self.model_names


class EnsembleInference:
    """
    Ensemble inference with multiple models
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        model_names: List[str],
        weights: Optional[List[float]] = None,
        fusion_method: str = 'weighted_box_fusion',
        device: str = 'cuda',
        use_tta: bool = False,
        tta_config: Optional[Dict] = None
    ):
        self.models = models
        self.model_names = model_names
        self.fusion_method = fusion_method
        self.device = device
        self.use_tta = use_tta
        
        # Set weights
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]
        
        # Initialize TTA if enabled
        if use_tta:
            from .tta import TestTimeAugmentation, TTAConfig
            
            if tta_config is None:
                tta_config = TTAConfig()
            
            self.tta_engines = []
            for model in models:
                tta_engine = TestTimeAugmentation(model, tta_config, device)
                self.tta_engines.append(tta_engine)
        else:
            self.tta_engines = None
        
        print(f"Ensemble inference with {len(models)} models")
        print(f"  Fusion method: {fusion_method}")
        print(f"  Weights: {self.weights}")
        print(f"  TTA enabled: {use_tta}")
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        original_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Ensemble prediction
        """
        if original_size is None:
            original_size = image.shape[:2]
        
        all_predictions = []
        
        if self.use_tta and self.tta_engines:
            # Use TTA for each model
            for tta_engine, weight in zip(self.tta_engines, self.weights):
                predictions = tta_engine.predict(image, original_size)
                
                # Apply model weight
                if len(predictions['scores']) > 0:
                    predictions['scores'] *= weight
                    predictions['weight'] = weight
                    all_predictions.append(predictions)
        else:
            # Standard inference for each model
            for model, weight in zip(self.models, self.weights):
                predictions = self._predict_single(model, image, original_size)
                
                # Apply model weight
                if len(predictions['scores']) > 0:
                    predictions['scores'] *= weight
                    predictions['weight'] = weight
                    all_predictions.append(predictions)
        
        # Merge predictions
        if len(all_predictions) == 0:
            return {
                'boxes': np.zeros((0, 4), dtype=np.float32),
                'scores': np.zeros((0,), dtype=np.float32),
                'labels': np.zeros((0,), dtype=np.int32)
            }
        
        merged = self._merge_predictions(all_predictions)
        
        return merged
    
    def _predict_single(
        self,
        model: nn.Module,
        image: np.ndarray,
        original_size: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """Predict with single model"""
        # Preprocess
        img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        
        # Inference
        outputs = model(img_tensor)
        
        # Post-process
        predictions = self._post_process(outputs)
        
        # Scale to original size
        if len(predictions['boxes']) > 0:
            predictions = self._scale_predictions(
                predictions,
                image.shape[:2],
                original_size
            )
        
        return predictions
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image"""
        # Resize to model's expected size
        # (Assuming all models use same input size)
        target_size = 1024
        
        h, w = image.shape[:2]
        
        # Resize longer side to target_size
        if h > w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to float and normalize
        image_norm = padded.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image_norm = (image_norm - mean) / std
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).float()
        
        return image_tensor
    
    def _post_process(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Post-process model outputs"""
        # Handle different output formats
        if 'pred_logits' in outputs:
            logits = outputs['pred_logits'].softmax(-1)
            boxes = outputs['pred_boxes']
        elif 'logits' in outputs:
            logits = outputs['logits'].softmax(-1)
            boxes = outputs['boxes']
        else:
            raise ValueError("Unknown output format")
        
        # Filter by confidence
        scores, labels = logits[0][:, :-1].max(dim=-1)
        keep = scores > 0.1
        
        return {
            'boxes': boxes[0][keep].cpu().numpy(),
            'scores': scores[keep].cpu().numpy(),
            'labels': labels[keep].cpu().numpy()
        }
    
    def _scale_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        processed_size: Tuple[int, int],
        original_size: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """Scale predictions to original image size"""
        boxes = predictions['boxes'].copy()
        
        proc_h, proc_w = processed_size
        orig_h, orig_w = original_size
        
        # Scale from processed size to original size
        boxes[:, 0] *= orig_w / proc_w  # cx
        boxes[:, 1] *= orig_h / proc_h  # cy
        boxes[:, 2] *= orig_w / proc_w  # w
        boxes[:, 3] *= orig_h / proc_h  # h
        
        # Normalize to [0, 1] for consistency
        boxes[:, 0] /= orig_w
        boxes[:, 1] /= orig_h
        boxes[:, 2] /= orig_w
        boxes[:, 3] /= orig_h
        
        return {
            'boxes': boxes,
            'scores': predictions['scores'],
            'labels': predictions['labels']
        }
    
    def _merge_predictions(
        self,
        predictions: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Merge predictions from different models"""
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
        # Convert all boxes to xyxy format
        all_boxes_xyxy = []
        all_scores = []
        all_labels = []
        all_weights = []
        
        for pred in predictions:
            boxes = pred['boxes']
            weight = pred.get('weight', 1.0)
            
            # Convert to xyxy
            boxes_xyxy = np.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
            
            all_boxes_xyxy.append(boxes_xyxy)
            all_scores.append(pred['scores'])
            all_labels.append(pred['labels'])
            all_weights.append(np.full(len(boxes), weight))
        
        # Flatten
        boxes_flat = np.vstack(all_boxes_xyxy)
        scores_flat = np.concatenate(all_scores)
        labels_flat = np.concatenate(all_labels)
        weights_flat = np.concatenate(all_weights)
        
        # Apply NMS first to reduce duplicates
        keep = self._nms(boxes_flat, scores_flat, iou_threshold=0.5)
        
        boxes_flat = boxes_flat[keep]
        scores_flat = scores_flat[keep]
        labels_flat = labels_flat[keep]
        weights_flat = weights_flat[keep]
        
        # Group by IoU and average
        fused_boxes = []
        fused_scores = []
        fused_labels = []
        
        iou_threshold = 0.5
        
        while len(boxes_flat) > 0:
            # Take first box as reference
            ref_box = boxes_flat[0]
            ref_score = scores_flat[0]
            ref_label = labels_flat[0]
            ref_weight = weights_flat[0]
            
            # Find matching boxes
            matching_boxes = [ref_box]
            matching_scores = [ref_score * ref_weight]
            matching_weights = [ref_weight]
            
            # Compute IoU with remaining boxes
            if len(boxes_flat) > 1:
                ious = self._compute_iou(ref_box, boxes_flat[1:])
                
                # Find matches
                matches = np.where(ious >= iou_threshold)[0] + 1
                
                for idx in matches:
                    matching_boxes.append(boxes_flat[idx])
                    matching_scores.append(scores_flat[idx] * weights_flat[idx])
                    matching_weights.append(weights_flat[idx])
                
                # Remove matched boxes
                mask = np.ones(len(boxes_flat), dtype=bool)
                mask[0] = False
                mask[matches] = False
                
                boxes_flat = boxes_flat[mask]
                scores_flat = scores_flat[mask]
                labels_flat = labels_flat[mask]
                weights_flat = weights_flat[mask]
            else:
                boxes_flat = boxes_flat[1:]
                scores_flat = scores_flat[1:]
                labels_flat = labels_flat[1:]
                weights_flat = weights_flat[1:]
            
            # Fuse matching boxes
            if matching_boxes:
                # Weighted average
                total_weight = sum(matching_weights)
                weighted_boxes = np.sum(
                    [box * weight for box, weight in zip(matching_boxes, matching_weights)],
                    axis=0
                ) / total_weight
                
                fused_score = sum(matching_scores) / total_weight
                
                fused_boxes.append(weighted_boxes)
                fused_scores.append(fused_score)
                fused_labels.append(ref_label)
        
        # Convert back to cxcywh format
        fused_boxes_cxcywh = np.zeros((len(fused_boxes), 4))
        for i, box in enumerate(fused_boxes):
            fused_boxes_cxcywh[i, 0] = (box[0] + box[2]) / 2  # cx
            fused_boxes_cxcywh[i, 1] = (box[1] + box[3]) / 2  # cy
            fused_boxes_cxcywh[i, 2] = box[2] - box[0]  # w
            fused_boxes_cxcywh[i, 3] = box[3] - box[1]  # h
        
        # Normalize to [0, 1]
        # (Assuming original image size for normalization)
        
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
            weight = pred.get('weight', 1.0)
            
            # Apply weight to scores
            weighted_scores = pred['scores'] * weight
            
            all_boxes.append(pred['boxes'])
            all_scores.append(weighted_scores)
            all_labels.append(pred['labels'])
        
        # Flatten
        boxes_flat = np.vstack(all_boxes)
        scores_flat = np.concatenate(all_scores)
        labels_flat = np.concatenate(all_labels)
        
        # Apply NMS
        keep = self._nms(boxes_flat, scores_flat)
        
        return {
            'boxes': boxes_flat[keep],
            'scores': scores_flat[keep],
            'labels': labels_flat[keep]
        }
    
    def _average_merge(
        self,
        predictions: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Simple average merge"""
        # Weighted average
        total_weight = sum(p.get('weight', 1.0) for p in predictions)
        
        avg_boxes = np.sum(
            [p['boxes'] * p.get('weight', 1.0) for p in predictions],
            axis=0
        ) / total_weight
        
        avg_scores = np.sum(
            [p['scores'] * p.get('weight', 1.0) for p in predictions],
            axis=0
        ) / total_weight
        
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
            if len(order) > 1:
                ious = self._compute_iou(boxes_xyxy[i], boxes_xyxy[order[1:]])
                
                # Keep boxes with IoU <= threshold
                inds = np.where(ious <= iou_threshold)[0]
                order = order[inds + 1]
            else:
                break
        
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
    
    def predict_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 4,
        show_progress: bool = True
    ) -> List[Dict[str, np.ndarray]]:
        """
        Predict on batch of images
        """
        all_predictions = []
        
        if show_progress:
            pbar = tqdm(total=len(images), desc="Ensemble inference")
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            for image in batch:
                predictions = self.predict(image)
                all_predictions.append(predictions)
                
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        return all_predictions
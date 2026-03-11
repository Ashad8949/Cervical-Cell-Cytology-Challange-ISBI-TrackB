import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
from scipy.optimize import linear_sum_assignment

class DETRLoss(nn.Module):
    """
    DETR loss function with Hungarian matching
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        matcher: str = "hungarian",
        weight_dict: Optional[Dict[str, float]] = None,
        eos_coef: float = 0.1,
        losses: List[str] = ["labels", "boxes", "cardinality"],
        focal_alpha: float = 0.25,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.focal_alpha = focal_alpha
        
        # Weight dictionary
        if weight_dict is None:
            weight_dict = {
                "loss_ce": 1,
                "loss_bbox": 5,
                "loss_giou": 2,
            }
        self.weight_dict = weight_dict
        
        # Matcher
        if matcher == "hungarian":
            self.matcher = HungarianMatcher(
                cost_class=weight_dict.get("loss_ce", 1),
                cost_bbox=weight_dict.get("loss_bbox", 5),
                cost_giou=weight_dict.get("loss_giou", 2),
            )
        
        # Loss functions
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef  # Background weight
        self.register_buffer('empty_weight', empty_weight)
        
        # Classification loss (focal loss)
        self.loss_ce = FocalLoss(
            alpha=focal_alpha,
            gamma=2.0,
            reduction='none'
        )
        
        # Box regression losses
        self.loss_bbox = nn.L1Loss(reduction='none')
        self.loss_giou = GIoULoss(reduction='none')
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Loss computation
        """
        # Retrieve the matching between predictions and targets
        indices = self.matcher(outputs, targets)
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))
        
        # Apply weights
        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]
        
        return losses
    
    def get_loss(
        self,
        loss: str,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Get the specific loss
        """
        if loss == 'labels':
            return self.loss_labels(outputs, targets, indices)
        elif loss == 'boxes':
            return self.loss_boxes(outputs, targets, indices)
        elif loss == 'cardinality':
            return self.loss_cardinality(outputs, targets, indices)
        else:
            raise ValueError(f"Unknown loss: {loss}")
    
    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Classification loss
        """
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([
            t["labels"][J] for t, (_, J) in zip(targets, indices)
        ])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        
        # Focal loss
        loss_ce = self.loss_ce(
            src_logits.transpose(1, 2),
            target_classes
        )
        
        # Apply class weights
        loss_ce = loss_ce * self.empty_weight[target_classes]
        loss_ce = loss_ce.sum() / self.empty_weight[target_classes].sum()
        
        return {'loss_ce': loss_ce}
    
    def loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Bounding box loss (L1 + GIoU)
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / len(target_boxes)
        
        # GIoU loss
        loss_giou = 1 - torch.diag(self.generalized_box_iou(
            self.box_cxcywh_to_xyxy(src_boxes),
            self.box_cxcywh_to_xyxy(target_boxes)
        ))
        loss_giou = loss_giou.mean()
        
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}
    
    def loss_cardinality(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Cardinality error (not a loss, just for logging)
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        
        # Count the number of predictions that are not "no-object"
        card_pred = (pred_logits.argmax(-1) != self.num_classes).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        
        return {'cardinality_error': card_err}
    
    def _get_src_permutation_idx(self, indices):
        """
        Get source indices for permutation
        """
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices):
        """
        Get target indices for permutation
        """
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
        ])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    @staticmethod
    def box_cxcywh_to_xyxy(x):
        """Convert from [cx, cy, w, h] to [x1, y1, x2, y2]"""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    
    @staticmethod
    def generalized_box_iou(boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/
        """
        # Ensure boxes are in [x1, y1, x2, y2] format
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        
        # Intersection
        inter_min = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        inter_max = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        inter_wh = (inter_max - inter_min).clamp(min=0)
        inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]
        
        # Union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - inter
        
        # IoU
        iou = inter / union
        
        # Enclosing box
        enclose_min = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        enclose_max = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        enclose_wh = (enclose_max - enclose_min).clamp(min=0)
        enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]
        
        # GIoU
        giou = iou - (enclose_area - union) / enclose_area
        
        return giou


class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for DETR
    """
    
    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 5,
        cost_giou: float = 2,
    ):
        super().__init__()
        
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, \
            "all costs cant be 0"
    
    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform the matching
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten predictions
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        # Prepare targets
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # Compute classification cost
        cost_class = -out_prob[:, tgt_ids]
        
        # Compute L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute giou cost
        cost_giou = -self.generalized_box_iou(
            self.box_cxcywh_to_xyxy(out_bbox),
            self.box_cxcywh_to_xyxy(tgt_bbox)
        )
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        # Hungarian matching
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64)
            )
            for i, j in indices
        ]
    
    @staticmethod
    def box_cxcywh_to_xyxy(x):
        """Convert from [cx, cy, w, h] to [x1, y1, x2, y2]"""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    
    @staticmethod
    def generalized_box_iou(boxes1, boxes2):
        """
        Generalized IoU
        """
        # Ensure boxes are in [x1, y1, x2, y2] format
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        
        # Intersection
        inter_min = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        inter_max = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        inter_wh = (inter_max - inter_min).clamp(min=0)
        inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]
        
        # Union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - inter
        
        # IoU
        iou = inter / (union + 1e-8)
        
        # Enclosing box
        enclose_min = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        enclose_max = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        enclose_wh = (enclose_max - enclose_min).clamp(min=0)
        enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]
        
        # GIoU
        giou = iou - (enclose_area - union) / (enclose_area + 1e-8)
        
        return giou


class FocalLoss(nn.Module):
    """
    Focal loss for classification
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Forward pass
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Forward pass
        """
        giou = self.generalized_box_iou(pred, target)
        loss = 1 - giou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    @staticmethod
    def generalized_box_iou(boxes1, boxes2):
        """
        Generalized IoU for pairs of boxes
        boxes1, boxes2: (N, 4) in [x1, y1, x2, y2] format
        Returns: (N,) tensor of GIoU values
        """
        # Intersection
        inter_min = torch.max(boxes1[:, :2], boxes2[:, :2])
        inter_max = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        inter_wh = (inter_max - inter_min).clamp(min=0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        
        # Union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - inter
        
        # IoU
        iou = inter / (union + 1e-8)
        
        # Enclosing box
        enclose_min = torch.min(boxes1[:, :2], boxes2[:, :2])
        enclose_max = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        enclose_wh = (enclose_max - enclose_min).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
        
        # GIoU
        giou = iou - (enclose_area - union) / (enclose_area + 1e-8)
        
        return giou


class CellDetectionLoss(nn.Module):
    """
    Custom loss for cell detection with additional constraints
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        size_weight: float = 0.1,
        density_weight: float = 0.05,
        overlap_weight: float = 0.1,
        consistency_weight: float = 0.05,
    ):
        super().__init__()
        
        self.base_loss = base_loss
        self.size_weight = size_weight
        self.density_weight = density_weight
        self.overlap_weight = overlap_weight
        self.consistency_weight = consistency_weight
        
        # Additional losses
        self.size_loss = SizeConstraintLoss()
        self.density_loss = DensityConstraintLoss()
        self.overlap_loss = OverlapConstraintLoss()
        self.consistency_loss = ConsistencyLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        image_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with additional constraints
        """
        # Base DETR loss
        losses = self.base_loss(outputs, targets)
        
        # Extract predictions
        pred_boxes = outputs['pred_boxes']
        pred_logits = outputs['pred_logits']
        
        # Size constraint
        size_loss = self.size_loss(pred_boxes)
        losses['loss_size'] = size_loss * self.size_weight
        
        # Density constraint
        if image_sizes is not None:
            density_loss = self.density_loss(pred_boxes, pred_logits, image_sizes)
            losses['loss_density'] = density_loss * self.density_weight
        
        # Overlap constraint
        overlap_loss = self.overlap_loss(pred_boxes)
        losses['loss_overlap'] = overlap_loss * self.overlap_weight
        
        # Multi-scale consistency
        if 'aux_outputs' in outputs:
            consistency_loss = self.consistency_loss(
                outputs['pred_boxes'],
                outputs['aux_outputs']
            )
            losses['loss_consistency'] = consistency_loss * self.consistency_weight
        
        return losses


class SizeConstraintLoss(nn.Module):
    """
    Constrain cell sizes to biologically plausible ranges
    """
    
    def __init__(self, min_size: float = 0.001, max_size: float = 0.5):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
    
    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Penalize boxes that are too small or too large
        """
        # Extract width and height
        widths = boxes[..., 2]  # w
        heights = boxes[..., 3]  # h
        
        # Penalty for too small
        small_penalty = F.relu(self.min_size - widths) + F.relu(self.min_size - heights)
        
        # Penalty for too large
        large_penalty = F.relu(widths - self.max_size) + F.relu(heights - self.max_size)
        
        return (small_penalty + large_penalty).mean()


class DensityConstraintLoss(nn.Module):
    """
    Constrain cell density to plausible ranges
    """
    
    def __init__(self, max_density: float = 0.3):
        super().__init__()
        self.max_density = max_density
    
    def forward(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        image_sizes: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Penalize predictions that are too dense
        """
        bs = boxes.shape[0]
        total_loss = 0
        
        for i in range(bs):
            # Get boxes and scores for this image
            img_boxes = boxes[i]
            img_scores = scores[i]
            
            # Filter by confidence
            conf_mask = img_scores.max(dim=-1)[0] > 0.3
            if not conf_mask.any():
                continue
            
            conf_boxes = img_boxes[conf_mask]
            
            # Compute total area of predicted cells
            areas = conf_boxes[:, 2] * conf_boxes[:, 3]  # w * h
            total_area = areas.sum()
            
            # Image area
            img_h, img_w = image_sizes[i]
            img_area = img_h * img_w
            
            # Density
            density = total_area / img_area
            
            # Penalty if density exceeds threshold
            penalty = F.relu(density - self.max_density)
            total_loss += penalty
        
        return total_loss / bs if bs > 0 else torch.tensor(0.0)


class OverlapConstraintLoss(nn.Module):
    """
    Penalize excessive overlap between predicted cells
    """
    
    def __init__(self, max_iou: float = 0.7):
        super().__init__()
        self.max_iou = max_iou
    
    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute overlap penalty
        """
        bs, num_queries, _ = boxes.shape
        total_loss = 0
        
        for i in range(bs):
            img_boxes = boxes[i]
            
            # Compute pairwise IoU
            iou_matrix = self.pairwise_iou(img_boxes, img_boxes)
            
            # Remove self-comparisons
            iou_matrix = iou_matrix - torch.eye(num_queries, device=boxes.device)
            
            # Penalize high IoU
            penalty = F.relu(iou_matrix - self.max_iou)
            total_loss += penalty.sum() / (num_queries * (num_queries - 1))
        
        return total_loss / bs if bs > 0 else torch.tensor(0.0)
    
    @staticmethod
    def pairwise_iou(boxes1, boxes2):
        """
        Compute pairwise IoU between two sets of boxes
        """
        # Convert to xyxy
        boxes1_xyxy = DETRLoss.box_cxcywh_to_xyxy(boxes1)
        boxes2_xyxy = DETRLoss.box_cxcywh_to_xyxy(boxes2)
        
        # Compute intersection
        lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[:, :2])
        rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        # Compute union
        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
        union = area1[:, None] + area2 - inter
        
        # IoU
        iou = inter / (union + 1e-8)
        
        return iou


class ConsistencyLoss(nn.Module):
    """
    Ensure consistency between different decoder layers
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        final_boxes: torch.Tensor,
        aux_outputs: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute consistency loss between final and auxiliary predictions
        """
        if not aux_outputs:
            return torch.tensor(0.0, device=final_boxes.device)
        
        total_loss = 0
        
        for aux in aux_outputs:
            aux_boxes = aux['pred_boxes']
            
            # L2 distance between predictions
            loss = F.mse_loss(final_boxes, aux_boxes, reduction='mean')
            total_loss += loss
        
        return total_loss / len(aux_outputs)


class DenoisingLoss(nn.Module):
    """
    Loss for denoising training in DINO
    """
    
    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        denoising_outputs: Dict[str, torch.Tensor],
        noisy_targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute denoising loss for DINO training
        Reconstruction loss between denoised predictions and noisy input targets
        """
        if denoising_outputs is None or len(noisy_targets) == 0:
            return {'loss_denoising': torch.tensor(0.0)}
        
        pred_boxes = denoising_outputs.get('pred_boxes', None)
        pred_logits = denoising_outputs.get('pred_logits', None)
        
        if pred_boxes is None:
            return {'loss_denoising': torch.tensor(0.0)}
        
        # Collect target boxes
        target_boxes = torch.cat([t['boxes'] for t in noisy_targets], dim=0)
        target_labels = torch.cat([t['labels'] for t in noisy_targets], dim=0)
        
        # Flatten predictions to match targets
        pred_boxes_flat = pred_boxes.flatten(0, 1)[:len(target_boxes)]
        
        # L1 loss on boxes
        loss_bbox = F.l1_loss(pred_boxes_flat, target_boxes, reduction='mean')
        
        # Classification loss if logits available
        loss_cls = torch.tensor(0.0, device=pred_boxes.device)
        if pred_logits is not None:
            pred_logits_flat = pred_logits.flatten(0, 1)[:len(target_labels)]
            loss_cls = F.cross_entropy(pred_logits_flat, target_labels, reduction='mean')
        
        total_loss = (loss_bbox + loss_cls) * self.weight
        
        return {'loss_denoising': total_loss}
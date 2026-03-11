import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Optional, Callable, Any
import numpy as np
from tqdm.auto import tqdm

class CurriculumTrainer:
    """
    Curriculum learning for cell detection
    Progressively increases difficulty
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        difficulty_fn: Callable,
        stages: List[Dict[str, Any]],
        device: str = 'cuda'
    ):
        self.model = model
        self.dataset = dataset
        self.difficulty_fn = difficulty_fn
        self.stages = stages
        self.device = device
        self.current_stage = 0
        
        # Compute difficulty scores for all samples
        self.difficulty_scores = self._compute_difficulty_scores()
        
        print(f"Curriculum learning with {len(stages)} stages")
    
    def _compute_difficulty_scores(self) -> np.ndarray:
        """Compute difficulty scores for all samples"""
        scores = []
        
        print("Computing difficulty scores...")
        for i in tqdm(range(len(self.dataset))):
            # Get sample
            _, target = self.dataset[i]
            
            # Compute difficulty based on:
            # 1. Number of cells
            # 2. Cell sizes
            # 3. Overlap between cells
            # 4. Cell density
            
            num_cells = len(target['boxes'])
            
            if num_cells == 0:
                difficulty = 0.0
            else:
                # Average cell size
                boxes = target['boxes']
                areas = boxes[:, 2] * boxes[:, 3]  # w * h
                avg_size = areas.mean().item()
                
                # Overlap (approximate)
                overlap_score = self._compute_overlap_score(boxes)
                
                # Density
                img_area = target['size'][0] * target['size'][1]
                density = num_cells * avg_size / img_area
                
                # Combined difficulty score
                difficulty = (
                    0.3 * min(num_cells / 100, 1.0) +  # Normalized cell count
                    0.2 * (1 - min(avg_size / 0.5, 1.0)) +  # Smaller cells are harder
                    0.3 * overlap_score +  # Overlap makes it harder
                    0.2 * min(density / 0.5, 1.0)  # Density
                )
            
            scores.append(difficulty)
        
        return np.array(scores)
    
    def _compute_overlap_score(self, boxes: torch.Tensor) -> float:
        """Compute overlap score between boxes"""
        if len(boxes) < 2:
            return 0.0
        
        # Convert to xyxy
        boxes_xyxy = self._cxcywh_to_xyxy(boxes)
        
        # Compute pairwise IoU
        ious = self._pairwise_iou(boxes_xyxy)
        
        # Average IoU (excluding self)
        mask = ~torch.eye(len(boxes), dtype=torch.bool, device=boxes.device)
        avg_iou = ious[mask].mean().item() if mask.any() else 0.0
        
        return avg_iou
    
    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    @staticmethod
    def _pairwise_iou(boxes: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU between boxes"""
        n = boxes.shape[0]
        ious = torch.zeros((n, n), device=boxes.device)
        
        for i in range(n):
            box1 = boxes[i]
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            
            for j in range(n):
                if i == j:
                    continue
                
                box2 = boxes[j]
                
                # Intersection
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                
                if x2 <= x1 or y2 <= y1:
                    intersection = 0
                else:
                    intersection = (x2 - x1) * (y2 - y1)
                
                # Union
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union = area1 + area2 - intersection
                
                ious[i, j] = intersection / union if union > 0 else 0
        
        return ious
    
    def get_current_dataset(self) -> Dataset:
        """Get dataset for current stage"""
        stage_config = self.stages[self.current_stage]
        max_difficulty = stage_config.get('max_difficulty', 1.0)
        min_difficulty = stage_config.get('min_difficulty', 0.0)
        
        # Filter samples by difficulty
        mask = (self.difficulty_scores >= min_difficulty) & \
               (self.difficulty_scores <= max_difficulty)
        
        indices = np.where(mask)[0]
        
        # Create subset dataset
        from torch.utils.data import Subset
        subset = Subset(self.dataset, indices)
        
        print(f"Stage {self.current_stage + 1}: Using {len(indices)} samples "
              f"(difficulty range: {min_difficulty:.2f}-{max_difficulty:.2f})")
        
        return subset
    
    def advance_stage(self) -> bool:
        """Advance to next stage if criteria met"""
        if self.current_stage >= len(self.stages) - 1:
            return False
        
        # Check if criteria for advancing are met
        # This could be based on validation performance, epochs, etc.
        # For simplicity, we'll advance based on epochs
        stage_config = self.stages[self.current_stage]
        current_epochs = stage_config.get('current_epochs', 0)
        required_epochs = stage_config.get('epochs', 10)
        
        if current_epochs >= required_epochs:
            self.current_stage += 1
            print(f"Advancing to stage {self.current_stage + 1}")
            return True
        
        return False
    
    def update_stage_epochs(self):
        """Update epoch count for current stage"""
        self.stages[self.current_stage]['current_epochs'] = \
            self.stages[self.current_stage].get('current_epochs', 0) + 1


class ProgressiveResizing:
    """
    Progressive resizing training
    Gradually increase image resolution
    """
    
    def __init__(
        self,
        resolutions: List[int],
        epochs_per_resolution: List[int],
        base_transform: Callable,
        device: str = 'cuda'
    ):
        self.resolutions = resolutions
        self.epochs_per_resolution = epochs_per_resolution
        self.base_transform = base_transform
        self.device = device
        
        self.current_resolution_idx = 0
        self.current_epoch = 0
        
        print(f"Progressive resizing with resolutions: {resolutions}")
        print(f"Epochs per resolution: {epochs_per_resolution}")
    
    def get_current_resolution(self) -> int:
        """Get current target resolution"""
        return self.resolutions[self.current_resolution_idx]
    
    def get_transform(self) -> Callable:
        """Get transform for current resolution"""
        current_res = self.get_current_resolution()
        
        # Modify base transform to use current resolution
        # This depends on how your transforms are structured
        # For simplicity, we'll assume base_transform can be configured
        return self.base_transform(image_size=current_res)
    
    def update_epoch(self, metrics: Dict[str, float]) -> bool:
        """
        Update epoch count and check if should advance resolution
        
        Returns:
            bool: Whether resolution was changed
        """
        self.current_epoch += 1
        
        # Check if should advance resolution
        required_epochs = self.epochs_per_resolution[self.current_resolution_idx]
        
        if self.current_epoch >= required_epochs:
            # Check if metrics are good enough to advance
            if self._should_advance(metrics):
                return self.advance_resolution()
            else:
                # Continue with current resolution
                print(f"Metrics not good enough to advance resolution. Continuing...")
                self.current_epoch = 0  # Reset counter
                return False
        
        return False
    
    def advance_resolution(self) -> bool:
        """Advance to next resolution"""
        if self.current_resolution_idx >= len(self.resolutions) - 1:
            return False
        
        self.current_resolution_idx += 1
        self.current_epoch = 0
        
        print(f"Advancing to resolution: {self.get_current_resolution()}")
        return True
    
    def _should_advance(self, metrics: Dict[str, float]) -> bool:
        """Determine if should advance based on metrics"""
        # Example criteria:
        # 1. Validation loss has plateaued
        # 2. mAP is above threshold
        # 3. Training loss has converged
        
        mAP = metrics.get('mAP', 0)
        val_loss = metrics.get('val_loss', float('inf'))
        
        # Simple criteria: mAP > 0.5 and val_loss < 1.0
        return mAP > 0.5 and val_loss < 1.0
    
    def state_dict(self) -> Dict:
        """Get state dict for checkpointing"""
        return {
            'current_resolution_idx': self.current_resolution_idx,
            'current_epoch': self.current_epoch,
            'resolutions': self.resolutions,
            'epochs_per_resolution': self.epochs_per_resolution
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state dict from checkpoint"""
        self.current_resolution_idx = state_dict['current_resolution_idx']
        self.current_epoch = state_dict['current_epoch']
        self.resolutions = state_dict['resolutions']
        self.epochs_per_resolution = state_dict['epochs_per_resolution']


class SelfTraining:
    """
    Self-training with pseudo-labels
    """
    
    def __init__(
        self,
        model: nn.Module,
        labeled_dataset: Dataset,
        unlabeled_dataset: Dataset,
        confidence_threshold: float = 0.7,
        device: str = 'cuda'
    ):
        self.model = model
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        self.pseudo_labels = []
    
    def generate_pseudo_labels(self) -> Dataset:
        """Generate pseudo-labels for unlabeled data"""
        self.model.eval()
        
        print("Generating pseudo-labels...")
        
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm
        
        dataloader = DataLoader(
            self.unlabeled_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4
        )
        
        pseudo_labeled_data = []
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Post-process
                predictions = self._post_process(outputs)
                
                # Filter by confidence
                for i, pred in enumerate(predictions):
                    if len(pred['boxes']) > 0:
                        # Average confidence
                        avg_confidence = pred['scores'].mean()
                        
                        if avg_confidence > self.confidence_threshold:
                            # Add to pseudo-labeled data
                            pseudo_labeled_data.append({
                                'image': images[i].cpu(),
                                'boxes': torch.tensor(pred['boxes']),
                                'labels': torch.tensor(pred['labels'])
                            })
        
        print(f"Generated {len(pseudo_labeled_data)} pseudo-labeled samples")
        
        # Create dataset from pseudo-labels
        from torch.utils.data import Dataset as TorchDataset
        
        class PseudoLabeledDataset(TorchDataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                return item['image'], {
                    'boxes': item['boxes'],
                    'labels': item['labels']
                }
        
        return PseudoLabeledDataset(pseudo_labeled_data)
    
    def _post_process(self, outputs: Dict[str, torch.Tensor]) -> List[Dict]:
        """Post-process model outputs"""
        predictions = []
        
        logits = outputs['pred_logits'].softmax(-1)
        boxes = outputs['pred_boxes']
        
        batch_size = logits.shape[0]
        
        for i in range(batch_size):
            scores, labels = logits[i][:, :-1].max(dim=-1)
            keep = scores > 0.1  # Lower threshold for pseudo-labeling
            
            prediction = {
                'boxes': boxes[i][keep].cpu().numpy(),
                'scores': scores[keep].cpu().numpy(),
                'labels': labels[keep].cpu().numpy()
            }
            predictions.append(prediction)
        
        return predictions
    
    def combine_datasets(self) -> Dataset:
        """Combine labeled and pseudo-labeled datasets"""
        pseudo_dataset = self.generate_pseudo_labels()
        
        # Combine datasets
        from torch.utils.data import ConcatDataset
        combined = ConcatDataset([self.labeled_dataset, pseudo_dataset])
        
        print(f"Combined dataset size: {len(combined)} "
              f"({len(self.labeled_dataset)} labeled + {len(pseudo_dataset)} pseudo-labeled)")
        
        return combined
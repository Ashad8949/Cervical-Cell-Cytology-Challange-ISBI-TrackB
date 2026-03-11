import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RIVADataset(Dataset):
    """
    Dataset for RIVA Cervical Cytology Challenge - Track B
    Handles loading of images and annotations with proper formatting
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transforms: Optional[A.Compose] = None,
        is_train: bool = True,
        image_size: int = 1024,
        use_cache: bool = False
    ):
        """
        Args:
            df: DataFrame containing annotations
            image_dir: Directory containing images
            transforms: Albumentations transforms
            is_train: Whether in training mode
            image_size: Target image size
            use_cache: Cache images in memory
        """
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.is_train = is_train
        self.image_size = image_size
        self.use_cache = use_cache
        
        # Group by image
        self.image_groups = self._group_by_image()
        self.image_ids = list(self.image_groups.keys())
        
        # Cache
        self.image_cache = {}
        self.metadata_cache = {}
        
        print(f"Dataset initialized with {len(self.image_ids)} images")
        print(f"Total annotations: {len(self.df)}")
        
    def _group_by_image(self) -> Dict[str, pd.DataFrame]:
        """Group annotations by image filename"""
        groups = {}
        for img_name, group_df in self.df.groupby('image_filename'):
            groups[img_name] = group_df
        return groups
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_name = self.image_ids[idx]
        
        # Load image (from cache or disk)
        image = self._load_image(img_name)
        original_h, original_w = image.shape[:2]
        
        # Get annotations for this image
        annotations = self.image_groups[img_name]
        
        # Convert annotations to boxes
        boxes, labels = self._parse_annotations(annotations, original_w, original_h)
        
        # Apply transforms
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
            
        # Convert to [cx, cy, w, h] normalized format
        # shape is (C, H, W)
        img_h, img_w = image.shape[1], image.shape[2]
        
        normalized_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Calculate center, width, height
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            
            # Normalize to [0, 1]
            normalized_boxes.append([
                cx / img_w,
                cy / img_h,
                w / img_w,
                h / img_h
            ])
            
        boxes = normalized_boxes
        
        # Convert to tensors
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([original_h, original_w]),
            'size': torch.tensor(image.shape[1:]),  # H, W
            'image_name': img_name
        }
        
        return image, target
    
    def _load_image(self, img_name: str) -> np.ndarray:
        """Load image from disk with caching"""
        if self.use_cache and img_name in self.image_cache:
            return self.image_cache[img_name]
        
        # Search for image in train/val/test directories
        possible_paths = [
            os.path.join(self.image_dir, 'train', img_name),
            os.path.join(self.image_dir, 'val', img_name),
            os.path.join(self.image_dir, 'test', img_name),
            os.path.join(self.image_dir, img_name)
        ]
        
        for img_path in possible_paths:
            if os.path.exists(img_path):
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError(f"Failed to load image: {img_path}")
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply preprocessing
                image = self._preprocess_image(image)
                
                if self.use_cache:
                    self.image_cache[img_name] = image
                
                return image
        
        raise FileNotFoundError(f"Image not found: {img_name}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply image preprocessing"""
        # Keep as uint8 [0, 255] — albumentations A.Normalize handles
        # the conversion to float32 and ImageNet normalization
        # (A.Normalize expects [0,255] input by default with max_pixel_value=255)
        
        # Optional: stain normalization
        # image = stain_normalization(image)
        
        return image
    
    def _parse_annotations(
        self, 
        annotations: pd.DataFrame, 
        image_w: int, 
        image_h: int
    ) -> Tuple[List[List[float]], List[int]]:
        """Convert dataframe annotations to bounding boxes"""
        boxes = []
        labels = []
        
        for _, row in annotations.iterrows():
            # Center format: x, y are center coordinates; width, height are box dimensions
            cx = row['x']
            cy = row['y']
            width = row['width']
            height = row['height']
            
            # Convert center format to x1, y1, x2, y2 (pascal_voc)
            x1 = cx - width / 2
            y1 = cy - height / 2
            x2 = cx + width / 2
            y2 = cy + height / 2
            
            # Clip to image boundaries
            x1 = np.clip(x1, 0, image_w - 1)
            y1 = np.clip(y1, 0, image_h - 1)
            x2 = np.clip(x2, 1, image_w)
            y2 = np.clip(y2, 1, image_h)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            boxes.append([x1, y1, x2, y2])
            labels.append(0)  # Single class
        
        return boxes, labels
    
    def get_image_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for specific image"""
        img_name = self.image_ids[idx]
        
        if img_name in self.metadata_cache:
            return self.metadata_cache[img_name]
        
        image = self._load_image(img_name)
        annotations = self.image_groups[img_name]
        
        metadata = {
            'image_name': img_name,
            'original_size': image.shape[:2],
            'num_cells': len(annotations),
            'cell_sizes': annotations[['width', 'height']].values.tolist(),
            'density': len(annotations) / (image.shape[0] * image.shape[1]) * 1e6
        }
        
        self.metadata_cache[img_name] = metadata
        return metadata


class RIVACollateFn:
    """Custom collate function for RIVA dataset"""
    
    def __call__(self, batch: List[Tuple]) -> Tuple[torch.Tensor, List[Dict]]:
        return collate_fn(batch)


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Collate function to handle variable size inputs/targets
    
    Args:
        batch: List of (image, target) tuples
    
    Returns:
        images: Stacked images [B, C, H, W]
        targets: List of target dicts
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
        
    images = torch.stack(images, dim=0)
    
    return images, targets
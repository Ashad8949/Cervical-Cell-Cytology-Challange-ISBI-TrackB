import albumentations as A
import cv2
import numpy as np
from typing import List, Tuple, Optional
from albumentations.pytorch import ToTensorV2

class MedicalAugmentations:
    """Medical imaging specific augmentations"""
    
    @staticmethod
    def stain_augmentation() -> A.Compose:
        """Augmentations for stain variations"""
        return A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.7
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.ChannelShuffle(p=0.1),
            A.ToGray(p=0.1),
            A.ChannelDropout(channel_drop_range=(1, 1), p=0.1),
        ])
    
    @staticmethod
    def focus_augmentation() -> A.Compose:
        """Augmentations for focus variations"""
        return A.Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            ], p=0.4),
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=0.2
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                brightness_by_max=True,
                p=0.5
            ),
        ])
    
    @staticmethod
    def morphological_augmentation() -> A.Compose:
        """Morphological augmentations"""
        return A.Compose([
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.2
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=0.2
            ),
            A.OpticalDistortion(
                distort_limit=0.05,
                shift_limit=0.05,
                p=0.2
            ),
        ])
    
    @staticmethod
    def copy_paste_augmentation(max_cells: int = 10) -> A.Compose:
        """Copy-paste augmentation for cells"""
        # This would require custom implementation
        # For now, use CoarseDropout to simulate missing cells
        return A.Compose([
            A.CoarseDropout(
                max_holes=10,
                max_height=50,
                max_width=50,
                min_holes=1,
                min_height=20,
                min_width=20,
                fill_value=0,
                p=0.3
            ),
        ])


class TrainAugmentations:
    """Training augmentations for RIVA dataset"""
    
    def __init__(
        self,
        image_size: int = 1024,
        strong_aug: bool = True,
        use_medical_aug: bool = True
    ):
        self.image_size = image_size
        self.strong_aug = strong_aug
        self.use_medical_aug = use_medical_aug
        
        # Base transforms (always applied)
        base_transforms = [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0
            ),
        ]
        
        # Geometric augmentations
        geometric_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-5, 5),
                shear=(-5, 5),
                p=0.5
            ),
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),
                p=0.3
            ),
        ]
        
        # Color augmentations
        color_transforms = [
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        ]
        
        # Medical-specific augmentations (flattened — nested Compose objects
        # lack bbox_params and break geometric bbox tracking)
        medical_transforms = []
        if use_medical_aug:
            # Stain / colour augmentations (pixel-only, bbox-safe)
            medical_transforms += [
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=30,
                    val_shift_limit=20, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.15, contrast=0.15,
                    saturation=0.15, hue=0.05, p=0.3
                ),
                # Focus augmentations
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.3),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                ], p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.3
                ),
            ]
        
        # Strong augmentations (bbox-safe only)
        strong_transforms = []
        if strong_aug:
            strong_transforms = [
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=4,
                    min_width=4,
                    fill_value=0,
                    p=0.2
                ),
                A.PixelDropout(dropout_prob=0.01, p=0.1),
            ]
        
        # Normalization
        normalize = [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]
        
        # Combine all transforms
        all_transforms = (
            base_transforms +
            geometric_transforms +
            color_transforms +
            medical_transforms +
            strong_transforms +
            normalize
        )
        
        self.transform = A.Compose(
            all_transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_visibility=0.3,
                clip=True
            )
        )
    
    def __call__(self, image, bboxes, labels):
        return self.transform(image=image, bboxes=bboxes, labels=labels)


class ValAugmentations:
    """Validation augmentations (minimal)"""
    
    def __init__(self, image_size: int = 1024):
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            clip=True
        ))
    
    def __call__(self, image, bboxes, labels):
        return self.transform(image=image, bboxes=bboxes, labels=labels)


class TestAugmentations:
    """Test augmentations (no bounding boxes)"""
    
    def __init__(self, image_size: int = 1024):
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def __call__(self, image):
        return self.transform(image=image)
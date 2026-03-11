import torch
import torch.nn as nn
import timm
from typing import List, Tuple, Optional, Dict, Any
import torch.nn.functional as F

class PathDINOBackbone(nn.Module):
    """
    PathDINO backbone pre-trained on pathology images
    """
    
    # Map config shorthand names to actual timm model identifiers
    MODEL_NAME_MAP = {
        "path_dino": "vit_base_patch14_dinov2.lvd142m",
        "pathdino": "vit_base_patch14_dinov2.lvd142m",
    }

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        img_size: int = 1024,
        feature_dim: int = 768,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        
        # Resolve config shorthand to actual timm model name
        resolved_name = self.MODEL_NAME_MAP.get(model_name, model_name)
        
        # Load PathDINO model
        self.backbone = timm.create_model(
            resolved_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,  # No classification head
            drop_path_rate=drop_path_rate
        )
        
        self.feature_dim = feature_dim
        self.num_features = self.backbone.num_features
        
        # Feature pyramid levels
        self.num_levels = 4
        
        # Register hooks for multi-scale features
        self.features = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks to extract intermediate features"""
        
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Register hooks at different transformer blocks
        for i, block in enumerate(self.backbone.blocks):
            if i in [2, 5, 8, 11]:  # Extract features from these blocks
                block.register_forward_hook(get_hook(f'block_{i}'))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning multi-scale features
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            Dictionary of features at different scales
        """
        # Clear previous features
        self.features.clear()
        
        # Forward through backbone
        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        
        # Forward through transformer blocks
        for block in self.backbone.blocks:
            x = block(x)
        
        # Global feature (CLS token)
        global_feat = x[:, 0]
        
        # Extract multi-scale features
        features = {
            'global': global_feat,
            'multi_scale': list(self.features.values()),
            'patch_tokens': x[:, 1:]  # Remove CLS token
        }
        
        return features
    
    def get_feature_channels(self) -> List[int]:
        """Get number of channels at each feature level"""
        return [self.feature_dim] * self.num_levels


class SwinTransformerBackbone(nn.Module):
    """
    Swin Transformer backbone with hierarchical features
    """
    
    def __init__(
        self,
        model_name: str = "swin_large_patch4_window12_384",
        pretrained: bool = True,
        img_size: int = 1024,
        drop_path_rate: float = 0.2
    ):
        super().__init__()
        
        # Load Swin Transformer
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,
            drop_path_rate=drop_path_rate
        )
        
        # Feature dimensions at different stages
        self.feature_dims = self.backbone.num_features
        self.num_levels = 4
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning hierarchical features
        """
        features = []
        
        # Patch embedding
        x = self.backbone.patch_embed(x)
        
        # Absolute position embedding
        if self.backbone.absolute_pos_embed is not None:
            x = x + self.backbone.absolute_pos_embed
        
        x = self.backbone.pos_drop(x)
        
        # Forward through stages
        for stage in self.backbone.layers:
            x = stage(x)
            features.append(x)
        
        # Global average pooling
        global_feat = x.mean(dim=1)
        
        return {
            'global': global_feat,
            'multi_scale': features,
            'patch_tokens': x
        }
    
    def get_feature_channels(self) -> List[int]:
        """Get feature channels at each level"""
        return self.feature_dims


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt V2 backbone for local feature extraction
    """
    
    def __init__(
        self,
        model_name: str = "convnextv2_base",
        pretrained: bool = True,
        img_size: int = 1024,
        drop_path_rate: float = 0.2
    ):
        super().__init__()
        
        # Load ConvNeXt V2
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate
        )
        
        # Extract feature dimensions
        self.num_levels = 4
        if hasattr(self.backbone, 'feature_info'):
            if isinstance(self.backbone.feature_info, list):
                self.feature_dims = [x['num_chs'] for x in self.backbone.feature_info]
            else:
                self.feature_dims = self.backbone.feature_info.channels()
        else:
            self.feature_dims = [128, 256, 512, 1024]  # Fallback for base
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning CNN features
        """
        features = []
        
        # Stem
        x = self.backbone.stem(x)
        
        # Forward through stages
        for stage in self.backbone.stages:
            x = stage(x)
            features.append(x)
        
        # Global average pooling
        global_feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
        
        return {
            'global': global_feat,
            'multi_scale': features,
            'final': x
        }
    
    def get_feature_channels(self) -> List[int]:
        return self.feature_dims


class MaxViTBackbone(nn.Module):
    """
    MaxViT backbone combining CNNs and Transformers
    """
    
    def __init__(
        self,
        model_name: str = "maxvit_base_tf_512.in1k",
        pretrained: bool = True,
        img_size: int = 1024
    ):
        super().__init__()
        
        # Load MaxViT
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0
        )
        
        self.num_levels = 4
        self.feature_dims = [64, 128, 256, 512]
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = []
        
        # Stem
        x = self.backbone.stem(x)
        
        # Forward through blocks
        for block in self.backbone.blocks:
            x = block(x)
            features.append(x)
        
        # Global pooling
        global_feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
        
        return {
            'global': global_feat,
            'multi_scale': features,
            'final': x
        }
    
    def get_feature_channels(self) -> List[int]:
        return self.feature_dims


class CaiTBackbone(nn.Module):
    """
    CaiT backbone with class attention for small object detection
    """
    
    def __init__(
        self,
        model_name: str = "cait_xxs24_224",
        pretrained: bool = True,
        img_size: int = 1024
    ):
        super().__init__()
        
        # Load CaiT
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0
        )
        
        self.feature_dim = self.backbone.num_features
        self.num_levels = 1  # CaiT doesn't have hierarchical features
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Forward through backbone
        x = self.backbone(x)
        
        return {
            'global': x,
            'multi_scale': [x],
            'patch_tokens': x
        }
    
    def get_feature_channels(self) -> List[int]:
        return [self.feature_dim]
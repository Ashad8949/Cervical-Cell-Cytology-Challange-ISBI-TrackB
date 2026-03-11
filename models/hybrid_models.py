import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from .backbones import *
from .detection_heads import *

class HybridCellDetector(nn.Module):
    """
    Hybrid model combining CNN and Transformer backbones
    """
    
    def __init__(
        self,
        cnn_backbone_name: str = "convnextv2_base",
        transformer_backbone_name: str = "vit_base_patch14_dinov2.lvd142m",
        detection_head_name: str = "dino_detr",
        num_classes: int = 1,
        num_queries: int = 500,
        img_size: int = 1024,
        pretrained: bool = True,
        fusion_method: str = "attention",
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        
        # CNN backbone (local features)
        self.cnn_backbone = ConvNeXtBackbone(
            model_name=cnn_backbone_name,
            pretrained=pretrained,
            img_size=img_size,
            drop_path_rate=drop_path_rate
        )
        
        # Transformer backbone (global context)
        self.transformer_backbone = PathDINOBackbone(
            model_name=transformer_backbone_name,
            pretrained=pretrained,
            img_size=img_size,
            drop_path_rate=drop_path_rate
        )
        
        # Feature fusion
        self.fusion_method = fusion_method
        if fusion_method == "attention":
            self.fusion = CrossAttentionFusion(
                cnn_dim=self.cnn_backbone.feature_dims[-1],
                trans_dim=self.transformer_backbone.feature_dim,
                hidden_dim=256
            )
        elif fusion_method == "concat":
            self.fusion = ConcatFusion(
                cnn_dim=self.cnn_backbone.feature_dims[-1],
                trans_dim=self.transformer_backbone.feature_dim,
                hidden_dim=256
            )
        elif fusion_method == "gating":
            self.fusion = GatingFusion(
                cnn_dim=self.cnn_backbone.feature_dims[-1],
                trans_dim=self.transformer_backbone.feature_dim,
                hidden_dim=256
            )
        
        # Detection head
        if detection_head_name == "dino_detr":
            self.detection_head = DINO_DETRHead(
                in_channels=256,
                hidden_dim=256,
                num_classes=num_classes,
                num_queries=num_queries,
                num_decoder_layers=6,
                num_encoder_layers=6,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation="relu",
                two_stage=True,
                with_box_refine=True,
                num_patterns=0,
                num_feature_levels=5,
            )
        elif detection_head_name == "deformable":
            self.detection_head = DeformableDETRHead(
                in_channels=256,
                hidden_dim=256,
                num_classes=num_classes,
                num_queries=num_queries,
                num_decoder_layers=6,
                num_feature_levels=4,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation="relu",
                with_box_refine=True,
                as_two_stage=False,
            )
        
        # Feature pyramid network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.cnn_backbone.feature_dims,
            out_channels=256
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for non-pretrained modules only.
        
        Backbone weights are loaded from pretrained checkpoints and
        must NOT be reinitialised here.
        """
        # Collect parameter ids from pretrained backbones so we skip them
        pretrained_params = set()
        for backbone in [self.cnn_backbone, self.transformer_backbone]:
            pretrained_params.update(id(p) for p in backbone.parameters())
        
        for m in self.modules():
            # Skip backbone sub-modules entirely
            if any(m is sub for sub in self.cnn_backbone.modules()) or \
               any(m is sub for sub in self.transformer_backbone.modules()):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            Dictionary with predictions
        """
        # Extract features from both backbones
        cnn_features = self.cnn_backbone(x)
        trans_features = self.transformer_backbone(x)
        
        # Get multi-scale CNN features
        cnn_multi_scale = cnn_features['multi_scale']
        
        # Fuse features
        if self.fusion_method == "attention":
            # Reshape transformer features
            bs, hw, c = trans_features['patch_tokens'].shape
            h = w = int(math.sqrt(hw))
            trans_patches = trans_features['patch_tokens'].transpose(1, 2).reshape(bs, c, h, w)
            
            # Fuse with last CNN feature
            fused = self.fusion(cnn_multi_scale[-1], trans_patches)
        else:
            # Other fusion methods
            fused = self.fusion(cnn_features['final'], trans_features['global'])
        
        # Build multi-scale features for detection head
        fpn_features = self.fpn(cnn_multi_scale)
        fpn_features.append(fused)  # Add fused features as highest level
        
        # Pass through detection head
        predictions = self.detection_head(fpn_features)
        
        return predictions


class MultiScaleDetector(nn.Module):
    """
    Multi-scale detector with progressive refinement
    """
    
    def __init__(
        self,
        backbone_name: str = "path_dino",
        num_scales: int = 3,
        num_classes: int = 1,
        img_size: int = 1024,
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.num_scales = num_scales
        self.img_size = img_size
        
        # Backbone
        if backbone_name == "path_dino":
            self.backbone = PathDINOBackbone(
                model_name="vit_base_patch14_dinov2.lvd142m",
                pretrained=pretrained,
                img_size=img_size,
                drop_path_rate=0.1
            )
        elif backbone_name == "swin":
            self.backbone = SwinTransformerBackbone(
                model_name="swin_large_patch4_window12_384",
                pretrained=pretrained,
                img_size=img_size,
                drop_path_rate=0.2
            )
        
        # Multi-scale heads
        self.detection_heads = nn.ModuleList([
            DINO_DETRHead(
                in_channels=256,
                hidden_dim=256,
                num_classes=num_classes,
                num_queries=200,
                num_decoder_layers=4,
                num_encoder_layers=4,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation="relu",
                two_stage=True,
                with_box_refine=True,
                num_patterns=0,
                num_feature_levels=4,
            )
            for _ in range(num_scales)
        ])
        
        # Scale fusion
        self.scale_fusion = ScaleFusionModule(
            num_scales=num_scales,
            hidden_dim=256
        )
        
        # Progressive refinement
        self.refinement = ProgressiveRefinement(
            num_stages=3,
            hidden_dim=256
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.backbone(x)
        
        # Multi-scale processing
        all_predictions = []
        for i, head in enumerate(self.detection_heads):
            # Process at different scales
            scale_factor = 1.0 / (2 ** i)
            scaled_features = self._scale_features(features, scale_factor)
            predictions = head(scaled_features)
            all_predictions.append(predictions)
        
        # Fuse predictions from different scales
        fused = self.scale_fusion(all_predictions)
        
        # Refine predictions
        refined = self.refinement(fused)
        
        return refined
    
    def _scale_features(self, features: Dict[str, torch.Tensor], scale: float):
        """Scale features by given factor"""
        # Implementation depends on feature format
        pass


class GraphEnhancedDetector(nn.Module):
    """
    Detector enhanced with graph neural networks
    """
    
    def __init__(
        self,
        backbone_name: str = "path_dino",
        num_classes: int = 1,
        img_size: int = 1024,
        pretrained: bool = True,
        graph_layers: int = 2,
        graph_heads: int = 4,
    ):
        super().__init__()
        
        # Backbone
        self.backbone = PathDINOBackbone(
            model_name="vit_base_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            img_size=img_size,
            drop_path_rate=0.1
        )
        
        # Detection head
        self.detection_head = DINO_DETRHead(
            in_channels=256,
            hidden_dim=256,
            num_classes=num_classes,
            num_queries=300,
            num_decoder_layers=6,
            num_encoder_layers=6,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            two_stage=True,
            with_box_refine=True,
            num_patterns=0,
            num_feature_levels=4,
        )
        
        # Graph refinement
        self.graph_refinement = GraphRefinementModule(
            node_dim=256,
            edge_dim=64,
            hidden_dim=256,
            num_layers=graph_layers,
            num_heads=graph_heads
        )
        
        # Message passing
        self.message_passing = MessagePassingLayer(
            node_dim=256,
            edge_dim=64,
            hidden_dim=256
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.backbone(x)
        
        # Initial detection
        predictions = self.detection_head([features['final']])
        
        # Build graph from predictions
        boxes = predictions['pred_boxes']
        scores = predictions['pred_logits'].softmax(dim=-1)[..., 0]
        
        # Apply graph refinement
        refined_boxes = self.graph_refinement(boxes, scores, features['global'])
        
        # Update predictions
        predictions['pred_boxes'] = refined_boxes
        
        return predictions


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module"""
    
    def __init__(self, cnn_dim: int, trans_dim: int, hidden_dim: int):
        super().__init__()
        
        self.cnn_proj = nn.Conv2d(cnn_dim, hidden_dim, 1)
        self.trans_proj = nn.Conv2d(trans_dim, hidden_dim, 1)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.output_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
    
    def forward(self, cnn_feat: torch.Tensor, trans_feat: torch.Tensor):
        # Project to same dimension
        cnn_proj = self.cnn_proj(cnn_feat)
        trans_proj = self.trans_proj(trans_feat)
        
        # Reshape for attention
        bs, c, h, w = cnn_proj.shape
        cnn_flat = cnn_proj.flatten(2).transpose(1, 2)  # [bs, h*w, c]
        trans_flat = trans_proj.flatten(2).transpose(1, 2)  # [bs, h*w, c]
        
        # Cross-attention
        attended, _ = self.attention(
            query=cnn_flat,
            key=trans_flat,
            value=trans_flat
        )
        
        # Reshape back
        attended = attended.transpose(1, 2).reshape(bs, c, h, w)
        
        # Output projection
        output = self.output_proj(attended)
        
        return output


class ConcatFusion(nn.Module):
    """Concatenation fusion module"""
    
    def __init__(self, cnn_dim: int, trans_dim: int, hidden_dim: int):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Conv2d(cnn_dim + trans_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
    
    def forward(self, cnn_feat: torch.Tensor, trans_feat: torch.Tensor):
        # Ensure same spatial size
        if cnn_feat.shape[-2:] != trans_feat.shape[-2:]:
            trans_feat = F.interpolate(
                trans_feat, 
                size=cnn_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Concatenate
        fused = torch.cat([cnn_feat, trans_feat], dim=1)
        
        # Fusion
        output = self.fusion(fused)
        
        return output


class GatingFusion(nn.Module):
    """Gated fusion module"""
    
    def __init__(self, cnn_dim: int, trans_dim: int, hidden_dim: int):
        super().__init__()
        
        self.cnn_gate = nn.Sequential(
            nn.Conv2d(cnn_dim, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.trans_gate = nn.Sequential(
            nn.Conv2d(trans_dim, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Conv2d(cnn_dim + trans_dim, hidden_dim, 1)
    
    def forward(self, cnn_feat: torch.Tensor, trans_feat: torch.Tensor):
        # Ensure same spatial size
        if cnn_feat.shape[-2:] != trans_feat.shape[-2:]:
            trans_feat = F.interpolate(
                trans_feat,
                size=cnn_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Compute gates
        cnn_gate = self.cnn_gate(cnn_feat)
        trans_gate = self.trans_gate(trans_feat)
        
        # Apply gates
        cnn_weighted = cnn_feat * cnn_gate
        trans_weighted = trans_feat * trans_gate
        
        # Concatenate and fuse
        fused = torch.cat([cnn_weighted, trans_weighted], dim=1)
        output = self.fusion(fused)
        
        return output


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale features"""
    
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
            output_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Build pyramid features
        
        Args:
            features: List of features from different levels
        
        Returns:
            List of pyramid features
        """
        # Lateral connections
        laterals = []
        for i, (feat, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            laterals.append(lateral_conv(feat))
        
        # Build pyramid from top to bottom
        pyramid_features = []
        for i in range(len(laterals) - 1, -1, -1):
            if i == len(laterals) - 1:
                # Top level
                pyramid_feat = laterals[i]
            else:
                # Add upsampled feature from higher level
                pyramid_feat = laterals[i] + F.interpolate(
                    pyramid_features[-1],
                    size=laterals[i].shape[-2:],
                    mode='nearest'
                )
            
            # Output convolution
            pyramid_feat = self.output_convs[i](pyramid_feat)
            pyramid_features.append(pyramid_feat)
        
        # Reverse to get from low to high resolution
        pyramid_features = pyramid_features[::-1]
        
        return pyramid_features


class ScaleFusionModule(nn.Module):
    """Fuse predictions from multiple scales"""
    
    def __init__(self, num_scales: int, hidden_dim: int):
        super().__init__()
        
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, predictions: List[Dict[str, torch.Tensor]]):
        # Weighted fusion of predictions
        fused_logits = 0
        fused_boxes = 0
        
        total_weight = self.scale_weights.sum()
        
        for i, pred in enumerate(predictions):
            weight = self.scale_weights[i] / total_weight
            fused_logits += weight * pred['pred_logits']
            fused_boxes += weight * pred['pred_boxes']
        
        return {
            'pred_logits': fused_logits,
            'pred_boxes': fused_boxes
        }


class ProgressiveRefinement(nn.Module):
    """Progressive refinement of predictions"""
    
    def __init__(self, num_stages: int, hidden_dim: int):
        super().__init__()
        
        self.num_stages = num_stages
        
        self.refinement_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4)  # x, y, w, h
            )
            for _ in range(num_stages)
        ])
    
    def forward(self, predictions: Dict[str, torch.Tensor]):
        boxes = predictions['pred_boxes']
        
        # Progressive refinement
        for layer in self.refinement_layers:
            # Compute refinement delta
            delta = layer(predictions['pred_logits'])
            
            # Apply refinement
            boxes = boxes + delta
        
        predictions['pred_boxes'] = boxes
        return predictions


class GraphRefinementModule(nn.Module):
    """Graph-based refinement of detections"""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4
    ):
        super().__init__()
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Node decoder
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # Refinement delta
        )
    
    def forward(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        context: torch.Tensor
    ):
        bs, num_nodes, _ = boxes.shape
        
        # Encode nodes
        node_features = self.node_encoder(
            torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)
        )
        
        # Build adjacency matrix
        adj_matrix = self._build_adjacency(boxes)
        
        # Encode edges
        edge_features = self.edge_encoder(adj_matrix)
        
        # Graph attention
        for gat_layer in self.gat_layers:
            node_features = gat_layer(node_features, edge_features)
        
        # Decode refinement
        refinement = self.node_decoder(node_features)
        
        # Apply refinement
        refined_boxes = boxes + refinement
        
        return refined_boxes
    
    def _build_adjacency(self, boxes: torch.Tensor):
        """Build adjacency matrix based on box IoU"""
        # Compute IoU between all boxes
        # Implementation omitted for brevity
        pass


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer"""
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        
        self.output = nn.Linear(out_dim, out_dim)
        
    def forward(self, nodes: torch.Tensor, edges: torch.Tensor):
        bs, num_nodes, _ = nodes.shape
        
        # Linear projections
        Q = self.query(nodes).reshape(bs, num_nodes, self.num_heads, self.head_dim)
        K = self.key(nodes).reshape(bs, num_nodes, self.num_heads, self.head_dim)
        V = self.value(nodes).reshape(bs, num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) / math.sqrt(self.head_dim)
        
        # Add edge information
        scores = scores + edges.unsqueeze(1)
        
        # Softmax
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.einsum('bhqk,bkhd->bqhd', attention, V)
        out = out.reshape(bs, num_nodes, -1)
        
        # Output projection
        out = self.output(out)
        
        return out


class MessagePassingLayer(nn.Module):
    """Message Passing Layer for graph refinement"""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        
        self.message_fn = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_fn = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, nodes: torch.Tensor, edges: torch.Tensor):
        bs, num_nodes, _ = nodes.shape
        
        # Compute messages
        messages = torch.zeros_like(nodes)
        
        for i in range(num_nodes):
            # Gather neighbor information
            neighbor_messages = []
            
            for j in range(num_nodes):
                if i != j:
                    # Concatenate node features and edge features
                    message_input = torch.cat([
                        nodes[:, i, :],
                        nodes[:, j, :],
                        edges[:, i, j, :]
                    ], dim=-1)
                    
                    # Compute message
                    message = self.message_fn(message_input)
                    neighbor_messages.append(message)
            
            if neighbor_messages:
                # Aggregate messages
                aggregated = torch.stack(neighbor_messages, dim=1).mean(dim=1)
                
                # Update node
                update_input = torch.cat([nodes[:, i, :], aggregated], dim=-1)
                messages[:, i, :] = self.update_fn(update_input)
        
        return nodes + messages
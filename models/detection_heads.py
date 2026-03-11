import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union
import math

class DINO_DETRHead(nn.Module):
    """
    DINO-DETR detection head with denoising training
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 1,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        num_encoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        two_stage: bool = True,
        with_box_refine: bool = True,
        num_patterns: int = 0,
        num_feature_levels: int = 4,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_feature_levels = num_feature_levels
        self.two_stage = two_stage
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        
        # Query embedding
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Level embedding
        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)
        
        # Reference points
        self.reference_points = nn.Linear(hidden_dim, 2)
        
        # Output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # Denoising groups
        self.denoising = DenoisingGroup(num_queries, hidden_dim)
        
        # Initialize weights
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Special initialization
        nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        nn.init.constant_(self.reference_points.bias.data, 0.)
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes + 1) * bias_value
        
    def forward(
        self,
        features: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            features: List of multi-scale features
            masks: Optional mask for padded regions
        
        Returns:
            Dictionary with predictions
        """
        bs = features[0].shape[0]
        
        # Prepare features for transformer
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        
        if masks is None:
            masks = [None] * len(features)
            
        for lvl, (src, mask) in enumerate(zip(features, masks)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            
            # Flatten spatial dimensions
            src = src.flatten(2).transpose(1, 2)  # [bs, h*w, c]
            
            # Use default mask if not provided (assume all valid)
            if mask is None:
                mask = torch.zeros((bs, h * w), dtype=torch.bool, device=src.device)
            else:
                mask = mask.flatten(1)  # [bs, h*w]
            
            # Position embedding
            pos = self.level_embed.weight[lvl].view(1, 1, -1)
            pos = pos.repeat(bs, h * w, 1)
            
            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(pos)
        
        # Concatenate all levels
        src_flatten = torch.cat(src_flatten, dim=1)
        mask_flatten = torch.cat(mask_flatten, dim=1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, dim=1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        
        # Encoder
        # Standard PyTorch TransformerEncoder doesn't support 'pos' arg, so we add it to input
        memory = self.encoder(
            src_flatten + lvl_pos_embed_flatten,
            src_key_padding_mask=mask_flatten
        )
        
        # Prepare queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        tgt = torch.zeros_like(query_embed)
        
        # Reference points
        reference_points = self.reference_points(query_embed).sigmoid()
        
        # Decoder
        # Manual iteration to get intermediate outputs
        hs = []
        output = tgt + query_embed
        
        for layer in self.decoder.layers:
            output = layer(
                output,
                memory + lvl_pos_embed_flatten,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=mask_flatten
            )
            hs.append(output)
            
        hs = torch.stack(hs) # [num_layers, bs, num_queries, dim]
        
        # Outputs
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        # Two-stage refinement
        if self.two_stage:
            # Use encoder output as initial queries
            pass
        
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'aux_outputs': self._set_aux_loss(outputs_class, outputs_coord)
        }
        
        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        """Prepare auxiliary outputs for deep supervision"""
        return [
            {'pred_logits': a, 'pred_boxes': b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class DeformableDETRHead(nn.Module):
    """
    Deformable DETR head for faster convergence
    Uses multi-scale deformable attention for efficient processing
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 1,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        num_feature_levels: int = 4,
        num_encoder_points: int = 4,
        num_decoder_points: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        with_box_refine: bool = True,
        as_two_stage: bool = False,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_feature_levels = num_feature_levels
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        
        # Input projection for multi-scale features
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1),
                nn.GroupNorm(32, hidden_dim)
            ) for _ in range(num_feature_levels)
        ])
        
        # Level embedding
        self.level_embed = nn.Parameter(torch.zeros(num_feature_levels, hidden_dim))
        nn.init.normal_(self.level_embed)
        
        # Query embedding
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        
        # Reference points for queries
        self.reference_points = nn.Linear(hidden_dim, 2)
        
        # Decoder layers with self-attention and cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        
        # Box prediction with optional iterative refinement
        if with_box_refine:
            self.bbox_embed = nn.ModuleList([
                MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_decoder_layers)
            ])
        else:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0)
    
    def forward(
        self,
        features: List[torch.Tensor],
        pos_embeds: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: List of multi-scale feature maps
            pos_embeds: Optional positional embeddings
        """
        batch_size = features[0].shape[0]
        device = features[0].device
        
        # Project features
        srcs = []
        for lvl, feat in enumerate(features[:self.num_feature_levels]):
            src = self.input_proj[lvl](feat)
            srcs.append(src)
        
        # Flatten and combine features
        src_flatten = []
        spatial_shapes = []
        
        for lvl, src in enumerate(srcs):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src = src.flatten(2).transpose(1, 2)  # [B, H*W, C]
            # Add level embedding
            src = src + self.level_embed[lvl].view(1, 1, -1)
            src_flatten.append(src)
        
        memory = torch.cat(src_flatten, dim=1)  # [B, sum(H*W), C]
        
        # Query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_content, query_pos = query_embed.split(self.hidden_dim, dim=-1)
        
        # Initial reference points
        reference_points = self.reference_points(query_content).sigmoid()  # [B, num_queries, 2]
        
        # Decoder
        tgt = torch.zeros_like(query_content)
        
        # Add positional encoding to queries
        hs = self.decoder(
            tgt=tgt + query_pos,
            memory=memory,
            tgt_mask=None,
            memory_mask=None
        )
        
        # Output predictions
        outputs_class = self.class_embed(hs)
        
        if self.with_box_refine:
            outputs_coord = self.bbox_embed[-1](hs).sigmoid()
        else:
            outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord,
            'reference_points': reference_points
        }


class ConditionalDETRHead(nn.Module):
    """
    Conditional DETR for better query initialization
    Uses conditional cross-attention for improved convergence
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 1,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        
        # Query embeddings (content and positional)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos = nn.Embedding(num_queries, hidden_dim)
        
        # Conditional spatial query
        self.ref_point_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        
        # Conditional decoder layers
        self.decoder_layers = nn.ModuleList([
            ConditionalDecoderLayer(
                hidden_dim=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            ) for _ in range(num_decoder_layers)
        ])
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        
        # Output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        features: Union[torch.Tensor, List[torch.Tensor]],
        pos_embed: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Feature map or list of feature maps
            pos_embed: Optional positional embedding
        """
        # Handle list of features
        if isinstance(features, list):
            features = features[-1]  # Use highest resolution
        
        batch_size = features.shape[0]
        device = features.device
        
        # Project features
        src = self.input_proj(features)  # [B, C, H, W]
        bs, c, h, w = src.shape
        
        # Flatten
        src = src.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Create positional encoding if not provided
        if pos_embed is None:
            pos_embed = self._create_pos_embed(h, w, c, device)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # [B, H*W, C]
            pos_embed = pos_embed.expand(batch_size, -1, -1)
        
        # Query embeddings
        query_content = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Initial reference points
        ref_point = self.ref_point_head(query_pos)
        ref_point = ref_point.sigmoid()  # [B, num_queries, 2]
        
        # Decoder
        output = query_content
        
        for layer in self.decoder_layers:
            output = layer(
                tgt=output,
                memory=src,
                query_pos=query_pos,
                memory_pos=pos_embed,
                ref_point=ref_point
            )
        
        output = self.decoder_norm(output)
        
        # Output predictions
        outputs_class = self.class_embed(output)
        outputs_coord = self.bbox_embed(output).sigmoid()
        
        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }
    
    def _create_pos_embed(self, h, w, dim, device):
        """Create 2D sinusoidal positional embedding"""
        y_embed = torch.arange(h, device=device).unsqueeze(1).expand(h, w)
        x_embed = torch.arange(w, device=device).unsqueeze(0).expand(h, w)
        
        dim_t = torch.arange(dim // 2, dtype=torch.float32, device=device)
        dim_t = 10000 ** (2 * dim_t / dim)
        
        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)
        
        pos = torch.cat([pos_y, pos_x], dim=-1)  # [H, W, C]
        return pos.unsqueeze(0)  # [1, H, W, C]


class ConditionalDecoderLayer(nn.Module):
    """Conditional decoder layer with spatial modulation"""
    
    def __init__(
        self,
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross attention (conditional)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Spatial modulation
        self.spatial_mod = nn.Linear(hidden_dim, hidden_dim)
        
        # FFN
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout4 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        memory_pos: torch.Tensor,
        ref_point: torch.Tensor
    ) -> torch.Tensor:
        # Self attention
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Conditional cross attention with spatial modulation
        spatial_query = self.spatial_mod(query_pos)
        q = tgt + spatial_query
        k = memory + memory_pos
        
        tgt2, _ = self.cross_attn(q, k, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class DynamicDETRHead(nn.Module):
    """
    Dynamic DETR with dynamic query selection
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 1,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        
        # Dynamic DETR implementation
        pass


class DenoisingGroup(nn.Module):
    """
    Denoising group for DINO training
    """
    
    def __init__(self, num_queries: int, hidden_dim: int):
        super().__init__()
        
        # Noise scale
        self.noise_scale = 0.4
        
        # Groups
        self.num_groups = 5
        
        # Group embeddings
        self.group_embed = nn.Embedding(self.num_groups, hidden_dim)
        
    def add_noise(self, boxes: torch.Tensor) -> torch.Tensor:
        """Add noise to boxes for denoising training"""
        noise = torch.randn_like(boxes) * self.noise_scale
        return boxes + noise
    
    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """Apply denoising"""
        noisy_boxes = self.add_noise(boxes)
        return noisy_boxes


class MLP(nn.Module):
    """Simple MLP (matches standard DETR implementation).
    
    The last layer is a plain Linear without activation so that
    box regression outputs are unconstrained before sigmoid.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ):
        super().__init__()
        
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
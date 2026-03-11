"""
Swin Transformer Backbone for RT-DETR Detection.

Replaces RT-DETR's default backbone (HGNetV2/ResNet) with a pretrained
Swin Transformer (timm), keeping the RT-DETR hybrid encoder and
transformer decoder intact.

RT-DETR architecture:
    Backbone → Hybrid Encoder (AIFI + CCFM) → Transformer Decoder → Detection

This module replaces only the Backbone stage with Swin Transformer,
producing multi-scale features at strides 8, 16, 32 that feed into
RT-DETR's hybrid encoder.

Usage:
    from swin_backbone import SwinRTDETRTrainer

    SwinRTDETRTrainer.swin_variant = 'swin_base_patch4_window12_384_in22k'
    trainer = SwinRTDETRTrainer(overrides={'model': 'rtdetr-l.pt', ...})
    trainer.train()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("timm is required: pip install timm")

from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.models.rtdetr.train import RTDETRTrainer


class SwinRTDETRModel(RTDETRDetectionModel):
    """RTDETRDetectionModel with Swin Transformer backbone.

    Builds the standard RT-DETR architecture, then:
    - Adds a pretrained Swin backbone (timm) for feature extraction
    - Adds 1x1 conv channel adapters to match RT-DETR encoder expectations
    - Freezes unused CNN backbone layers
    - Overrides forward pass to route Swin features through RT-DETR encoder/decoder

    RT-DETR expects multi-scale features from backbone at strides 8, 16, 32.
    Swin stages 1, 2, 3 produce features at 1/8, 1/16, 1/32 resolution.
    These are adapted to match the channel dimensions expected by RT-DETR's
    hybrid encoder (AIFI + CCFM modules).

    The Swin model is created with img_size matching the training resolution
    so PatchEmbed and attention masks are correct.
    """

    def __init__(
        self,
        cfg="rtdetr-l.yaml",
        ch=3,
        nc=None,
        swin_variant="swin_base_patch4_window12_384_in22k",
        imgsz=1280,
        verbose=True,
    ):
        # Build standard RT-DETR architecture (backbone + encoder + decoder)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # Store expected image size for runtime adaptive resizing
        self.swin_imgsz = imgsz

        # --- Identify backbone output channel sizes from built RT-DETR ---
        # RT-DETR's hybrid encoder expects multi-scale feature maps.
        # We need to find what channels the encoder expects as input.
        backbone_out_channels = self._detect_backbone_channels()
        if verbose:
            print(f"  RT-DETR backbone channels: {backbone_out_channels}")

        # --- Swin Transformer backbone ---
        swin_full = timm.create_model(
            swin_variant,
            pretrained=True,
            img_size=imgsz,
        )

        # Keep only the pieces we need for feature extraction
        self.swin_patch_embed = swin_full.patch_embed
        # Allow variable input sizes (validation may use different sizes)
        if hasattr(self.swin_patch_embed, 'strict_img_size'):
            self.swin_patch_embed.strict_img_size = False
        else:
            self.swin_patch_embed.img_size = None
        self.swin_pos_drop = (
            swin_full.pos_drop if hasattr(swin_full, "pos_drop") else nn.Identity()
        )
        self.swin_layers = swin_full.layers  # nn.Sequential of 4 SwinTransformerStage
        self.swin_norms = nn.ModuleList()

        # Build per-stage norms
        embed_dim = swin_full.embed_dim
        for i in range(1, 4):  # stages 1, 2, 3
            dim = embed_dim * (2 ** i)
            self.swin_norms.append(nn.LayerNorm(dim))

        swin_ch = [embed_dim * (2 ** i) for i in range(1, 4)]  # e.g. [256, 512, 1024]

        del swin_full

        # --- Channel adapters: Swin channels → RT-DETR expected channels ---
        # RT-DETR expects 3 feature levels with specific channel counts
        if len(backbone_out_channels) >= 3:
            rtdetr_p3_ch = backbone_out_channels[0]
            rtdetr_p4_ch = backbone_out_channels[1]
            rtdetr_p5_ch = backbone_out_channels[2]
        else:
            # Fallback: use common RT-DETR-L channel sizes
            rtdetr_p3_ch = 256
            rtdetr_p4_ch = 256
            rtdetr_p5_ch = 256

        self.adapt_p3 = self._make_adapter(swin_ch[0], rtdetr_p3_ch)
        self.adapt_p4 = self._make_adapter(swin_ch[1], rtdetr_p4_ch)
        self.adapt_p5 = self._make_adapter(swin_ch[2], rtdetr_p5_ch)

        # --- ImageNet normalisation buffers (Swin expects normalised input) ---
        self.register_buffer(
            "swin_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "swin_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # --- Freeze original CNN backbone layers ---
        # Find and freeze the backbone portion of the model
        self._freeze_original_backbone()

        if verbose:
            n_swin = sum(
                p.numel()
                for mod in [self.swin_patch_embed, self.swin_layers, self.swin_norms]
                for p in mod.parameters()
            )
            n_adapt = sum(
                p.numel()
                for m in [self.adapt_p3, self.adapt_p4, self.adapt_p5]
                for p in m.parameters()
            )
            print(f"\n  Swin backbone   : {swin_variant}")
            print(f"  Swin img_size   : {imgsz}")
            print(f"  Swin channels   : {swin_ch}")
            print(f"  RT-DETR channels: P3={rtdetr_p3_ch}, P4={rtdetr_p4_ch}, P5={rtdetr_p5_ch}")
            print(f"  Swin params     : {n_swin:,}")
            print(f"  Adapter params  : {n_adapt:,}")
            print(f"  CNN backbone    : FROZEN (replaced by Swin)\n")

    # ------------------------------------------------------------------
    def _detect_backbone_channels(self):
        """Detect the channel dimensions the RT-DETR encoder expects.

        Walks the model layers to find the first encoder/neck module and
        inspects what input channels it expects for each scale level.
        """
        channels = []

        # Strategy 1: Look through model layers for the encoder input projections
        for m in self.model:
            module = m
            # RT-DETR hybrid encoder often has input_proj or similar
            if hasattr(module, 'input_proj'):
                for proj in module.input_proj:
                    # Each projection is typically a Conv2d or Sequential
                    if isinstance(proj, nn.Sequential):
                        for sub in proj:
                            if isinstance(sub, nn.Conv2d):
                                channels.append(sub.weight.shape[1])
                                break
                    elif isinstance(proj, nn.Conv2d):
                        channels.append(proj.weight.shape[1])
                if channels:
                    return channels

        # Strategy 2: Inspect backbone output layers directly
        for i, m in enumerate(self.model):
            if hasattr(m, 'cv2'):
                conv = m.cv2
                while hasattr(conv, 'conv'):
                    conv = conv.conv
                if hasattr(conv, 'weight'):
                    channels.append(conv.weight.shape[0])
            elif hasattr(m, 'conv') and hasattr(m.conv, 'weight'):
                channels.append(m.conv.weight.shape[0])

        # Return last 3 unique channel sizes (P3, P4, P5)
        if len(channels) >= 3:
            return channels[-3:]

        # Fallback: common RT-DETR-L dimensions
        return [256, 256, 256]

    # ------------------------------------------------------------------
    def _freeze_original_backbone(self):
        """Freeze the original CNN backbone layers.

        RT-DETR model structure typically has backbone layers in the
        first portion of self.model. We freeze everything before the
        encoder/decoder modules.
        """
        # Find where the backbone ends (look for encoder-like modules)
        backbone_end = 0
        for i, m in enumerate(self.model):
            # Once we hit a module with 'input_proj' or attention layers,
            # we've reached the encoder
            if hasattr(m, 'input_proj') or hasattr(m, 'encoder'):
                backbone_end = i
                break
            # Also check for transformer-related attributes
            if hasattr(m, 'self_attn') or hasattr(m, 'multihead_attn'):
                backbone_end = i
                break

        if backbone_end == 0:
            # Conservative: freeze first half of model layers
            backbone_end = len(self.model) // 3

        for i in range(backbone_end):
            for p in self.model[i].parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    @staticmethod
    def _make_adapter(c_in: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    # ------------------------------------------------------------------
    def _swin_features(self, x):
        """Extract multi-scale features from Swin stages 1, 2, 3.

        Handles input sizes that differ from self.swin_imgsz by resizing
        before PatchEmbed and scaling feature maps back afterwards.
        """
        H_in, W_in = x.shape[2], x.shape[3]
        need_resize = (H_in != self.swin_imgsz or W_in != self.swin_imgsz)
        if need_resize:
            x = F.interpolate(
                x,
                size=(self.swin_imgsz, self.swin_imgsz),
                mode="bilinear",
                align_corners=False,
            )

        x = self.swin_patch_embed(x)
        x = self.swin_pos_drop(x)

        feats = []
        for i, stage in enumerate(self.swin_layers):
            x = stage(x)
            if i >= 1:  # collect stages 1, 2, 3 (skip stage 0)
                out = self.swin_norms[i - 1](x)
                out = out.permute(0, 3, 1, 2).contiguous()
                if need_resize:
                    stride = 2 ** (i + 2)
                    target_h = H_in // stride
                    target_w = W_in // stride
                    out = F.interpolate(
                        out,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                feats.append(out)
        return feats  # [P3 (1/8), P4 (1/16), P5 (1/32)]

    # ------------------------------------------------------------------
    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Forward pass: Swin backbone → channel adapt → RT-DETR encoder/decoder.

        During super().__init__(), a dummy forward pass runs to compute strides
        before Swin is attached. Fall back to standard forward in that case.
        """
        if not hasattr(self, "swin_patch_embed"):
            return super()._predict_once(x, profile, visualize, embed)

        # 1. Normalise for Swin (preprocessing already scales to [0,1])
        x_norm = (x - self.swin_mean) / self.swin_std

        # 2. Extract Swin multi-scale features
        feats = self._swin_features(x_norm)
        p3 = self.adapt_p3(feats[0])   # 1/8
        p4 = self.adapt_p4(feats[1])   # 1/16
        p5 = self.adapt_p5(feats[2])   # 1/32

        # 3. Determine where backbone ends and encoder begins
        # Build feature cache as in the original model
        y = []
        head_start = None

        for i, m in enumerate(self.model):
            if hasattr(m, 'input_proj') or hasattr(m, 'encoder'):
                head_start = i
                break

        if head_start is None:
            head_start = len(self.model) // 3

        # Fill y cache for backbone layers (not actually used, just placeholders)
        for i in range(head_start):
            y.append(None)

        # 4. Now pass Swin features through encoder + decoder
        # The encoder module typically concatenates multi-scale features
        # We need to inject our features at the right point
        x_out = None
        for i, m in enumerate(self.model):
            if i < head_start:
                # Skip backbone layers
                continue

            # Handle the concat/encoder input
            if m.f != -1:
                if isinstance(m.f, int):
                    x_in = y[m.f] if m.f < len(y) and y[m.f] is not None else x_out
                else:
                    # Multi-input: replace backbone references with Swin features
                    x_in = []
                    for j in m.f:
                        if j == -1:
                            x_in.append(x_out)
                        elif j < len(y) and y[j] is not None:
                            x_in.append(y[j])
                        else:
                            x_in.append(x_out)
            else:
                x_in = x_out

            # For the first encoder layer, inject Swin features
            if i == head_start:
                # RT-DETR encoder expects a list of feature maps [P3, P4, P5]
                if hasattr(m, 'input_proj'):
                    x_out = m([p3, p4, p5])
                else:
                    x_out = m(x_in) if x_in is not None else m(p5)
            else:
                x_out = m(x_in)

            # Extend y cache
            while len(y) <= m.i:
                y.append(None)
            if m.i < len(y):
                y[m.i] = x_out if m.i in self.save else None
            else:
                y.append(x_out if m.i in self.save else None)

        return x_out

    # ------------------------------------------------------------------
    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """Predict override to handle both training and inference."""
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)


# ======================================================================
# Custom Trainer
# ======================================================================

class SwinRTDETRTrainer(RTDETRTrainer):
    """RTDETRTrainer that builds a SwinRTDETRModel."""

    swin_variant: str = "swin_base_patch4_window12_384_in22k"

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SwinRTDETRModel(
            cfg=cfg or self.args.model,
            nc=self.data["nc"],
            swin_variant=self.__class__.swin_variant,
            imgsz=self.args.imgsz,
            verbose=verbose,
        )
        if weights:
            model.load(weights)
        return model
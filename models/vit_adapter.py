"""
Module A (Frozen Backbone) + Module C (Parameter-Efficient Adapter / MoE).

Architecture:
    ViT-B/16 (frozen) → Adapter/MoE layers (trainable) → Incremental classifier head (trainable)

Supports two modes via `use_moe` flag:
    - Single adapter (baseline, default)
    - MoE adapters (multiple experts with routing gate)
"""

import copy
from typing import Optional

import timm
import torch
import torch.nn as nn

from models.moe_adapter import MoEAdaptedBlock


# ── Adapter Layer ──────────────────────────────────────────────────────────

class Adapter(nn.Module):
    """Lightweight bottleneck adapter: down-proj → GELU → up-proj."""

    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, d_model)
        # Zero-init so the adapter is identity at start
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class AdaptedBlock(nn.Module):
    """Wraps a frozen ViT block and appends a trainable adapter."""

    def __init__(self, block: nn.Module, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.block = block
        self.adapter = Adapter(d_model, bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = x + self.adapter(x)  # residual connection
        return x


# ── Incremental Classifier Head ───────────────────────────────────────────

class IncrementalClassifier(nn.Module):
    """Linear classifier that can be expanded when new classes arrive."""

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    @property
    def num_classes(self) -> int:
        return self.fc.out_features

    def expand(self, num_new_classes: int) -> None:
        """Add neurons for new classes, preserving old weights."""
        old = self.fc
        new = nn.Linear(old.in_features, old.out_features + num_new_classes,
                        device=old.weight.device)
        # Copy old weights
        new.weight.data[:old.out_features] = old.weight.data
        new.bias.data[:old.out_features] = old.bias.data
        # Initialize new class weights to match the scale of old weights
        # so that initial logits for new classes are not wildly different.
        old_std = old.weight.data.std().item()
        nn.init.normal_(new.weight.data[old.out_features:], mean=0.0, std=old_std)
        nn.init.zeros_(new.bias.data[old.out_features:])
        self.fc = new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ── Full CIL Model ────────────────────────────────────────────────────────

class CILModel(nn.Module):
    """
    Complete CIL model: frozen ViT backbone + adapters/MoE + incremental head.

    Args:
        use_moe: If True, use MoE adapter layers instead of single adapters.
        num_experts: Number of initial experts per MoE layer.
        top_k: Number of experts activated per sample.
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        feature_dim: int = 768,
        adapter_bottleneck: int = 64,
        init_classes: int = 50,
        use_moe: bool = False,
        num_experts: int = 2,
        top_k: int = 1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_moe = use_moe

        # Load pre-trained ViT and remove its classification head
        vit = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

        # ── Module A: Freeze entire backbone ──
        for param in vit.parameters():
            param.requires_grad = False

        # ── Module C: Insert adapters into each transformer block ──
        adapted_blocks = nn.Sequential()
        for i, block in enumerate(vit.blocks):
            if use_moe:
                adapted_blocks.add_module(
                    str(i), MoEAdaptedBlock(
                        block, feature_dim, adapter_bottleneck,
                        num_experts, top_k,
                    )
                )
            else:
                adapted_blocks.add_module(
                    str(i), AdaptedBlock(block, feature_dim, adapter_bottleneck)
                )
        vit.blocks = adapted_blocks

        self.backbone = vit
        self.classifier = IncrementalClassifier(feature_dim, init_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract d-dimensional features from the backbone."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward: image → features → logits."""
        features = self.extract_features(x)
        return self.classifier(features)

    def expand_classes(self, num_new_classes: int) -> None:
        """Expand classifier head for a new incremental phase."""
        self.classifier.expand(num_new_classes)

    def add_moe_experts(self, num_new: int = 1) -> None:
        """Add new experts to all MoE layers (for new tasks)."""
        if not self.use_moe:
            return
        for block in self.backbone.blocks:
            if isinstance(block, MoEAdaptedBlock):
                block.moe_adapter.add_experts(num_new)

    def get_moe_balance_loss(self) -> torch.Tensor:
        """Aggregate load-balancing loss across all MoE layers."""
        if not self.use_moe:
            return torch.tensor(0.0)
        total = torch.tensor(0.0)
        for block in self.backbone.blocks:
            if isinstance(block, MoEAdaptedBlock):
                bl = block.moe_adapter.get_balance_loss()
                total = total + bl.cpu() if bl.device != total.device else total + bl
        return total

    def get_trainable_params(self) -> list:
        """Return only adapter/MoE + classifier parameters for the optimizer."""
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def freeze_copy(self) -> "CILModel":
        """Return a frozen copy of the current model (for KD)."""
        old_model = copy.deepcopy(self)
        for param in old_model.parameters():
            param.requires_grad = False
        old_model.eval()
        return old_model

    def get_routing_stats(self) -> list:
        """Get routing statistics from all MoE layers."""
        stats = []
        if not self.use_moe:
            return stats
        for i, block in enumerate(self.backbone.blocks):
            if isinstance(block, MoEAdaptedBlock):
                s = block.moe_adapter.get_routing_stats()
                s["block"] = i
                stats.append(s)
        return stats

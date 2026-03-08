"""Frozen ViT backbone with PEFT adapters and incremental classifier."""

import copy
from typing import Optional

import timm
import torch
import torch.nn as nn

from models.moe_adapter import MoEAdaptedBlock


class Adapter(nn.Module):
    """Bottleneck adapter: down → GELU → up (zero-init)."""

    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class AdaptedBlock(nn.Module):

    def __init__(self, block: nn.Module, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.block = block
        self.adapter = Adapter(d_model, bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = x + self.adapter(x)
        return x


class IncrementalClassifier(nn.Module):
    """Expandable linear classifier for incremental learning."""

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    @property
    def num_classes(self) -> int:
        return self.fc.out_features

    def expand(self, num_new_classes: int, scale_matched_init: bool = True) -> None:
        old = self.fc
        new = nn.Linear(old.in_features, old.out_features + num_new_classes,
                        device=old.weight.device)
        new.weight.data[:old.out_features] = old.weight.data
        new.bias.data[:old.out_features] = old.bias.data
        if scale_matched_init:
            old_std = old.weight.data.std().item()
            nn.init.normal_(new.weight.data[old.out_features:], mean=0.0, std=old_std)
            nn.init.zeros_(new.bias.data[old.out_features:])
        self.fc = new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CILModel(nn.Module):
    """Frozen ViT + adapters/MoE + incremental classifier head."""

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

        vit = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

        for param in vit.parameters():
            param.requires_grad = False

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
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(features)

    def expand_classes(self, num_new_classes: int, scale_matched_head_init: bool = True) -> None:
        self.classifier.expand(num_new_classes, scale_matched_init=scale_matched_head_init)

    def add_moe_experts(self, num_new: int = 1) -> None:
        if not self.use_moe:
            return
        for block in self.backbone.blocks:
            if isinstance(block, MoEAdaptedBlock):
                block.moe_adapter.add_experts(num_new)

    def get_moe_balance_loss(self) -> torch.Tensor:
        if not self.use_moe:
            return torch.tensor(0.0)
        total = torch.tensor(0.0)
        for block in self.backbone.blocks:
            if isinstance(block, MoEAdaptedBlock):
                bl = block.moe_adapter.get_balance_loss()
                total = total + bl.cpu() if bl.device != total.device else total + bl
        return total

    def get_moe_trigger_alignment_loss(self, trigger_strength: torch.Tensor) -> torch.Tensor:
        if not self.use_moe:
            return torch.tensor(0.0, device=trigger_strength.device)
        total = torch.tensor(0.0, device=trigger_strength.device)
        for block in self.backbone.blocks:
            if isinstance(block, MoEAdaptedBlock):
                total = total + block.moe_adapter.get_trigger_alignment_loss(trigger_strength)
        return total

    def get_trainable_params(self) -> list:
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def freeze_copy(self) -> "CILModel":
        old_model = copy.deepcopy(self)
        for param in old_model.parameters():
            param.requires_grad = False
        old_model.eval()
        return old_model

    def get_routing_stats(self) -> list:
        stats = []
        if not self.use_moe:
            return stats
        for i, block in enumerate(self.backbone.blocks):
            if isinstance(block, MoEAdaptedBlock):
                s = block.moe_adapter.get_routing_stats()
                s["block"] = i
                stats.append(s)
        return stats

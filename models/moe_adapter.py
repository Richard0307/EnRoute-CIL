"""MoE adapter layer with top-k routing and dynamic expert growth."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertAdapter(nn.Module):

    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class MoERouter(nn.Module):
    """Top-k routing gate for MoE."""

    def __init__(self, d_model: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            pooled = x.mean(dim=1)  # (B, d)
        else:
            pooled = x

        logits = self.gate(pooled)
        gate_scores = F.softmax(logits, dim=-1)

        top_weights, top_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return gate_scores, top_indices, top_weights

    def expand(self, num_new_experts: int) -> None:
        old = self.gate
        new_num = old.out_features + num_new_experts
        new_gate = nn.Linear(old.in_features, new_num, bias=False,
                             device=old.weight.device)
        new_gate.weight.data[:old.out_features] = old.weight.data
        nn.init.normal_(new_gate.weight.data[old.out_features:], std=0.01)
        self.gate = new_gate
        self.num_experts = new_num


class MoEAdapterLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        bottleneck: int = 64,
        num_experts: int = 2,
        top_k: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.bottleneck = bottleneck
        self.top_k = top_k

        self.experts = nn.ModuleList([
            ExpertAdapter(d_model, bottleneck) for _ in range(num_experts)
        ])
        self.router = MoERouter(d_model, num_experts, top_k)

        self.balance_loss_coeff = 0.01
        self._last_balance_loss = torch.tensor(0.0)

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def add_experts(self, num_new: int = 1) -> None:
        device = self.experts[0].down.weight.device
        for _ in range(num_new):
            expert = ExpertAdapter(self.d_model, self.bottleneck).to(device)
            self.experts.append(expert)
        self.router.expand(num_new)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_scores, top_indices, top_weights = self.router(x)
        self._last_balance_loss = self._compute_balance_loss(gate_scores)

        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_indices[:, k]
            weight = top_weights[:, k]

            for e_id in range(self.num_experts):
                mask = (expert_idx == e_id)
                if not mask.any():
                    continue

                if x.dim() == 3:
                    expert_input = x[mask]
                    expert_out = self.experts[e_id](expert_input)
                    w = weight[mask].unsqueeze(-1).unsqueeze(-1)
                    output[mask] += expert_out * w
                else:
                    expert_input = x[mask]
                    expert_out = self.experts[e_id](expert_input)
                    w = weight[mask].unsqueeze(-1)
                    output[mask] += expert_out * w

        return output

    def _compute_balance_loss(self, gate_scores: torch.Tensor) -> torch.Tensor:
        avg_gate = gate_scores.mean(dim=0)
        num_experts = gate_scores.shape[1]
        target = 1.0 / num_experts
        balance_loss = num_experts * (avg_gate * avg_gate).sum()
        return self.balance_loss_coeff * balance_loss

    def get_balance_loss(self) -> torch.Tensor:
        return self._last_balance_loss

    def get_routing_stats(self) -> dict:
        return {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
        }


class MoEAdaptedBlock(nn.Module):

    def __init__(
        self,
        block: nn.Module,
        d_model: int,
        bottleneck: int = 64,
        num_experts: int = 2,
        top_k: int = 1,
    ):
        super().__init__()
        self.block = block
        self.moe_adapter = MoEAdapterLayer(d_model, bottleneck, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = x + self.moe_adapter(x)
        return x

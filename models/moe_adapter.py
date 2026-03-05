"""
Module C: Energy-Routed MoE PEFT Updater.

Replaces the single shared adapter with a Mixture-of-Experts (MoE)
architecture. Each expert is a lightweight adapter; a learned routing
gate selects which expert(s) process each sample.

Key design choices:
    - Top-k routing (default k=1) to keep inference cost controlled.
    - Dormant expert activation: new experts can be added for new tasks.
    - Load-balancing auxiliary loss to prevent expert collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertAdapter(nn.Module):
    """Single lightweight adapter expert: down → GELU → up."""

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
    """
    Learned routing gate for MoE.

    For each token/sample, produces a probability distribution over experts.
    Top-k experts are selected for computation.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, d_model) or (B, seq, d_model) input features.

        Returns:
            gate_scores: (B, num_experts) softmax probabilities.
            top_indices: (B, top_k) selected expert indices.
            top_weights: (B, top_k) normalized weights for selected experts.
        """
        # Pool over sequence dimension if present
        if x.dim() == 3:
            pooled = x.mean(dim=1)  # (B, d)
        else:
            pooled = x

        logits = self.gate(pooled)  # (B, num_experts)
        gate_scores = F.softmax(logits, dim=-1)

        top_weights, top_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        # Renormalize top-k weights
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return gate_scores, top_indices, top_weights

    def expand(self, num_new_experts: int) -> None:
        """Add new expert slots to the routing gate."""
        old = self.gate
        new_num = old.out_features + num_new_experts
        new_gate = nn.Linear(old.in_features, new_num, bias=False,
                             device=old.weight.device)
        new_gate.weight.data[:old.out_features] = old.weight.data
        # Initialize new expert gates with small random values
        nn.init.normal_(new_gate.weight.data[old.out_features:], std=0.01)
        self.gate = new_gate
        self.num_experts = new_num


class MoEAdapterLayer(nn.Module):
    """
    MoE adapter layer with top-k routing.

    Replaces a single adapter with multiple experts + a router.
    Supports dynamic expert addition for new tasks.
    """

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

        # Load-balancing loss coefficient
        self.balance_loss_coeff = 0.01
        self._last_balance_loss = torch.tensor(0.0)

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def add_experts(self, num_new: int = 1) -> None:
        """Add new dormant experts for a new task."""
        device = self.experts[0].down.weight.device
        for _ in range(num_new):
            expert = ExpertAdapter(self.d_model, self.bottleneck).to(device)
            self.experts.append(expert)
        self.router.expand(num_new)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq, d) or (B, d) input.

        Returns:
            output: same shape as x, weighted sum of expert outputs.
        """
        gate_scores, top_indices, top_weights = self.router(x)

        # Compute load-balancing loss
        self._last_balance_loss = self._compute_balance_loss(gate_scores)

        # Compute weighted expert outputs
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_indices[:, k]  # (B,)
            weight = top_weights[:, k]      # (B,)

            # Group samples by expert for efficiency
            for e_id in range(self.num_experts):
                mask = (expert_idx == e_id)
                if not mask.any():
                    continue

                if x.dim() == 3:
                    expert_input = x[mask]            # (n, seq, d)
                    expert_out = self.experts[e_id](expert_input)
                    w = weight[mask].unsqueeze(-1).unsqueeze(-1)  # (n, 1, 1)
                    output[mask] += expert_out * w
                else:
                    expert_input = x[mask]            # (n, d)
                    expert_out = self.experts[e_id](expert_input)
                    w = weight[mask].unsqueeze(-1)    # (n, 1)
                    output[mask] += expert_out * w

        return output

    def _compute_balance_loss(self, gate_scores: torch.Tensor) -> torch.Tensor:
        """
        Load-balancing loss to prevent expert collapse.

        Encourages uniform expert utilization across the batch.
        """
        # Average gate probability per expert across batch
        avg_gate = gate_scores.mean(dim=0)  # (num_experts,)
        # Fraction of samples each expert is selected for
        num_experts = gate_scores.shape[1]
        target = 1.0 / num_experts
        balance_loss = num_experts * (avg_gate * avg_gate).sum()
        return self.balance_loss_coeff * balance_loss

    def get_balance_loss(self) -> torch.Tensor:
        return self._last_balance_loss

    def get_routing_stats(self) -> dict:
        """Return routing statistics for diagnostics."""
        return {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
        }


class MoEAdaptedBlock(nn.Module):
    """Wraps a frozen ViT block with a MoE adapter layer."""

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

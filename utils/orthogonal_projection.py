"""
Module D extension: Orthogonal Projection for gradient updates.

Projects new-task gradients onto the null space of the historical
parameter subspace, preventing interference with previously learned
representations.

Reference: OPLoRA / Orthogonal Projection for Continual Learning.

Core formula:
    ∇W_proj = (I - U_old @ U_old^T) @ ∇W_new

Where U_old contains the principal directions of old-task weight updates,
obtained via SVD of accumulated weight changes.
"""

from typing import Dict, List

import torch
import torch.nn as nn


class OrthogonalProjector:
    """
    Manages orthogonal projection bases for each trainable parameter.

    After each task, records the weight change direction and accumulates
    the historical subspace. During training on new tasks, projects
    gradients to avoid interfering with old-task directions.
    """

    def __init__(self, max_rank: int = 20):
        """
        Args:
            max_rank: Maximum rank of the historical subspace per parameter.
                      Controls memory usage and projection strength.
        """
        self.max_rank = max_rank
        # param_name → (U_old basis matrix, shape depends on param)
        self._bases: Dict[str, torch.Tensor] = {}
        # param_name → snapshot of param values before current task
        self._snapshots: Dict[str, torch.Tensor] = {}

    def snapshot_params(self, model: nn.Module) -> None:
        """
        Take a snapshot of current trainable parameters before training.

        Call this at the start of each new task (before optimizer step).
        """
        self._snapshots = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._snapshots[name] = param.data.detach().clone()

    def update_bases(self, model: nn.Module) -> None:
        """
        Update the historical subspace after training a task.

        Computes the weight change delta, extracts its principal directions
        via SVD, and merges them into the accumulated basis.

        Call this after each task's training is complete.
        """
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self._snapshots:
                continue

            # Compute weight change
            delta = (param.data - self._snapshots[name]).detach().cpu()

            # Flatten to 2D for SVD (handle both weight matrices and bias vectors)
            original_shape = delta.shape
            if delta.dim() == 1:
                delta_2d = delta.unsqueeze(0)  # (1, d)
            else:
                delta_2d = delta.reshape(original_shape[0], -1)  # (out, in)

            # SVD of the change direction
            try:
                U, S, _ = torch.linalg.svd(delta_2d, full_matrices=False)
            except RuntimeError:
                continue

            # Keep only significant directions (above threshold)
            threshold = S.max() * 0.01 if S.numel() > 0 and S.max() > 0 else 0
            significant = S > threshold
            k = min(int(significant.sum().item()), self.max_rank)
            if k == 0:
                continue

            new_dirs = U[:, :k]  # (out, k)

            # Merge with existing basis
            if name in self._bases:
                old_basis = self._bases[name]
                combined = torch.cat([old_basis, new_dirs], dim=1)
                # Re-orthogonalize and trim to max_rank
                try:
                    U_combined, S_combined, _ = torch.linalg.svd(
                        combined, full_matrices=False
                    )
                except RuntimeError:
                    continue
                keep = min(self.max_rank, U_combined.shape[1])
                self._bases[name] = U_combined[:, :keep]
            else:
                self._bases[name] = new_dirs

        # Clear snapshots
        self._snapshots = {}

    @torch.no_grad()
    def project_gradients(self, model: nn.Module) -> None:
        """
        Project gradients onto the null space of historical directions.

        ∇W_proj = (I - U_old @ U_old^T) @ ∇W_new

        Call this after loss.backward() but before optimizer.step().
        """
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if name not in self._bases:
                continue

            U_old = self._bases[name].to(param.grad.device)
            grad = param.grad

            original_shape = grad.shape
            if grad.dim() == 1:
                grad_2d = grad.unsqueeze(0)
            else:
                grad_2d = grad.reshape(original_shape[0], -1)

            # Project: g_proj = g - U @ U^T @ g
            projection = U_old @ (U_old.t() @ grad_2d.t())  # (out, batch)
            grad_projected = grad_2d - projection.t()

            param.grad.data = grad_projected.reshape(original_shape)

    def has_basis(self) -> bool:
        """Whether any historical basis has been accumulated."""
        return len(self._bases) > 0

    def get_stats(self) -> Dict[str, int]:
        """Return statistics about the stored bases."""
        stats = {}
        for name, basis in self._bases.items():
            stats[name] = basis.shape[1]
        return stats

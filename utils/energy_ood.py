"""
Module B: Energy-Based Open-World Detector.

Computes energy scores on classifier logits to detect OOD samples.
Reference: "Energy-based Out-of-distribution Detection" (Liu et al., 2020)

Energy score:
    E(h) = -T * log( sum_i exp(f_i(h) / T) )

Low energy → in-distribution (confident prediction).
High energy → out-of-distribution (novel / unknown class).
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_energy_scores(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute energy score for a batch of logits.

    E(h) = -T * log( sum_i exp(f_i(h) / T) )

    Args:
        logits: (B, C) raw classifier outputs.
        temperature: Temperature scaling parameter.

    Returns:
        energies: (B,) energy scores. Higher = more likely OOD.
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


@torch.no_grad()
def collect_energy_scores(
    model: nn.Module,
    dataloader: DataLoader,
    temperature: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:
    """
    Collect energy scores for all samples in a dataloader.

    Returns:
        scores: (N,) numpy array of energy scores.
    """
    model.eval()
    all_scores = []
    for images, _ in dataloader:
        images = images.to(device)
        logits = model(images)
        scores = compute_energy_scores(logits, temperature)
        all_scores.append(scores.cpu().numpy())
    return np.concatenate(all_scores, axis=0)


def calibrate_threshold(
    id_scores: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """
    Calibrate OOD threshold from in-distribution energy scores.

    Sets tau so that `percentile`% of ID samples fall below the threshold.
    Samples with energy > tau are flagged as OOD.

    Args:
        id_scores: Energy scores from in-distribution data.
        percentile: Percentage of ID data to keep (e.g., 95 means
                    5% false positive rate on ID data).

    Returns:
        tau: Energy threshold.
    """
    return float(np.percentile(id_scores, percentile))


def evaluate_ood(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate OOD detection quality using energy scores.

    Metrics:
        - AUROC: Area under ROC curve (higher is better).
        - FPR@95TPR: False positive rate when 95% of OOD are detected.

    Args:
        id_scores:  Energy scores from in-distribution data.
        ood_scores: Energy scores from out-of-distribution data.

    Returns:
        Dictionary with AUROC and FPR@95TPR.
    """
    # Labels: 0 = ID, 1 = OOD
    # Higher energy → more OOD, so we use energy as the "positive" score
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])

    # Sort by score descending (highest energy first)
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]

    # Compute ROC
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))

    if n_pos == 0 or n_neg == 0:
        return {"auroc": 0.0, "fpr_at_95tpr": 1.0}

    tp = 0
    fp = 0
    tpr_list = []
    fpr_list = []

    for lab in sorted_labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)

    # AUROC via trapezoidal rule
    auroc = float(np.trapz(tpr_arr, fpr_arr))

    # FPR@95TPR: find the FPR when TPR first reaches 0.95
    idx_95 = np.searchsorted(tpr_arr, 0.95)
    if idx_95 >= len(fpr_arr):
        idx_95 = len(fpr_arr) - 1
    fpr_at_95tpr = float(fpr_arr[idx_95])

    return {"auroc": auroc, "fpr_at_95tpr": fpr_at_95tpr}

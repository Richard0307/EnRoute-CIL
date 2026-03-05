"""
Standard CIL evaluation metrics: Average Accuracy (AA) and Average Forgetting (AF).

Reference: blueprint.md Section 3.
"""

from typing import Dict, List

import numpy as np


def compute_average_accuracy(acc_matrix: np.ndarray) -> float:
    """
    Average Accuracy after the last task.

    AA = (1/T) * sum_{i=1}^{T} A_{T,i}

    Args:
        acc_matrix: (T, T) matrix where acc_matrix[t][i] = accuracy on task i
                    after training task t. Upper-triangular entries are 0.

    Returns:
        AA value as a float.
    """
    T = acc_matrix.shape[0]
    last_row = acc_matrix[T - 1, :T]
    return float(np.mean(last_row))


def compute_average_forgetting(acc_matrix: np.ndarray) -> float:
    """
    Average Forgetting after the last task.

    AF = (1/(T-1)) * sum_{i=1}^{T-1} max_{t in {1..T-1}} (A_{t,i} - A_{T,i})

    Args:
        acc_matrix: (T, T) matrix.

    Returns:
        AF value (lower is better).
    """
    T = acc_matrix.shape[0]
    if T <= 1:
        return 0.0

    forgetting = []
    for i in range(T - 1):  # task 0 to T-2
        # Best accuracy on task i across all stages before T
        best_before = max(acc_matrix[t][i] for t in range(i, T - 1))
        # Accuracy on task i after last stage
        final = acc_matrix[T - 1][i]
        forgetting.append(best_before - final)

    return float(np.mean(forgetting))


def print_metrics(acc_matrix: np.ndarray, task_id: int) -> None:
    """Print current AA and AF after a given task."""
    T = task_id + 1
    sub = acc_matrix[:T, :T]
    aa = compute_average_accuracy(sub)
    af = compute_average_forgetting(sub)
    print(f"  [Task {task_id}] Average Accuracy: {aa:.4f} | "
          f"Average Forgetting: {af:.4f}")

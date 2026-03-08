"""
Generate AA/AF curves and accuracy heatmap for research proposal.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --matrix output/acc_matrix.npy
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def plot_task_accuracy_over_time(acc_matrix: np.ndarray, output_dir: Path) -> Path:
    """Plot each task's accuracy as training progresses through tasks."""
    T = acc_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8, 5))

    for task_i in range(T):
        # Task i's accuracy from the time it was learned onward
        x = list(range(task_i, T))
        y = [acc_matrix[t][task_i] * 100 for t in range(task_i, T)]
        ax.plot(x, y, marker="o", linewidth=2, markersize=6, label=f"Task {task_i}")

    ax.set_xlabel("After Training Task")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Per-Task Accuracy Over Incremental Learning")
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"Task {i}" for i in range(T)])
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([50, 102])

    path = output_dir / "task_accuracy_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_aa_af_progression(acc_matrix: np.ndarray, output_dir: Path) -> Path:
    """Plot AA and AF after each task."""
    T = acc_matrix.shape[0]

    aa_values = []
    af_values = []

    for t in range(T):
        # AA after task t
        aa = np.mean(acc_matrix[t, :t + 1]) * 100
        aa_values.append(aa)

        # AF after task t
        if t == 0:
            af_values.append(0.0)
        else:
            forgetting = []
            for i in range(t):
                best = max(acc_matrix[s][i] for s in range(i, t))
                forgetting.append((best - acc_matrix[t][i]) * 100)
            af_values.append(np.mean(forgetting))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_aa = "#2196F3"
    color_af = "#F44336"

    ax1.plot(range(T), aa_values, color=color_aa, marker="s", linewidth=2.5,
             markersize=8, label="Average Accuracy (AA)", zorder=5)
    ax1.set_xlabel("After Training Task")
    ax1.set_ylabel("Average Accuracy (%)", color=color_aa)
    ax1.tick_params(axis="y", labelcolor=color_aa)
    ax1.set_ylim([60, 100])

    ax2 = ax1.twinx()
    ax2.plot(range(T), af_values, color=color_af, marker="^", linewidth=2.5,
             markersize=8, label="Average Forgetting (AF)", zorder=5)
    ax2.set_ylabel("Average Forgetting (%)", color=color_af)
    ax2.tick_params(axis="y", labelcolor=color_af)
    ax2.set_ylim([0, 30])

    ax1.set_xticks(range(T))
    ax1.set_xticklabels([f"Task {i}" for i in range(T)])
    ax1.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    ax1.set_title("Average Accuracy & Forgetting Over Incremental Tasks")

    # Annotate final values
    ax1.annotate(f"AA={aa_values[-1]:.1f}%", xy=(T - 1, aa_values[-1]),
                 xytext=(T - 1.8, aa_values[-1] - 5), fontsize=11, fontweight="bold",
                 color=color_aa)
    ax2.annotate(f"AF={af_values[-1]:.1f}%", xy=(T - 1, af_values[-1]),
                 xytext=(T - 1.8, af_values[-1] + 3), fontsize=11, fontweight="bold",
                 color=color_af)

    path = output_dir / "aa_af_progression.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_accuracy_heatmap(acc_matrix: np.ndarray, output_dir: Path) -> Path:
    """Plot accuracy matrix as a heatmap."""
    T = acc_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6))

    # Mask upper triangle (not yet learned)
    display = acc_matrix.copy() * 100
    mask = np.triu(np.ones_like(display, dtype=bool), k=1)
    display[mask] = np.nan

    im = ax.imshow(display, cmap="YlGnBu", vmin=50, vmax=100, aspect="equal")

    # Add text annotations
    for i in range(T):
        for j in range(T):
            if not mask[i][j]:
                val = display[i][j]
                color = "white" if val > 85 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels([f"Task {i}" for i in range(T)])
    ax.set_yticklabels([f"After Task {i}" for i in range(T)])
    ax.set_xlabel("Evaluated On")
    ax.set_ylabel("Trained Up To")
    ax.set_title("Accuracy Matrix (%)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy (%)")

    path = output_dir / "accuracy_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _load_runtime_metrics(metrics_path: Path) -> dict[str, np.ndarray] | None:
    """Load runtime metrics if available."""
    if not metrics_path.exists():
        return None

    with np.load(metrics_path) as data:
        metrics = {key: data[key] for key in data.files}
    return metrics


def plot_max_memory_allocated(runtime_metrics: dict[str, np.ndarray], output_dir: Path) -> Path:
    """Plot max allocated memory after each task."""
    task_ids = runtime_metrics["task_ids"]
    max_memory_mb = runtime_metrics["task_max_memory_mb"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(task_ids, max_memory_mb, marker="o", linewidth=2.5, color="#5E35B1")
    ax.set_xlabel("After Training Task")
    ax.set_ylabel("Max Memory Allocated (MB)")
    ax.set_title("Max Memory Allocated by Task")
    ax.set_xticks(task_ids)
    ax.set_xticklabels([f"Task {i}" for i in task_ids])
    ax.grid(True, alpha=0.3)

    path = output_dir / "max_memory_allocated.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_old_task_accuracy(runtime_metrics: dict[str, np.ndarray], output_dir: Path) -> Path:
    """Plot mean accuracy on old tasks after each incremental stage."""
    task_ids = runtime_metrics["task_ids"]
    old_acc = runtime_metrics["task_old_task_accuracy"] * 100.0
    valid = np.isfinite(old_acc)

    fig, ax = plt.subplots(figsize=(8, 5))
    if np.any(valid):
        ax.plot(task_ids[valid], old_acc[valid], marker="s", linewidth=2.5, color="#00897B")
        y_min = max(0.0, float(np.min(old_acc[valid]) - 5.0))
        y_max = min(100.0, float(np.max(old_acc[valid]) + 5.0))
        if y_max - y_min < 5.0:
            y_max = min(100.0, y_min + 5.0)
        ax.set_ylim([y_min, y_max])
    else:
        ax.text(0.5, 0.5, "N/A (only base task)", transform=ax.transAxes,
                ha="center", va="center", fontsize=12)

    ax.set_xlabel("After Training Task")
    ax.set_ylabel("Accuracy on Old Tasks (%)")
    ax.set_title("Old-Task Accuracy Progression")
    ax.set_xticks(task_ids)
    ax.set_xticklabels([f"Task {i}" for i in task_ids])
    ax.grid(True, alpha=0.3)

    path = output_dir / "accuracy_on_old_tasks.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_time_per_epoch_latency(runtime_metrics: dict[str, np.ndarray], output_dir: Path) -> Path:
    """Plot average time per epoch and latency per batch across tasks."""
    task_ids = runtime_metrics["task_ids"]
    epoch_time = runtime_metrics["task_avg_epoch_time_sec"]
    latency_ms = runtime_metrics["task_avg_batch_latency_ms"]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_time = "#1565C0"
    color_lat = "#EF6C00"

    ax1.plot(task_ids, epoch_time, marker="o", linewidth=2.5,
             color=color_time, label="Avg Time per Epoch (s)")
    ax1.set_xlabel("After Training Task")
    ax1.set_ylabel("Avg Time per Epoch (s)", color=color_time)
    ax1.tick_params(axis="y", labelcolor=color_time)
    ax1.set_xticks(task_ids)
    ax1.set_xticklabels([f"Task {i}" for i in task_ids])
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(task_ids, latency_ms, marker="^", linewidth=2.5,
             color=color_lat, label="Avg Latency (ms/batch)")
    ax2.set_ylabel("Avg Latency (ms/batch)", color=color_lat)
    ax2.tick_params(axis="y", labelcolor=color_lat)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    ax1.set_title("Time per Epoch and Latency by Task")

    path = output_dir / "time_per_epoch_latency.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_ood_metrics(runtime_metrics: dict[str, np.ndarray], output_dir: Path) -> Path:
    """Plot OOD detection metrics (AUROC and FPR@95TPR) across tasks."""
    task_ids = runtime_metrics["ood_task_ids"]
    auroc = runtime_metrics["ood_auroc"]
    fpr = runtime_metrics["ood_fpr_at_95tpr"]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_auroc = "#4CAF50"
    color_fpr = "#FF5722"

    ax1.plot(task_ids, auroc * 100, marker="o", linewidth=2.5,
             color=color_auroc, label="AUROC (%)")
    ax1.set_xlabel("After Training Task")
    ax1.set_ylabel("AUROC (%)", color=color_auroc)
    ax1.tick_params(axis="y", labelcolor=color_auroc)
    ax1.set_ylim([50, 102])

    ax2 = ax1.twinx()
    ax2.plot(task_ids, fpr * 100, marker="^", linewidth=2.5,
             color=color_fpr, label="FPR@95TPR (%)")
    ax2.set_ylabel("FPR@95TPR (%)", color=color_fpr)
    ax2.tick_params(axis="y", labelcolor=color_fpr)
    ax2.set_ylim([0, 100])

    ax1.set_xticks(task_ids)
    ax1.set_xticklabels([f"Task {i}" for i in task_ids])
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)
    ax1.set_title("OOD Detection Quality (Energy-Based)")

    path = output_dir / "ood_metrics.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_ood_threshold_stability(runtime_metrics: dict[str, np.ndarray], output_dir: Path) -> Path:
    """Plot dynamic OOD threshold against ID/OOD energy means."""
    task_ids = runtime_metrics["ood_task_ids"]
    threshold = runtime_metrics["ood_threshold"]
    id_mean = runtime_metrics["ood_id_mean_energy"]
    ood_mean = runtime_metrics["ood_ood_mean_energy"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(task_ids, threshold, marker="o", linewidth=2.5, color="#6A1B9A", label="Dynamic threshold")
    ax.plot(task_ids, id_mean, marker="s", linewidth=2.0, color="#00897B", label="ID mean energy")
    ax.plot(task_ids, ood_mean, marker="^", linewidth=2.0, color="#EF6C00", label="OOD mean energy")
    ax.set_xlabel("After Training Task")
    ax.set_ylabel("Energy")
    ax.set_title("Dynamic OOD Threshold Stability")
    ax.set_xticks(task_ids)
    ax.set_xticklabels([f"Task {i}" for i in task_ids])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    path = output_dir / "ood_threshold_stability.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_ood_trigger_dynamics(runtime_metrics: dict[str, np.ndarray], output_dir: Path) -> Path:
    """Plot offline OOD trigger ratio and expert activation events."""
    task_ids = runtime_metrics["ood_trigger_task_ids"]
    flagged_ratio = runtime_metrics["ood_trigger_flagged_ratio"] * 100.0
    activated = runtime_metrics["ood_trigger_active"]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_ratio = "#3949AB"
    color_active = "#C62828"

    ax1.plot(task_ids, flagged_ratio, marker="o", linewidth=2.5,
             color=color_ratio, label="Flagged ratio (%)")
    ax1.set_xlabel("Current Task")
    ax1.set_ylabel("Flagged ratio (%)", color=color_ratio)
    ax1.tick_params(axis="y", labelcolor=color_ratio)
    ax1.set_xticks(task_ids)
    ax1.set_xticklabels([f"Task {i}" for i in task_ids])
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.step(task_ids, activated, where="mid", linewidth=2.0,
             color=color_active, label="Expert activated")
    ax2.set_ylabel("Activation", color=color_active)
    ax2.tick_params(axis="y", labelcolor=color_active)
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_yticks([0, 1])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    ax1.set_title("Offline OOD Trigger Dynamics")

    path = output_dir / "ood_trigger_dynamics.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _compute_final_aa_af(acc_matrix: np.ndarray) -> tuple[float, float]:
    """Compute final AA/AF in percentage for reporting."""
    T = acc_matrix.shape[0]
    aa = float(np.mean(acc_matrix[T - 1, :T]) * 100)

    if T <= 1:
        return aa, 0.0

    af_list = []
    for i in range(T - 1):
        best = max(acc_matrix[s][i] for s in range(i, T - 1))
        af_list.append((best - acc_matrix[T - 1][i]) * 100)
    af = float(np.mean(af_list)) if af_list else 0.0
    return aa, af


def generate_all_plots(
    acc_matrix: np.ndarray,
    output_dir: Path | str,
    runtime_metrics_path: Path | str | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    """Generate all plots and return output paths."""
    if acc_matrix.ndim != 2 or acc_matrix.shape[0] != acc_matrix.shape[1]:
        raise ValueError(f"Expected square 2D acc_matrix, got shape={acc_matrix.shape}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aa, af = _compute_final_aa_af(acc_matrix)
    if verbose:
        print(f"Loaded accuracy matrix: {acc_matrix.shape}")
        print(f"Final AA: {aa:.2f}%  |  Final AF: {af:.2f}%")
        print()

    plot_paths = {
        "task_accuracy_curves": plot_task_accuracy_over_time(acc_matrix, output_dir),
        "aa_af_progression": plot_aa_af_progression(acc_matrix, output_dir),
        "accuracy_heatmap": plot_accuracy_heatmap(acc_matrix, output_dir),
    }

    if runtime_metrics_path is None:
        runtime_metrics_path = output_dir / "training_metrics.npz"
    else:
        runtime_metrics_path = Path(runtime_metrics_path)

    runtime_metrics = _load_runtime_metrics(runtime_metrics_path)
    if runtime_metrics is not None:
        plot_paths["max_memory_allocated"] = plot_max_memory_allocated(runtime_metrics, output_dir)
        plot_paths["accuracy_on_old_tasks"] = plot_old_task_accuracy(runtime_metrics, output_dir)
        plot_paths["time_per_epoch_latency"] = plot_time_per_epoch_latency(runtime_metrics, output_dir)
        # OOD metrics plots (Module B)
        if "ood_auroc" in runtime_metrics:
            plot_paths["ood_metrics"] = plot_ood_metrics(runtime_metrics, output_dir)
        if "ood_threshold" in runtime_metrics:
            plot_paths["ood_threshold_stability"] = plot_ood_threshold_stability(runtime_metrics, output_dir)
        if "ood_trigger_task_ids" in runtime_metrics:
            plot_paths["ood_trigger_dynamics"] = plot_ood_trigger_dynamics(runtime_metrics, output_dir)
    elif verbose:
        print(f"Runtime metrics not found at {runtime_metrics_path}; skipped runtime plots.")

    if verbose:
        for path in plot_paths.values():
            print(f"Saved: {path}")
        print(f"\nAll plots saved to {output_dir}/")

    return plot_paths


def main():
    parser = argparse.ArgumentParser(description="Plot CIL experiment results")
    parser.add_argument("--matrix", type=str, default="output/acc_matrix.npy",
                        help="Path to accuracy matrix .npy file")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save plots")
    parser.add_argument("--metrics", type=str, default="output/training_metrics.npz",
                        help="Path to runtime metrics .npz file")
    args = parser.parse_args()

    acc_matrix = np.load(args.matrix)
    generate_all_plots(
        acc_matrix=acc_matrix,
        output_dir=args.output_dir,
        runtime_metrics_path=args.metrics,
        verbose=True,
    )


if __name__ == "__main__":
    main()

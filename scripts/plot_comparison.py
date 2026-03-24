"""
Generate publication-quality comparison figures directly from acc_matrix.npy files.

Produces three figures:
  1. acc_heatmaps_<dataset>.png   -- side-by-side acc matrix heatmaps (4 methods)
  2. forgetting_curves_<dataset>.png -- per-task accuracy over time (ours only)
  3. aa_af_tradeoff.png            -- AA vs AF scatter for both datasets

Usage:
    python scripts/plot_comparison.py
    python scripts/plot_comparison.py --output_dir figures/comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

METHOD_LABELS = {
    "ours":         "Ours (ERO-MoE)",
    "l2p":          "L2P",
    "coda_prompt":  "CodaPrompt",
    "moe_adapters": "MoE-Adapters",
}
METHODS = list(METHOD_LABELS.keys())
COLORS  = ["#e6194b", "#4363d8", "#f58231", "#3cb44b"]
MARKERS = ["o", "s", "^", "D"]


def load_matrix(base: Path, dataset: str, method: str) -> np.ndarray | None:
    p = base / dataset / method / "mean_acc_matrix.npy"
    if not p.exists():
        return None
    return np.load(p)


def aa_af(m: np.ndarray) -> tuple[float, float]:
    n = m.shape[0]
    aa = float(m[n - 1, :n].mean())
    af = float(np.mean([m[i, i] - m[n - 1, i] for i in range(n - 1)]))
    return aa, af


# ── Figure 1: acc matrix heatmaps ──────────────────────────────────────────
def plot_heatmaps(base: Path, dataset: str, output_dir: Path) -> Path:
    matrices = {k: load_matrix(base, dataset, k) for k in METHODS}
    available = {k: v for k, v in matrices.items() if v is not None}
    n_methods = len(available)

    fig, axes = plt.subplots(1, n_methods, figsize=(3.2 * n_methods, 3.2))
    if n_methods == 1:
        axes = [axes]

    for ax, (method, m) in zip(axes, available.items()):
        T = m.shape[0]
        # mask upper triangle (not yet trained)
        mask = np.triu(np.ones_like(m, dtype=bool), k=1)
        display = np.where(mask, np.nan, m)

        im = ax.imshow(display, vmin=0, vmax=1, cmap="RdYlGn", aspect="equal")
        ax.set_title(METHOD_LABELS[method], fontsize=11, pad=6)
        ax.set_xlabel("Tested on Task")
        ax.set_ylabel("After Training Task")
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels([str(i) for i in range(T)], fontsize=8)
        ax.set_yticklabels([str(i) for i in range(T)], fontsize=8)

        for i in range(T):
            for j in range(T):
                if not mask[i, j]:
                    val = m[i, j]
                    color = "white" if val < 0.4 or val > 0.75 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Accuracy", fontsize=8)

    aa_val, af_val = aa_af(available.get("ours", next(iter(available.values()))))
    fig.suptitle(
        f"{dataset.upper()} — Accuracy Matrix Comparison  "
        f"(Ours: AA={aa_val:.1%}, AF={af_val:.1%})",
        fontsize=12, y=1.01,
    )
    out = output_dir / f"acc_heatmaps_{dataset}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ── Figure 2: per-task forgetting curves (ours) ────────────────────────────
def plot_forgetting_curves(base: Path, dataset: str, output_dir: Path) -> Path:
    m = load_matrix(base, dataset, "ours")
    if m is None:
        raise FileNotFoundError(f"No ours matrix for {dataset}")
    T = m.shape[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.get_cmap("tab10", T)

    for task_i in range(T):
        x = list(range(task_i, T))
        y = [m[t, task_i] * 100 for t in range(task_i, T)]
        ax.plot(x, y, marker="o", linewidth=2, markersize=6,
                color=cmap(task_i), label=f"Task {task_i}")

    ax.set_xlabel("After Training Task")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"{dataset.upper()} — Per-Task Accuracy Over Time (Ours)")
    ax.set_xticks(range(T))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    ax.grid(axis="y", alpha=0.3)

    out = output_dir / f"forgetting_curves_{dataset}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ── Figure 3: AA vs AF scatter (both datasets) ─────────────────────────────
def plot_aa_af_tradeoff(base: Path, datasets: list[str], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        for idx, method in enumerate(METHODS):
            m = load_matrix(base, dataset, method)
            if m is None:
                continue
            aa, af = aa_af(m)
            ax.scatter(af * 100, aa * 100,
                       color=COLORS[idx], marker=MARKERS[idx],
                       s=120, zorder=3, label=METHOD_LABELS[method])
            ax.annotate(METHOD_LABELS[method],
                        xy=(af * 100, aa * 100),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8.5)

        ax.set_xlabel("Average Forgetting (AF, %)  ←  lower is better")
        ax.set_ylabel("Average Accuracy (AA, %)  →  higher is better")
        ax.set_title(f"{dataset.upper()}")
        ax.grid(alpha=0.3)
        # draw ideal direction arrow
        ax.annotate("", xy=(0.05, 0.95), xytext=(0.15, 0.80),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
        ax.text(0.06, 0.96, "ideal", transform=ax.transAxes,
                fontsize=8, color="gray", va="bottom")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("AA–AF Tradeoff: Plasticity vs. Stability", fontsize=13)

    out = output_dir / "aa_af_tradeoff.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",   default="output/benchmark_sota")
    parser.add_argument("--output_dir", default="output/comparison_figures")
    parser.add_argument("--datasets",   nargs="+", default=["statefarm", "cifar100"])
    args = parser.parse_args()

    base       = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        print(f"\n[{dataset}]")
        try:
            p = plot_heatmaps(base, dataset, output_dir)
            print(f"  heatmaps      → {p}")
        except Exception as e:
            print(f"  heatmaps: {e}")

        try:
            p = plot_forgetting_curves(base, dataset, output_dir)
            print(f"  forgetting    → {p}")
        except Exception as e:
            print(f"  forgetting: {e}")

    try:
        p = plot_aa_af_tradeoff(base, args.datasets, output_dir)
        print(f"\n  AA-AF tradeoff → {p}")
    except Exception as e:
        print(f"  AA-AF tradeoff: {e}")


if __name__ == "__main__":
    main()

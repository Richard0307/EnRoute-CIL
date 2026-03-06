#!/usr/bin/env python3
"""
Run multiple seeds for a single experiment configuration and aggregate results.

Examples:
    python scripts/run_multiseed.py \
        --seeds 42 43 44 \
        --output_root output/quick_fix_multiseed \
        -- \
        --epochs 5 --use_moe --use_energy_ood --use_ortho_proj \
        --der_alpha 0.3 --save_best

    python scripts/run_multiseed.py \
        --aggregate_only \
        --seeds 42 43 44 \
        --output_root output/quick_fix_multiseed
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_results import generate_all_plots


README_RESULTS_START = "<!-- MULTISEED_RESULTS_START -->"
README_RESULTS_END = "<!-- MULTISEED_RESULTS_END -->"


def _parse_seed_list(raw_items: list[str]) -> list[int]:
    seeds: list[int] = []
    for item in raw_items:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            seeds.append(int(part))
    unique = sorted(set(seeds))
    if not unique:
        raise ValueError("At least one seed must be provided.")
    return unique


def _strip_remainder_separator(items: list[str]) -> list[str]:
    if items and items[0] == "--":
        return items[1:]
    return items


def _validate_passthrough_args(main_args: list[str]) -> None:
    reserved = {
        "--seed",
        "--output_dir",
        "--checkpoint_dir",
        "--resume",
        "--resume_path",
    }
    for arg in main_args:
        if arg in reserved or any(arg.startswith(f"{flag}=") for flag in reserved):
            raise ValueError(
                f"{arg} is managed by run_multiseed.py and should not be passed through."
            )


def _compute_aa_af(acc_matrix: np.ndarray) -> tuple[float, float]:
    total_tasks = acc_matrix.shape[0]
    aa = float(np.mean(acc_matrix[total_tasks - 1, :total_tasks]))
    if total_tasks <= 1:
        return aa, 0.0

    forgetting = [
        max(acc_matrix[s][i] for s in range(i, total_tasks - 1)) - acc_matrix[total_tasks - 1][i]
        for i in range(total_tasks - 1)
    ]
    return aa, float(np.mean(forgetting))


def _load_runtime_metrics(metrics_path: Path) -> dict[str, np.ndarray] | None:
    if not metrics_path.exists():
        return None
    with np.load(metrics_path) as data:
        return {key: data[key] for key in data.files}


def _build_seed_command(
    python_exe: str,
    main_script: Path,
    main_args: list[str],
    output_dir: Path,
    seed: int,
) -> list[str]:
    return [
        python_exe,
        str(main_script),
        *main_args,
        "--seed",
        str(seed),
        "--output_dir",
        str(output_dir),
    ]


def _run_single_seed(
    python_exe: str,
    main_script: Path,
    main_args: list[str],
    output_root: Path,
    seed: int,
    skip_existing: bool,
) -> Path:
    seed_dir = output_root / f"seed_{seed}"
    acc_matrix_path = seed_dir / "acc_matrix.npy"

    if skip_existing and acc_matrix_path.exists():
        print(f"[run_multiseed] Skip seed {seed}: found {acc_matrix_path}")
        return seed_dir

    seed_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_seed_command(python_exe, main_script, main_args, seed_dir, seed)
    print(f"[run_multiseed] Running seed {seed}")
    print(f"[run_multiseed] Command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return seed_dir


def _summarize_seed(seed: int, seed_dir: Path) -> dict[str, Any]:
    acc_matrix_path = seed_dir / "acc_matrix.npy"
    if not acc_matrix_path.exists():
        raise FileNotFoundError(f"Missing acc_matrix: {acc_matrix_path}")

    acc_matrix = np.load(acc_matrix_path)
    aa, af = _compute_aa_af(acc_matrix)

    summary: dict[str, Any] = {
        "seed": seed,
        "seed_dir": str(seed_dir),
        "aa": aa,
        "af": af,
        "acc_matrix": acc_matrix,
    }

    metrics = _load_runtime_metrics(seed_dir / "training_metrics.npz")
    if metrics is not None:
        summary["runtime_metrics"] = metrics

        old_task_acc = metrics.get("task_old_task_accuracy")
        if old_task_acc is not None:
            valid = old_task_acc[np.isfinite(old_task_acc)]
            summary["final_old_task_accuracy"] = float(valid[-1]) if valid.size > 0 else float("nan")

        max_memory = metrics.get("task_max_memory_mb")
        if max_memory is not None and max_memory.size > 0:
            summary["max_memory_mb"] = float(np.max(max_memory))

        epoch_time = metrics.get("task_avg_epoch_time_sec")
        if epoch_time is not None and epoch_time.size > 0:
            summary["mean_epoch_time_sec"] = float(np.mean(epoch_time))

        latency = metrics.get("task_avg_batch_latency_ms")
        if latency is not None and latency.size > 0:
            summary["mean_batch_latency_ms"] = float(np.mean(latency))

        ood_auroc = metrics.get("ood_auroc")
        if ood_auroc is not None and ood_auroc.size > 0:
            summary["final_ood_auroc"] = float(ood_auroc[-1])

        ood_fpr = metrics.get("ood_fpr_at_95tpr")
        if ood_fpr is not None and ood_fpr.size > 0:
            summary["final_ood_fpr_at_95tpr"] = float(ood_fpr[-1])

    return summary


def _aggregate_metric_arrays(metric_dicts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not metric_dicts:
        return {}

    shared_keys = set(metric_dicts[0].keys())
    for metrics in metric_dicts[1:]:
        shared_keys &= set(metrics.keys())

    aggregated: dict[str, np.ndarray] = {}
    index_like = {"task_ids", "epoch_task_ids", "epoch_indices", "ood_task_ids"}

    for key in sorted(shared_keys):
        arrays = [metrics[key] for metrics in metric_dicts]
        first_shape = arrays[0].shape
        if any(arr.shape != first_shape for arr in arrays):
            continue

        stack = np.stack(arrays, axis=0)
        if key in index_like:
            if np.allclose(stack, stack[0], equal_nan=True):
                aggregated[key] = stack[0]
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            aggregated[key] = np.nanmean(stack, axis=0)
            aggregated[f"{key}_std"] = np.nanstd(stack, axis=0)

    return aggregated


def _write_summary_files(
    output_root: Path,
    run_summaries: list[dict[str, Any]],
    aggregate_summary: dict[str, Any],
) -> None:
    csv_path = output_root / "multiseed_results.csv"
    json_path = output_root / "multiseed_summary.json"
    md_path = output_root / "multiseed_summary.md"

    csv_fields = [
        "seed",
        "aa",
        "af",
        "final_old_task_accuracy",
        "max_memory_mb",
        "mean_epoch_time_sec",
        "mean_batch_latency_ms",
        "final_ood_auroc",
        "final_ood_fpr_at_95tpr",
        "seed_dir",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for item in run_summaries:
            row = {field: item.get(field, "") for field in csv_fields}
            writer.writerow(row)

    json_payload = {
        "per_seed": [
            {
                key: value
                for key, value in item.items()
                if key not in {"acc_matrix", "runtime_metrics"}
            }
            for item in run_summaries
        ],
        "aggregate": aggregate_summary,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Multi-seed Summary",
        "",
        f"Seeds: {', '.join(str(item['seed']) for item in run_summaries)}",
        "",
        "## Aggregate",
        "",
        f"- AA mean±std: {aggregate_summary['aa_mean']:.4f} ± {aggregate_summary['aa_std']:.4f}",
        f"- AF mean±std: {aggregate_summary['af_mean']:.4f} ± {aggregate_summary['af_std']:.4f}",
    ]

    optional_keys = [
        ("final_old_task_accuracy_mean", "Final old-task accuracy mean±std"),
        ("max_memory_mb_mean", "Max memory (MB) mean±std"),
        ("mean_epoch_time_sec_mean", "Mean epoch time (s) mean±std"),
        ("mean_batch_latency_ms_mean", "Mean batch latency (ms) mean±std"),
        ("final_ood_auroc_mean", "Final OOD AUROC mean±std"),
        ("final_ood_fpr_at_95tpr_mean", "Final OOD FPR@95TPR mean±std"),
    ]
    for mean_key, label in optional_keys:
        std_key = mean_key.replace("_mean", "_std")
        if mean_key in aggregate_summary and std_key in aggregate_summary:
            lines.append(
                f"- {label}: {aggregate_summary[mean_key]:.4f} ± {aggregate_summary[std_key]:.4f}"
            )

    lines.extend([
        "",
        "## Per-seed",
        "",
        "| seed | AA | AF | output_dir |",
        "|---:|---:|---:|---|",
    ])
    for item in run_summaries:
        lines.append(
            f"| {item['seed']} | {item['aa']:.4f} | {item['af']:.4f} | {item['seed_dir']} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_pct_pm(mean: float, std: float) -> str:
    return f"{mean * 100:.2f}% ± {std * 100:.2f}%"


def _format_float_pm(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _build_readme_results_block(
    output_root: Path,
    run_summaries: list[dict[str, Any]],
    aggregate_summary: dict[str, Any],
) -> str:
    seeds = ", ".join(str(item["seed"]) for item in run_summaries)
    lines = [
        "### Latest Multi-seed Summary",
        "",
        f"Generated automatically from `{output_root}` across seeds `{seeds}`.",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]

    final_old = "-"
    if "final_old_task_accuracy_mean" in aggregate_summary:
        final_old = _format_pct_pm(
            aggregate_summary["final_old_task_accuracy_mean"],
            aggregate_summary["final_old_task_accuracy_std"],
        )

    peak_mem = "-"
    if "max_memory_mb_mean" in aggregate_summary:
        peak_mem = _format_float_pm(
            aggregate_summary["max_memory_mb_mean"],
            aggregate_summary["max_memory_mb_std"],
            digits=1,
        )

    mean_epoch = "-"
    if "mean_epoch_time_sec_mean" in aggregate_summary:
        mean_epoch = _format_float_pm(
            aggregate_summary["mean_epoch_time_sec_mean"],
            aggregate_summary["mean_epoch_time_sec_std"],
            digits=2,
        )

    final_ood = "-"
    if "final_ood_auroc_mean" in aggregate_summary:
        final_ood = _format_float_pm(
            aggregate_summary["final_ood_auroc_mean"],
            aggregate_summary["final_ood_auroc_std"],
            digits=4,
        )

    final_ood_fpr = "-"
    if "final_ood_fpr_at_95tpr_mean" in aggregate_summary:
        final_ood_fpr = _format_pct_pm(
            aggregate_summary["final_ood_fpr_at_95tpr_mean"],
            aggregate_summary["final_ood_fpr_at_95tpr_std"],
        )

    mean_latency = "-"
    if "mean_batch_latency_ms_mean" in aggregate_summary:
        mean_latency = _format_float_pm(
            aggregate_summary["mean_batch_latency_ms_mean"],
            aggregate_summary["mean_batch_latency_ms_std"],
            digits=2,
        )

    lines.extend([
        f"| Experiment Root | `{output_root}` |",
        f"| Seeds | `{seeds}` |",
        f"| Final AA (mean ± std) | `{_format_pct_pm(aggregate_summary['aa_mean'], aggregate_summary['aa_std'])}` |",
        f"| Final AF (mean ± std) | `{_format_pct_pm(aggregate_summary['af_mean'], aggregate_summary['af_std'])}` |",
        f"| Final Old-Task Accuracy (mean ± std) | `{final_old}` |",
        f"| Final OOD AUROC (mean ± std) | `{final_ood}` |",
        f"| Final OOD FPR@95TPR (mean ± std) | `{final_ood_fpr}` |",
        f"| Peak Memory (MB, mean ± std) | `{peak_mem}` |",
        f"| Mean Epoch Time (s, mean ± std) | `{mean_epoch}` |",
        f"| Mean Batch Latency (ms, mean ± std) | `{mean_latency}` |",
    ])

    lines.extend([
        "",
        "| Seed | Final AA | Final AF | Output Directory |",
        "|---:|---:|---:|---|",
    ])
    for item in run_summaries:
        lines.append(
            f"| {item['seed']} | {_format_pct(item['aa'])} | {_format_pct(item['af'])} | "
            f"`{item['seed_dir']}` |"
        )

    lines.append(
        f"| mean ± std | {_format_pct_pm(aggregate_summary['aa_mean'], aggregate_summary['aa_std'])} | "
        f"{_format_pct_pm(aggregate_summary['af_mean'], aggregate_summary['af_std'])} | "
        f"`{output_root}` |"
    )
    lines.append("")
    return "\n".join(lines)


def _update_readme_results(
    readme_path: Path,
    output_root: Path,
    run_summaries: list[dict[str, Any]],
    aggregate_summary: dict[str, Any],
) -> None:
    if not readme_path.exists():
        print(f"[run_multiseed] Warning: README not found, skip update: {readme_path}")
        return

    content = readme_path.read_text(encoding="utf-8")
    generated_block = _build_readme_results_block(output_root, run_summaries, aggregate_summary)
    wrapped = f"{README_RESULTS_START}\n{generated_block}\n{README_RESULTS_END}"

    if README_RESULTS_START in content and README_RESULTS_END in content:
        start = content.index(README_RESULTS_START)
        end = content.index(README_RESULTS_END) + len(README_RESULTS_END)
        updated = content[:start] + wrapped + content[end:]
    else:
        anchor = "## Current Results in Repository Artifacts"
        if anchor not in content:
            print(
                f"[run_multiseed] Warning: '{anchor}' not found in {readme_path}; "
                "skip README update."
            )
            return
        insert_at = content.index(anchor) + len(anchor)
        updated = content[:insert_at] + "\n\n" + wrapped + content[insert_at:]

    readme_path.write_text(updated, encoding="utf-8")


def _aggregate_runs(output_root: Path, seeds: list[int], readme_path: Path | None = None) -> None:
    run_summaries = [_summarize_seed(seed, output_root / f"seed_{seed}") for seed in seeds]

    aa_values = np.array([item["aa"] for item in run_summaries], dtype=float)
    af_values = np.array([item["af"] for item in run_summaries], dtype=float)
    acc_matrices = np.stack([item["acc_matrix"] for item in run_summaries], axis=0)

    mean_acc_matrix = np.mean(acc_matrices, axis=0)
    std_acc_matrix = np.std(acc_matrices, axis=0)
    np.save(output_root / "mean_acc_matrix.npy", mean_acc_matrix)
    np.save(output_root / "std_acc_matrix.npy", std_acc_matrix)

    runtime_metrics = [
        item["runtime_metrics"]
        for item in run_summaries
        if "runtime_metrics" in item
    ]
    aggregated_metrics = _aggregate_metric_arrays(runtime_metrics)
    metrics_path = output_root / "multiseed_metrics.npz"
    if len(runtime_metrics) != len(run_summaries):
        aggregated_metrics = {}
        metrics_path = None
        print(
            "[run_multiseed] Warning: runtime metrics are incomplete across seeds; "
            "aggregate plots will use mean acc_matrix only."
        )
    elif aggregated_metrics:
        np.savez(metrics_path, **aggregated_metrics)
    else:
        metrics_path = None

    aggregate_summary: dict[str, Any] = {
        "num_seeds": len(seeds),
        "aa_mean": float(np.mean(aa_values)),
        "aa_std": float(np.std(aa_values)),
        "af_mean": float(np.mean(af_values)),
        "af_std": float(np.std(af_values)),
        "mean_acc_matrix_path": str(output_root / "mean_acc_matrix.npy"),
        "std_acc_matrix_path": str(output_root / "std_acc_matrix.npy"),
    }

    for key in [
        "final_old_task_accuracy",
        "max_memory_mb",
        "mean_epoch_time_sec",
        "mean_batch_latency_ms",
        "final_ood_auroc",
        "final_ood_fpr_at_95tpr",
    ]:
        values = np.array([item[key] for item in run_summaries if key in item], dtype=float)
        if values.size == len(run_summaries):
            aggregate_summary[f"{key}_mean"] = float(np.nanmean(values))
            aggregate_summary[f"{key}_std"] = float(np.nanstd(values))

    _write_summary_files(output_root, run_summaries, aggregate_summary)

    try:
        plot_paths = generate_all_plots(
            acc_matrix=mean_acc_matrix,
            output_dir=output_root,
            runtime_metrics_path=metrics_path,
            verbose=False,
        )
    except Exception as exc:
        print(f"[run_multiseed] Warning: failed to generate aggregate plots: {exc}")
    else:
        aggregate_summary["plot_paths"] = {name: str(path) for name, path in plot_paths.items()}

    summary_path = output_root / "multiseed_summary.json"
    if summary_path.exists():
        current = json.loads(summary_path.read_text(encoding="utf-8"))
        current["aggregate"] = aggregate_summary
        summary_path.write_text(
            json.dumps(current, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if readme_path is not None:
        _update_readme_results(readme_path, output_root, run_summaries, aggregate_summary)

    print("[run_multiseed] Aggregate summary")
    print(
        f"[run_multiseed] AA mean±std = {aggregate_summary['aa_mean']:.4f} ± "
        f"{aggregate_summary['aa_std']:.4f}"
    )
    print(
        f"[run_multiseed] AF mean±std = {aggregate_summary['af_mean']:.4f} ± "
        f"{aggregate_summary['af_std']:.4f}"
    )
    print(f"[run_multiseed] Wrote: {output_root / 'multiseed_results.csv'}")
    print(f"[run_multiseed] Wrote: {output_root / 'multiseed_summary.md'}")
    print(f"[run_multiseed] Wrote: {output_root / 'multiseed_summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and aggregate multi-seed CIL experiments")
    parser.add_argument(
        "--seeds",
        nargs="+",
        required=True,
        help="Seed list, e.g. --seeds 42 43 44 or --seeds 42,43,44",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory for all seed runs and aggregate outputs",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch main.py",
    )
    parser.add_argument(
        "--main_script",
        type=str,
        default=str(REPO_ROOT / "main.py"),
        help="Path to the main training script",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a seed if <output_root>/seed_<seed>/acc_matrix.npy already exists",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Do not launch training; only aggregate existing seed outputs",
    )
    parser.add_argument(
        "--readme_path",
        type=str,
        default=str(REPO_ROOT / "README.md"),
        help="README path to update after aggregation",
    )
    parser.add_argument(
        "--no_update_readme",
        action="store_true",
        help="Do not update README after aggregation",
    )
    parser.add_argument(
        "main_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to main.py. Put them after --",
    )
    args = parser.parse_args()
    args.seeds = _parse_seed_list(args.seeds)
    args.main_args = _strip_remainder_separator(args.main_args)
    return args


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    main_script = Path(args.main_script)
    if not main_script.is_absolute():
        main_script = (REPO_ROOT / main_script).resolve()

    readme_path = None
    if not args.no_update_readme:
        readme_path = Path(args.readme_path)
        if not readme_path.is_absolute():
            readme_path = (REPO_ROOT / readme_path).resolve()

    if not args.aggregate_only:
        if not args.main_args:
            raise SystemExit(
                "No main.py arguments provided. Pass them after --, "
                "or use --aggregate_only."
            )
        try:
            _validate_passthrough_args(args.main_args)
        except ValueError as exc:
            raise SystemExit(f"[Error] {exc}") from exc

        for seed in args.seeds:
            _run_single_seed(
                python_exe=args.python,
                main_script=main_script,
                main_args=args.main_args,
                output_root=output_root,
                seed=seed,
                skip_existing=args.skip_existing,
            )

    _aggregate_runs(output_root=output_root, seeds=args.seeds, readme_path=readme_path)


if __name__ == "__main__":
    main()

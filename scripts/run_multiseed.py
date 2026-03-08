#!/usr/bin/env python3
"""Run legacy multi-seed experiments or unified benchmark sweeps."""

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
BENCHMARK_METHODS = ("ours", "l2p", "coda_prompt", "moe_adapters")
BENCHMARK_DATASETS = ("cifar100", "statefarm")


def _parse_seed_list(raw_items: list[str]) -> list[int]:
    seeds: list[int] = []
    for item in raw_items:
        for part in item.split(","):
            part = part.strip()
            if part:
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


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def _run_subprocess(cmd: list[str], cwd: Path) -> None:
    print(f"[run_multiseed] Command: {' '.join(str(part) for part in cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


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
    _run_subprocess(cmd, REPO_ROOT)
    return seed_dir


def _build_benchmark_seed_command(args: argparse.Namespace, method: str, dataset: str, seed: int, seed_dir: Path) -> list[str]:
    benchmark_script = Path(args.benchmark_script)
    if not benchmark_script.is_absolute():
        benchmark_script = (REPO_ROOT / benchmark_script).resolve()
    cmd = [
        args.python,
        str(benchmark_script),
        "--method", method,
        "--dataset", dataset,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
        "--output_dir", str(seed_dir),
    ]
    if args.fast_mode:
        cmd.append("--fast_mode")
    return cmd


def _run_single_benchmark_seed(
    args: argparse.Namespace,
    method: str,
    dataset: str,
    seed: int,
    group_root: Path,
) -> Path:
    seed_dir = group_root / f"seed_{seed}"
    summary_path = seed_dir / "benchmark_summary.json"
    if args.skip_existing and summary_path.exists():
        print(f"[run_multiseed] Skip {dataset}/{method}/seed_{seed}: found {summary_path}")
        return seed_dir

    seed_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_benchmark_seed_command(args, method, dataset, seed, seed_dir)
    print(f"[run_multiseed] Running {dataset}/{method}/seed_{seed}")
    _run_subprocess(cmd, REPO_ROOT)
    return seed_dir


def _resolve_acc_matrix_path(seed_dir: Path, summary: dict[str, Any] | None) -> Path:
    if summary is not None and summary.get("acc_matrix_path"):
        path = Path(summary["acc_matrix_path"])
        if not path.is_absolute():
            path = (seed_dir / path).resolve()
        if path.exists():
            return path
    return seed_dir / "acc_matrix.npy"


def _summarize_seed(seed: int, seed_dir: Path) -> dict[str, Any]:
    benchmark_summary = _load_json(seed_dir / "benchmark_summary.json")
    acc_matrix_path = _resolve_acc_matrix_path(seed_dir, benchmark_summary)
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

    if benchmark_summary is not None:
        for key in [
            "method",
            "dataset",
            "final_old_task_accuracy",
            "final_ood_auroc",
            "final_ood_fpr_at_95tpr",
            "total_params",
            "trainable_params",
            "trainable_ratio",
            "codaprompt_final_avg_acc",
            "moe_final_avg_acc",
        ]:
            if key in benchmark_summary:
                summary[key] = benchmark_summary[key]

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
        stats = _load_json(seed_dir / "model_stats.json") or {}
        if stats:
            summary["total_params"] = stats.get("total_params")
            summary["trainable_params"] = stats.get("trainable_params")
            summary["trainable_ratio"] = stats.get("trainable_ratio")
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
        "method",
        "dataset",
        "aa",
        "af",
        "final_old_task_accuracy",
        "trainable_ratio",
        "trainable_params",
        "total_params",
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
        ("trainable_ratio_mean", "Trainable ratio mean±std"),
        ("trainable_params_mean", "Trainable params mean±std"),
        ("total_params_mean", "Total params mean±std"),
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
            f"| {item['seed']} | {_format_pct(item['aa'])} | {_format_pct(item['af'])} | `{item['seed_dir']}` |"
        )
    lines.append(
        f"| mean ± std | {_format_pct_pm(aggregate_summary['aa_mean'], aggregate_summary['aa_std'])} | "
        f"{_format_pct_pm(aggregate_summary['af_mean'], aggregate_summary['af_std'])} | `{output_root}` |"
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
                f"[run_multiseed] Warning: '{anchor}' not found in {readme_path}; skip README update."
            )
            return
        insert_at = content.index(anchor) + len(anchor)
        updated = content[:insert_at] + "\n\n" + wrapped + content[insert_at:]

    readme_path.write_text(updated, encoding="utf-8")


def _aggregate_runs(output_root: Path, seeds: list[int], readme_path: Path | None = None) -> dict[str, Any]:
    run_summaries = [_summarize_seed(seed, output_root / f"seed_{seed}") for seed in seeds]

    aa_values = np.array([item["aa"] for item in run_summaries], dtype=float)
    af_values = np.array([item["af"] for item in run_summaries], dtype=float)
    acc_matrices = np.stack([item["acc_matrix"] for item in run_summaries], axis=0)

    mean_acc_matrix = np.mean(acc_matrices, axis=0)
    std_acc_matrix = np.std(acc_matrices, axis=0)
    np.save(output_root / "mean_acc_matrix.npy", mean_acc_matrix)
    np.save(output_root / "std_acc_matrix.npy", std_acc_matrix)

    runtime_metrics = [item["runtime_metrics"] for item in run_summaries if "runtime_metrics" in item]
    aggregated_metrics = _aggregate_metric_arrays(runtime_metrics)
    metrics_path = output_root / "multiseed_metrics.npz"
    if runtime_metrics and len(runtime_metrics) == len(run_summaries) and aggregated_metrics:
        np.savez(metrics_path, **aggregated_metrics)
    else:
        metrics_path = None
        aggregated_metrics = {}
        if runtime_metrics and len(runtime_metrics) != len(run_summaries):
            print(
                "[run_multiseed] Warning: runtime metrics are incomplete across seeds; aggregate plots will use mean acc_matrix only."
            )

    aggregate_summary: dict[str, Any] = {
        "num_seeds": len(seeds),
        "aa_mean": float(np.mean(aa_values)),
        "aa_std": float(np.std(aa_values)),
        "af_mean": float(np.mean(af_values)),
        "af_std": float(np.std(af_values)),
        "mean_acc_matrix_path": str(output_root / "mean_acc_matrix.npy"),
        "std_acc_matrix_path": str(output_root / "std_acc_matrix.npy"),
    }

    scalar_keys = [
        "final_old_task_accuracy",
        "trainable_ratio",
        "trainable_params",
        "total_params",
        "max_memory_mb",
        "mean_epoch_time_sec",
        "mean_batch_latency_ms",
        "final_ood_auroc",
        "final_ood_fpr_at_95tpr",
    ]
    for key in scalar_keys:
        values = [item[key] for item in run_summaries if item.get(key) is not None]
        if len(values) == len(run_summaries):
            array = np.array(values, dtype=float)
            aggregate_summary[f"{key}_mean"] = float(np.nanmean(array))
            aggregate_summary[f"{key}_std"] = float(np.nanstd(array))

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
        summary_path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")

    if readme_path is not None:
        _update_readme_results(readme_path, output_root, run_summaries, aggregate_summary)

    print("[run_multiseed] Aggregate summary")
    print(f"[run_multiseed] AA mean±std = {aggregate_summary['aa_mean']:.4f} ± {aggregate_summary['aa_std']:.4f}")
    print(f"[run_multiseed] AF mean±std = {aggregate_summary['af_mean']:.4f} ± {aggregate_summary['af_std']:.4f}")
    print(f"[run_multiseed] Wrote: {output_root / 'multiseed_results.csv'}")
    print(f"[run_multiseed] Wrote: {output_root / 'multiseed_summary.md'}")
    print(f"[run_multiseed] Wrote: {output_root / 'multiseed_summary.json'}")
    return aggregate_summary


def _benchmark_group_root(output_root: Path, dataset: str, method: str) -> Path:
    return output_root / dataset / method


def _write_benchmark_overview(output_root: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = output_root / "benchmark_overview.csv"
    json_path = output_root / "benchmark_overview.json"
    md_path = output_root / "benchmark_overview.md"

    fields = [
        "dataset",
        "method",
        "num_seeds",
        "aa_mean",
        "aa_std",
        "af_mean",
        "af_std",
        "final_old_task_accuracy_mean",
        "trainable_ratio_mean",
        "trainable_params_mean",
        "total_params_mean",
        "final_ood_auroc_mean",
        "final_ood_fpr_at_95tpr_mean",
        "output_root",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Benchmark Overview",
        "",
        "| Dataset | Method | AA (mean ± std) | AF (mean ± std) | Trainable Ratio | OOD AUROC | Robustness (AA std / AF std) | Output |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        aa = f"{row['aa_mean'] * 100:.2f}% ± {row['aa_std'] * 100:.2f}%"
        af = f"{row['af_mean'] * 100:.2f}% ± {row['af_std'] * 100:.2f}%"
        trainable_ratio = "-"
        if row.get("trainable_ratio_mean") is not None:
            trainable_ratio = f"{row['trainable_ratio_mean'] * 100:.2f}%"
        ood = "-"
        if row.get("final_ood_auroc_mean") is not None:
            ood = f"{row['final_ood_auroc_mean']:.4f}"
        robustness = f"{row['aa_std'] * 100:.2f}% / {row['af_std'] * 100:.2f}%"
        lines.append(
            f"| {row['dataset']} | {row['method']} | {aa} | {af} | {trainable_ratio} | {ood} | {robustness} | `{row['output_root']}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_benchmark(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    overview_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        for method in args.methods:
            group_root = _benchmark_group_root(output_root, dataset, method)
            group_root.mkdir(parents=True, exist_ok=True)
            if not args.aggregate_only:
                for seed in args.seeds:
                    _run_single_benchmark_seed(args, method, dataset, seed, group_root)
            aggregate_summary = _aggregate_runs(group_root, args.seeds, readme_path=None)
            overview_rows.append({
                "dataset": dataset,
                "method": method,
                "output_root": str(group_root),
                **aggregate_summary,
            })

    _write_benchmark_overview(output_root, overview_rows)
    print(f"[run_multiseed] Wrote benchmark overview: {output_root / 'benchmark_overview.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and aggregate multi-seed CIL experiments")
    parser.add_argument("--seeds", nargs="+", required=True, help="Seed list, e.g. --seeds 42 43 44 or --seeds 42,43,44")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory for all seed runs and aggregate outputs")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch training scripts")
    parser.add_argument("--skip_existing", action="store_true", help="Skip a seed if its summary already exists")
    parser.add_argument("--aggregate_only", action="store_true", help="Do not launch training; only aggregate existing outputs")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode across methods and datasets")
    parser.add_argument("--methods", nargs="+", default=list(BENCHMARK_METHODS), choices=BENCHMARK_METHODS, help="Benchmark methods")
    parser.add_argument("--datasets", nargs="+", default=list(BENCHMARK_DATASETS), choices=BENCHMARK_DATASETS, help="Benchmark datasets")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task in benchmark mode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size in benchmark mode")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device passed to benchmark runners")
    parser.add_argument("--fast_mode", action="store_true", help="Apply safe speed-oriented settings in benchmark runners")
    parser.add_argument("--benchmark_script", type=str, default=str(REPO_ROOT / "scripts" / "run_benchmark_method.py"), help="Benchmark runner script")
    parser.add_argument("--main_script", type=str, default=str(REPO_ROOT / "main.py"), help="Path to the main training script for legacy mode")
    parser.add_argument("--readme_path", type=str, default=str(REPO_ROOT / "README.md"), help="README path to update after legacy aggregation")
    parser.add_argument("--no_update_readme", action="store_true", help="Do not update README after legacy aggregation")
    parser.add_argument("main_args", nargs=argparse.REMAINDER, help="Arguments forwarded to main.py. Put them after --")
    args = parser.parse_args()
    args.seeds = _parse_seed_list(args.seeds)
    args.main_args = _strip_remainder_separator(args.main_args)
    return args


def main() -> None:
    args = parse_args()
    if args.benchmark:
        if args.main_args:
            raise SystemExit("Benchmark mode does not accept passthrough main.py arguments.")
        _run_benchmark(args)
        return

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
            raise SystemExit("No main.py arguments provided. Pass them after --, or use --aggregate_only.")
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

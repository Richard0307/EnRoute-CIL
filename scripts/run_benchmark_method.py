#!/usr/bin/env python3
"""Run one method/seed/dataset benchmark job and normalize outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common import (
    compute_aa_af_from_acc_matrix,
    ensure_dir,
    ensure_statefarm_processed_split,
    write_benchmark_summary,
    write_class_order_files,
)


METHODS = {"ours", "l2p", "coda_prompt", "moe_adapters"}
DATASETS = {"cifar100", "statefarm"}


def _schedule_for_dataset(dataset: str, init_override: int | None = None,
                          inc_override: int | None = None) -> tuple[int, int]:
    if init_override is not None and inc_override is not None:
        return init_override, inc_override
    if dataset == "cifar100":
        return 50, 10
    if dataset == "statefarm":
        return 5, 1
    raise ValueError(f"Unsupported dataset: {dataset}")


def _num_classes_for_dataset(dataset: str) -> int:
    if dataset == "cifar100":
        return 100
    if dataset == "statefarm":
        return 10
    raise ValueError(f"Unsupported dataset: {dataset}")


def _prepare_assets(dataset: str, seed: int, output_dir: Path) -> dict[str, Path]:
    assets_dir = ensure_dir(output_dir / "assets")
    num_classes = _num_classes_for_dataset(dataset)
    class_order_json, class_order_yaml, _ = write_class_order_files(
        root=assets_dir,
        dataset=dataset,
        seed=seed,
        num_classes=num_classes,
    )

    statefarm_processed_root = None
    if dataset == "statefarm":
        raw_root = REPO_ROOT / "data" / "raw" / "statefarm"
        processed_root = REPO_ROOT / "data" / "processed" / "statefarm_cl"
        statefarm_processed_root = ensure_statefarm_processed_split(
            raw_root=raw_root,
            processed_root=processed_root,
        )

    return {
        "class_order_json": class_order_json,
        "class_order_yaml": class_order_yaml,
        "statefarm_processed_root": statefarm_processed_root,
    }


def _load_training_metrics(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_subprocess(cmd: list[str], cwd: Path) -> None:
    print(f"[benchmark] Command: {' '.join(str(part) for part in cmd)}")
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def _collect_ours_summary(output_dir: Path, method: str, dataset: str, seed: int) -> dict[str, Any]:
    acc_matrix = np.load(output_dir / "acc_matrix.npy")
    aa, af = compute_aa_af_from_acc_matrix(acc_matrix)
    metrics = _load_training_metrics(output_dir / "training_metrics.npz") or {}
    stats_path = output_dir / "model_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    old_task_acc = None
    if "task_old_task_accuracy" in metrics and metrics["task_old_task_accuracy"].size > 0:
        valid = metrics["task_old_task_accuracy"][np.isfinite(metrics["task_old_task_accuracy"])]
        if valid.size > 0:
            old_task_acc = float(valid[-1])

    auroc = None
    fpr = None
    if "ood_auroc" in metrics and metrics["ood_auroc"].size > 0:
        auroc = float(metrics["ood_auroc"][-1])
    if "ood_fpr_at_95tpr" in metrics and metrics["ood_fpr_at_95tpr"].size > 0:
        fpr = float(metrics["ood_fpr_at_95tpr"][-1])

    return {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "aa": aa,
        "af": af,
        "final_old_task_accuracy": old_task_acc,
        "final_ood_auroc": auroc,
        "final_ood_fpr_at_95tpr": fpr,
        "total_params": stats.get("total_params"),
        "trainable_params": stats.get("trainable_params"),
        "trainable_ratio": stats.get("trainable_ratio"),
        "acc_matrix_path": str(output_dir / "acc_matrix.npy"),
    }


def _run_ours(args: argparse.Namespace, assets: dict[str, Path]) -> dict[str, Any]:
    init_classes, inc_classes = _schedule_for_dataset(
        args.dataset,
        args.init_classes_override,
        args.inc_classes_override,
    )
    if args.dataset == "cifar100":
        data_root = REPO_ROOT / "data" / "raw"
    else:
        data_root = REPO_ROOT / "data" / "processed"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--dataset", args.dataset,
        "--data_root", str(data_root),
        "--class_order_path", str(assets["class_order_json"]),
        "--init_classes", str(init_classes),
        "--inc_classes", str(inc_classes),
        "--max_tasks", str(args.max_tasks),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--output_dir", str(args.output_dir),
        "--num_workers", str(args.num_workers),
        "--no_auto_plots",
    ]
    if args.fast_mode:
        cmd.extend(["--no_auto_plots"])
    _run_subprocess(cmd, REPO_ROOT)
    return _collect_ours_summary(args.output_dir, args.method, args.dataset, args.seed)


def _write_coda_config(path: Path, dataset: str, epochs: int, batch_size: int, workers: int,
                       init_classes: int, inc_classes: int, max_tasks: int) -> None:
    if dataset == "cifar100":
        dataset_name = "CIFAR100"
        dataroot = str(REPO_ROOT / "data" / "raw")
    else:
        dataset_name = "StateFarm"
        dataroot = str(REPO_ROOT / "data" / "processed")

    payload = {
        "dataset": dataset_name,
        "first_split_size": init_classes,
        "other_split_size": inc_classes,
        "schedule": [epochs],
        "schedule_type": "cosine",
        "batch_size": batch_size,
        "optimizer": "Adam",
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0,
        "model_type": "zoo",
        "model_name": "vit_pt_imnet",
        "max_task": max_tasks,
        "dataroot": dataroot,
        "workers": workers,
        "validation": False,
        "train_aug": True,
        "rand_split": False,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _collect_coda_family_summary(output_dir: Path, method: str, dataset: str, seed: int) -> dict[str, Any]:
    global_path = output_dir / "results-acc" / "global.yaml"
    pt_path = output_dir / "results-acc" / "pt.yaml"
    if not global_path.exists() or not pt_path.exists():
        raise FileNotFoundError("Missing CODA/L2P result files.")

    global_payload = yaml.safe_load(global_path.read_text(encoding="utf-8"))
    pt_payload = yaml.safe_load(pt_path.read_text(encoding="utf-8"))
    acc_matrix = np.asarray(pt_payload["mean"], dtype=float).T / 100.0
    np.save(output_dir / "acc_matrix.npy", acc_matrix)

    aa, af = compute_aa_af_from_acc_matrix(acc_matrix)
    old_task_acc = float(np.mean(acc_matrix[-1, :-1])) if acc_matrix.shape[0] > 1 else None

    stats = _load_json(output_dir / "model_stats.json") or {}
    ood_payload = _load_json(output_dir / "ood_metrics.json") or {}
    final_ood = ood_payload.get("final") or {}

    return {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "aa": aa,
        "af": af,
        "final_old_task_accuracy": old_task_acc,
        "final_ood_auroc": final_ood.get("auroc"),
        "final_ood_fpr_at_95tpr": final_ood.get("fpr_at_95tpr"),
        "total_params": stats.get("total_params"),
        "trainable_params": stats.get("trainable_params"),
        "trainable_ratio": stats.get("trainable_ratio"),
        "codaprompt_final_avg_acc": global_payload["mean"][-1] / 100.0,
        "acc_matrix_path": str(output_dir / "acc_matrix.npy"),
    }


def _run_coda_family(args: argparse.Namespace, assets: dict[str, Path], learner_name: str) -> dict[str, Any]:
    repo_dir = REPO_ROOT / "third_party" / "CODA-Prompt"
    config_path = args.output_dir / "benchmark_config.yaml"
    init_classes, inc_classes = _schedule_for_dataset(
        args.dataset,
        args.init_classes_override,
        args.inc_classes_override,
    )
    _write_coda_config(
        config_path,
        args.dataset,
        args.epochs,
        args.batch_size,
        args.num_workers,
        init_classes,
        inc_classes,
        args.max_tasks,
    )
    prompt_param = {
        "CODAPrompt": ["100", "8", "0.0"],
        "L2P": ["30", "20", "-1"],
    }[learner_name]
    gpu = "0" if args.device.startswith("cuda") else "-1"
    cmd = [
        sys.executable,
        "run.py",
        "--config", str(config_path),
        "--gpuid", gpu,
        "--repeat", "1",
        "--overwrite", "1",
        "--seed", str(args.seed),
        "--class_order_path", str(assets["class_order_json"]),
        "--learner_type", "prompt",
        "--learner_name", learner_name,
        "--prompt_param", *prompt_param,
        "--log_dir", str(args.output_dir),
    ]
    _run_subprocess(cmd, repo_dir)
    return _collect_coda_family_summary(args.output_dir, args.method, args.dataset, args.seed)


def _write_moe_config(path: Path, dataset: str, epochs: int, batch_size: int,
                      output_dir: Path, seed: int, workers: int) -> None:
    if dataset == "cifar100":
        initial_increment = 50
        increment = 10
        dataset_root = str(REPO_ROOT / "data" / "raw")
    else:
        initial_increment = 5
        increment = 1
        dataset_root = str(REPO_ROOT / "data" / "processed")

    yaml_text = f"""hydra:
  run:
    dir: {output_dir}
  job:
    chdir: true

  job_logging:
    version: 1
    formatters:
      simple:
        format: '%(message)s'

class_order: ""
dataset_root: "{dataset_root}"
workdir: ""
log_path: "metrics.json"
model_name: "ViT-B/16"
prompt_template: "a bad photo of a {{}}."
batch_size: {batch_size}
workers: {workers}
increment: {increment}
initial_increment: {initial_increment}
scenario: "class"
dataset: "{dataset}"
seed: {seed}
weight_decay: 0.0
l2: 0
ce_method: 0
method: "MoE-Adapters"
lr: 1e-3
ls: 0.0
epochs: {epochs}
"""
    path.write_text(yaml_text, encoding="utf-8")


def _collect_moe_summary(output_dir: Path, method: str, dataset: str, seed: int) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("Missing MoE-Adapters metrics.json")

    task_rows: list[dict[str, Any]] = []
    summary_row: dict[str, Any] | None = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "task" in row:
                task_rows.append(row)
            else:
                summary_row = row

    total_tasks = len(task_rows)
    acc_matrix = np.zeros((total_tasks, total_tasks), dtype=float)
    for row in task_rows:
        task_id = int(row["task"])
        for eval_id, value in enumerate(row["acc_per_task"]):
            acc_matrix[task_id, eval_id] = float(value) / 100.0
    np.save(output_dir / "acc_matrix.npy", acc_matrix)

    aa, af = compute_aa_af_from_acc_matrix(acc_matrix)
    old_task_acc = float(np.mean(acc_matrix[-1, :-1])) if total_tasks > 1 else None
    stats = _load_json(output_dir / "model_stats.json") or {}
    ood_payload = _load_json(output_dir / "ood_metrics.json") or {}
    final_ood = ood_payload.get("final") or {}

    return {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "aa": aa,
        "af": af,
        "final_old_task_accuracy": old_task_acc,
        "final_ood_auroc": final_ood.get("auroc"),
        "final_ood_fpr_at_95tpr": final_ood.get("fpr_at_95tpr"),
        "total_params": stats.get("total_params"),
        "trainable_params": stats.get("trainable_params"),
        "trainable_ratio": stats.get("trainable_ratio"),
        "moe_final_avg_acc": (summary_row or {}).get("avg", None),
        "acc_matrix_path": str(output_dir / "acc_matrix.npy"),
    }


def _run_moe(args: argparse.Namespace, assets: dict[str, Path]) -> dict[str, Any]:
    repo_dir = REPO_ROOT / "third_party" / "MoE-Adapters4CL" / "cil"
    config_path = args.output_dir / "benchmark_config.yaml"
    _write_moe_config(
        config_path,
        args.dataset,
        args.epochs,
        args.batch_size,
        args.output_dir,
        args.seed,
        args.num_workers,
    )
    cmd = [
        sys.executable,
        "main.py",
        "--config-path", str(config_path.parent),
        "--config-name", config_path.stem,
        f"class_order={assets['class_order_yaml']}",
        f"seed={args.seed}",
    ]
    _run_subprocess(cmd, repo_dir)
    return _collect_moe_summary(args.output_dir, args.method, args.dataset, args.seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one benchmark method/seed job")
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_tasks", type=int, default=-1)
    parser.add_argument("--init_classes_override", type=int, default=None)
    parser.add_argument("--inc_classes_override", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fast_mode", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fast_mode:
        args.num_workers = max(args.num_workers, 8)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = _prepare_assets(args.dataset, args.seed, args.output_dir)

    if args.method == "ours":
        summary = _run_ours(args, assets)
    elif args.method == "l2p":
        summary = _run_coda_family(args, assets, learner_name="L2P")
    elif args.method == "coda_prompt":
        summary = _run_coda_family(args, assets, learner_name="CODAPrompt")
    elif args.method == "moe_adapters":
        summary = _run_moe(args, assets)
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    write_benchmark_summary(args.output_dir, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

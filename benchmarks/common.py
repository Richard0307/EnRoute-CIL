"""Shared utilities for benchmark orchestration."""

from __future__ import annotations

import csv
import json
import os
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np


STATEFARM_CLASS_NAMES = [
    "safe_driving",
    "texting_right",
    "talking_phone_right",
    "texting_left",
    "talking_phone_left",
    "operating_radio",
    "drinking",
    "reaching_behind",
    "hair_makeup",
    "talking_to_passenger",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_class_order_files(
    root: Path,
    dataset: str,
    seed: int,
    num_classes: int,
) -> tuple[Path, Path, list[int]]:
    ensure_dir(root)
    rng = np.random.RandomState(seed)
    class_order = rng.permutation(num_classes).tolist()

    json_path = root / f"{dataset}_seed{seed}.json"
    yaml_path = root / f"{dataset}_seed{seed}.yaml"
    json_path.write_text(
        json.dumps({"dataset": dataset, "seed": seed, "class_order": class_order}, indent=2),
        encoding="utf-8",
    )
    yaml_path.write_text(
        "class_order: [" + ", ".join(str(item) for item in class_order) + "]\n",
        encoding="utf-8",
    )
    return json_path, yaml_path, class_order


def _iter_statefarm_raw_samples(raw_root: Path) -> tuple[list[tuple[Path, str, str]], bool]:
    csv_candidates = [
        raw_root / "driver_imgs_list.csv",
        raw_root / "state-farm-distracted-driver-detection" / "driver_imgs_list.csv",
        raw_root / "imgs" / "driver_imgs_list.csv",
    ]
    train_candidates = [
        raw_root / "imgs" / "train",
        raw_root / "state-farm-distracted-driver-detection" / "imgs" / "train",
        raw_root / "train",
    ]

    train_dir = next((path for path in train_candidates if path.exists()), None)
    if train_dir is None:
        raise FileNotFoundError(
            f"Could not find State Farm train directory under {raw_root}"
        )

    csv_path = next((path for path in csv_candidates if path.exists()), None)
    samples: list[tuple[Path, str, str]] = []
    if csv_path is not None:
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                class_name = row["classname"]
                img_name = row["img"]
                subject = row["subject"]
                img_path = train_dir / class_name / img_name
                if img_path.exists():
                    samples.append((img_path, class_name, subject))
        return samples, True

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file():
                samples.append((img_path, class_dir.name, "unknown"))
    return samples, False


def _ensure_statefarm_raw_source(raw_root: Path) -> Path:
    csv_candidates = [
        raw_root / "driver_imgs_list.csv",
        raw_root / "state-farm-distracted-driver-detection" / "driver_imgs_list.csv",
        raw_root / "imgs" / "driver_imgs_list.csv",
    ]
    if any(path.exists() for path in csv_candidates):
        return raw_root

    zip_candidates = [
        raw_root / "state-farm-distracted-driver-detection.zip",
        raw_root / "statefarm.zip",
    ]
    zip_path = next((path for path in zip_candidates if path.exists()), None)
    if zip_path is None:
        return raw_root

    extract_root = raw_root / "_extracted"
    marker = extract_root / ".unzipped"
    if marker.exists():
        return extract_root

    _clear_directory(extract_root)
    ensure_dir(extract_root)
    with zipfile.ZipFile(zip_path, "r") as handle:
        handle.extractall(extract_root)
    marker.write_text(json.dumps({"zip_path": str(zip_path)}, indent=2), encoding="utf-8")
    return extract_root


def _clear_directory(root: Path) -> None:
    if not root.exists():
        return
    for item in root.rglob("*"):
        if item.is_symlink() or item.is_file():
            item.unlink()
    for item in sorted(root.rglob("*"), reverse=True):
        if item.is_dir():
            item.rmdir()


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        dst.write_bytes(src.read_bytes())


def ensure_statefarm_processed_split(
    raw_root: Path,
    processed_root: Path,
    split_seed: int = 2026,
    test_ratio: float = 0.2,
) -> Path:
    marker = processed_root / ".prepared"
    if marker.exists():
        return processed_root

    source_root = _ensure_statefarm_raw_source(raw_root)
    samples, has_driver = _iter_statefarm_raw_samples(source_root)
    if not samples:
        raise RuntimeError("No State Farm samples found for benchmark preparation.")

    _clear_directory(processed_root)
    train_root = ensure_dir(processed_root / "train")
    test_root = ensure_dir(processed_root / "test")

    if has_driver:
        subjects = sorted({subject for _, _, subject in samples})
        rng = np.random.RandomState(split_seed)
        rng.shuffle(subjects)
        split_idx = max(1, int(round(len(subjects) * (1.0 - test_ratio))))
        train_subjects = set(subjects[:split_idx])
        test_subjects = set(subjects[split_idx:]) or set(subjects[-1:])

        for img_path, class_name, subject in samples:
            split_root = train_root if subject in train_subjects else test_root
            dst_dir = ensure_dir(split_root / class_name)
            _link_or_copy(img_path, dst_dir / img_path.name)
    else:
        by_class: dict[str, list[Path]] = {}
        for img_path, class_name, _ in samples:
            by_class.setdefault(class_name, []).append(img_path)
        rng = np.random.RandomState(split_seed)
        for class_name, class_samples in sorted(by_class.items()):
            class_samples = list(class_samples)
            rng.shuffle(class_samples)
            split_idx = max(1, int(round(len(class_samples) * (1.0 - test_ratio))))
            train_samples = class_samples[:split_idx]
            test_samples = class_samples[split_idx:] or class_samples[-1:]
            for src in train_samples:
                dst_dir = ensure_dir(train_root / class_name)
                _link_or_copy(src, dst_dir / src.name)
            for src in test_samples:
                dst_dir = ensure_dir(test_root / class_name)
                _link_or_copy(src, dst_dir / src.name)

    marker.write_text(json.dumps({
        "split_seed": split_seed,
        "test_ratio": test_ratio,
        "num_samples": len(samples),
        "driver_based": has_driver,
        "source_root": str(source_root),
    }, indent=2), encoding="utf-8")
    return processed_root


def compute_aa_af_from_acc_matrix(acc_matrix: np.ndarray) -> tuple[float, float]:
    total_tasks = acc_matrix.shape[0]
    aa = float(np.mean(acc_matrix[total_tasks - 1, :total_tasks]))
    if total_tasks <= 1:
        return aa, 0.0
    forgetting = [
        max(acc_matrix[s][i] for s in range(i, total_tasks - 1)) - acc_matrix[total_tasks - 1][i]
        for i in range(total_tasks - 1)
    ]
    return aa, float(np.mean(forgetting))


def write_benchmark_summary(output_dir: Path, payload: dict) -> Path:
    path = output_dir / "benchmark_summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def mean_or_none(values: Iterable[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))

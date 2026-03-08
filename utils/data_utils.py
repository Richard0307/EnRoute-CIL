"""Dataset builders and task loaders for incremental learning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100, ImageFolder


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ClassSubset(Dataset):
    """Subset view restricted to a class list."""

    def __init__(self, dataset: Dataset, class_ids: List[int],
                 transform=None, label_map: Dict[int, int] | None = None):
        self.dataset = dataset
        self.class_ids = set(class_ids)
        self.transform = transform
        self.label_map = label_map
        targets = np.array(dataset.targets)
        self.indices = np.where(np.isin(targets, list(class_ids)))[0].tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)
        if self.label_map is not None:
            label = self.label_map[label]
        return img, label


class MixedDataset(Dataset):
    """Combine current-task data with replay exemplars."""

    def __init__(self, new_data: ClassSubset,
                 exemplar_data: List[Tuple],
                 exemplar_transform=None,
                 online_exemplar_augmentation: bool = True,
                 oversample_exemplars: bool = False):
        self.new_data = new_data
        self.exemplar_transform = exemplar_transform

        if online_exemplar_augmentation:
            processed_exemplars = exemplar_data
        else:
            processed_exemplars = []
            for img, label in exemplar_data:
                transformed = exemplar_transform(img) if exemplar_transform is not None else img
                processed_exemplars.append((transformed, label))
            self.exemplar_transform = None

        if oversample_exemplars and len(processed_exemplars) > 0:
            repeats = max(1, len(new_data) // len(processed_exemplars))
            self.exemplar_data = processed_exemplars * repeats
        else:
            self.exemplar_data = processed_exemplars

    def __len__(self) -> int:
        return len(self.new_data) + len(self.exemplar_data)

    def __getitem__(self, idx: int):
        if idx < len(self.new_data):
            return self.new_data[idx]
        img, label = self.exemplar_data[idx - len(self.new_data)]
        if self.exemplar_transform is not None:
            img = self.exemplar_transform(img)
        return img, label


def _load_class_order(num_classes: int, seed: int,
                      class_order_path: str = "") -> List[int]:
    if class_order_path:
        payload = json.loads(Path(class_order_path).read_text(encoding="utf-8"))
        class_order = payload["class_order"] if isinstance(payload, dict) else payload
        if len(class_order) != num_classes:
            raise ValueError(
                f"class order length mismatch: expected {num_classes}, got {len(class_order)}"
            )
        return [int(item) for item in class_order]

    rng = np.random.RandomState(seed)
    return rng.permutation(num_classes).tolist()


def _split_task_classes(class_order: List[int],
                        init_classes: int,
                        inc_classes: int) -> List[List[int]]:
    task_classes = [class_order[:init_classes]]
    remaining = class_order[init_classes:]
    for start in range(0, len(remaining), inc_classes):
        task_classes.append(remaining[start:start + inc_classes])
    return task_classes


def build_cifar100_tasks(
    data_root: str,
    init_classes: int = 50,
    inc_classes: int = 10,
    seed: int = 42,
    class_order_path: str = "",
) -> Tuple[List[List[int]], CIFAR100, CIFAR100]:
    train_dataset = CIFAR100(root=data_root, train=True, download=True, transform=None)
    test_dataset = CIFAR100(root=data_root, train=False, download=True, transform=None)
    class_order = _load_class_order(100, seed, class_order_path)
    task_classes = _split_task_classes(class_order, init_classes, inc_classes)
    return task_classes, train_dataset, test_dataset


def build_statefarm_tasks(
    data_root: str,
    init_classes: int = 5,
    inc_classes: int = 1,
    seed: int = 42,
    class_order_path: str = "",
) -> Tuple[List[List[int]], ImageFolder, ImageFolder]:
    processed_root = Path(data_root) / "statefarm_cl"
    train_dir = processed_root / "train"
    test_dir = processed_root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Missing processed State Farm benchmark split under "
            f"{processed_root}. Prepare it before training."
        )

    train_dataset = ImageFolder(str(train_dir), transform=None)
    test_dataset = ImageFolder(str(test_dir), transform=None)
    num_classes = len(train_dataset.classes)
    class_order = _load_class_order(num_classes, seed, class_order_path)
    task_classes = _split_task_classes(class_order, init_classes, inc_classes)
    return task_classes, train_dataset, test_dataset


def build_incremental_tasks(
    dataset: str,
    data_root: str,
    init_classes: int,
    inc_classes: int,
    seed: int,
    class_order_path: str = "",
):
    dataset_key = dataset.lower()
    if dataset_key == "cifar100":
        return build_cifar100_tasks(
            data_root=data_root,
            init_classes=init_classes,
            inc_classes=inc_classes,
            seed=seed,
            class_order_path=class_order_path,
        )
    if dataset_key == "statefarm":
        return build_statefarm_tasks(
            data_root=data_root,
            init_classes=init_classes,
            inc_classes=inc_classes,
            seed=seed,
            class_order_path=class_order_path,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_task_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    class_ids: List[int],
    batch_size: int = 64,
    num_workers: int = 4,
    exemplar_data: List[Tuple] | None = None,
    label_map: Dict[int, int] | None = None,
    online_exemplar_augmentation: bool = True,
    oversample_exemplars: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    train_subset = ClassSubset(train_dataset, class_ids,
                               transform=TRAIN_TRANSFORM, label_map=label_map)
    test_subset = ClassSubset(test_dataset, class_ids,
                              transform=TEST_TRANSFORM, label_map=label_map)

    if exemplar_data:
        train_subset = MixedDataset(
            train_subset,
            exemplar_data,
            exemplar_transform=TRAIN_TRANSFORM,
            online_exemplar_augmentation=online_exemplar_augmentation,
            oversample_exemplars=oversample_exemplars,
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader

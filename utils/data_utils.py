"""
CIFAR-100 incremental task data loader.

Splits 100 classes into: base phase (init_classes) + subsequent phases (inc_classes each).
All images are resized to 224×224 with ImageNet normalization (Rule 5 from methodology).
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100


# ── Transforms (aligned with ViT pre-training stats) ──────────────────────

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


# ── Subset that filters by class label ─────────────────────────────────────

class ClassSubset(Dataset):
    """A subset of a dataset that only contains samples from specified classes."""

    def __init__(self, dataset: CIFAR100, class_ids: List[int],
                 transform=None, label_map: Dict[int, int] = None):
        self.dataset = dataset
        self.class_ids = set(class_ids)
        self.transform = transform
        self.label_map = label_map  # original_class_id → contiguous_index
        # Find indices belonging to the target classes
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


# ── Build incremental tasks ────────────────────────────────────────────────

def build_cifar100_tasks(
    data_root: str,
    init_classes: int = 50,
    inc_classes: int = 10,
    seed: int = 42,
) -> Tuple[List[List[int]], CIFAR100, CIFAR100]:
    """
    Build the incremental class schedule and return raw datasets.

    Returns:
        task_classes: List of lists, each containing class IDs for that task.
        train_dataset: Raw CIFAR-100 train set (no transform applied here).
        test_dataset:  Raw CIFAR-100 test set (no transform applied here).
    """
    # Download CIFAR-100 via torchvision (uses data_root/cifar100/)
    train_dataset = CIFAR100(root=data_root, train=True, download=True,
                             transform=None)
    test_dataset = CIFAR100(root=data_root, train=False, download=True,
                            transform=None)

    # Deterministic class order
    rng = np.random.RandomState(seed)
    class_order = rng.permutation(100).tolist()

    # Split into tasks
    task_classes = []
    # Base phase
    task_classes.append(class_order[:init_classes])
    # Subsequent phases
    remaining = class_order[init_classes:]
    for start in range(0, len(remaining), inc_classes):
        task_classes.append(remaining[start:start + inc_classes])

    return task_classes, train_dataset, test_dataset


def get_task_dataloaders(
    train_dataset: CIFAR100,
    test_dataset: CIFAR100,
    class_ids: List[int],
    batch_size: int = 64,
    num_workers: int = 4,
    exemplar_data: List[Tuple] = None,
    label_map: Dict[int, int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/test DataLoaders for a specific incremental task.

    If exemplar_data is provided, it is mixed into the training set.
    """
    train_subset = ClassSubset(train_dataset, class_ids,
                               transform=TRAIN_TRANSFORM, label_map=label_map)
    test_subset = ClassSubset(test_dataset, class_ids,
                              transform=TEST_TRANSFORM, label_map=label_map)

    # Mix exemplars into training data
    if exemplar_data:
        train_subset = MixedDataset(train_subset, exemplar_data)

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)
    return train_loader, test_loader


class MixedDataset(Dataset):
    """Combines a ClassSubset with exemplar replay data."""

    def __init__(self, new_data: ClassSubset,
                 exemplar_data: List[Tuple]):
        self.new_data = new_data
        self.exemplar_data = exemplar_data

    def __len__(self) -> int:
        return len(self.new_data) + len(self.exemplar_data)

    def __getitem__(self, idx: int):
        if idx < len(self.new_data):
            return self.new_data[idx]
        else:
            return self.exemplar_data[idx - len(self.new_data)]

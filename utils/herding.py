"""
Module D (partial): Herding exemplar selection.

Selects K exemplars per class that are nearest to the class feature mean.
Reference: iCaRL (Rebuffi et al., 2017)
"""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_class_features(
    model,
    dataset,
    class_id: int,
    device: str = "cuda",
    batch_size: int = 64,
    label_map: dict = None,
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Extract features for all samples of a given class.

    Returns:
        features: (N, d) numpy array of feature vectors.
        samples:  List of (image_tensor, remapped_label) tuples for replay.
    """
    model.eval()
    # Collect indices for this class
    targets = np.array(dataset.dataset.targets)
    class_indices = np.where(targets == class_id)[0]

    features_list = []
    samples = []

    for idx in class_indices:
        img, label = dataset.dataset[idx]
        if dataset.transform is not None:
            img_t = dataset.transform(img)
        else:
            img_t = img
        # Remap label to contiguous index
        if label_map is not None:
            label = label_map[label]
        samples.append((img_t, label))

    # Batch feature extraction
    all_imgs = torch.stack([s[0] for s in samples]).to(device)
    for i in range(0, len(all_imgs), batch_size):
        batch = all_imgs[i:i + batch_size]
        feats = model.extract_features(batch)
        features_list.append(feats.cpu().numpy())

    features = np.concatenate(features_list, axis=0)
    return features, samples


def herding_select(
    features: np.ndarray,
    samples: List[Tuple],
    k: int = 20,
) -> List[Tuple]:
    """
    Herding selection: greedily pick K exemplars closest to the running mean.

    Args:
        features: (N, d) feature matrix for one class.
        samples:  Corresponding (image_tensor, label) list.
        k:        Number of exemplars to retain.

    Returns:
        selected: List of K (image_tensor, label) tuples.
    """
    k = min(k, len(features))
    class_mean = features.mean(axis=0)  # (d,)
    class_mean = class_mean / (np.linalg.norm(class_mean) + 1e-8)

    selected_indices = []
    selected_sum = np.zeros_like(class_mean)

    for _ in range(k):
        # Target: running mean should approach class mean
        candidate_means = (selected_sum[None, :] + features) / (
            len(selected_indices) + 1
        )
        # Normalize candidates
        norms = np.linalg.norm(candidate_means, axis=1, keepdims=True) + 1e-8
        candidate_means = candidate_means / norms
        # Pick the one closest to class mean
        distances = np.linalg.norm(candidate_means - class_mean[None, :], axis=1)
        # Exclude already selected
        distances[selected_indices] = np.inf
        best = np.argmin(distances)
        selected_indices.append(best)
        selected_sum += features[best]

    return [samples[i] for i in selected_indices]

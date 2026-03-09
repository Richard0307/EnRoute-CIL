#!/usr/bin/env python3
"""Create a 2D t-SNE plot from saved feature and label arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def _load_array(path: Path, key: str | None) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            if key is None:
                if len(data.files) != 1:
                    raise ValueError(f"{path} contains multiple arrays; pass --key")
                key = data.files[0]
            return data[key]
    raise ValueError(f"Unsupported array format: {path}")


def _sample(features: np.ndarray, labels: np.ndarray | None, max_samples: int, seed: int):
    if len(features) <= max_samples:
        return features, labels
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(features), size=max_samples, replace=False)
    indices.sort()
    sampled_labels = None if labels is None else labels[indices]
    return features[indices], sampled_labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--feature_key")
    parser.add_argument("--label_key")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="Feature t-SNE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()

    features = _load_array(args.features, args.feature_key)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {features.shape}")

    labels = None
    if args.labels is not None:
        labels = _load_array(args.labels, args.label_key).reshape(-1)
        if len(labels) != len(features):
            raise ValueError("Feature and label counts do not match")

    features, labels = _sample(features, labels, args.max_samples, args.seed)
    if len(features) < 3:
        raise ValueError("Need at least 3 samples for t-SNE")

    perplexity = min(args.perplexity, max(2.0, (len(features) - 1) / 3))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(features)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=160)
    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=8, alpha=0.8)
    else:
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap("tab20", len(unique_labels))
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=10,
                alpha=0.8,
                color=cmap(idx),
                label=str(label),
            )
        if len(unique_labels) <= 20:
            plt.legend(frameon=False, fontsize=8, markerscale=1.5)
    plt.title(args.title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE plot to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

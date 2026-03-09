#!/usr/bin/env python3
"""Plot a heatmap from a 2D tensor saved as .npy or .npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--key")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="Heatmap")
    parser.add_argument("--xlabel", default="Column")
    parser.add_argument("--ylabel", default="Row")
    parser.add_argument("--cmap", default="viridis")
    args = parser.parse_args()

    matrix = np.asarray(_load_array(args.input, args.key)).squeeze()
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {matrix.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=160)
    image = plt.imshow(matrix, aspect="auto", interpolation="nearest", cmap=args.cmap)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.title(args.title)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

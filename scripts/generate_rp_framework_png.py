#!/usr/bin/env python3
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Mermaid to PNG via Kroki.")
    parser.add_argument(
        "--input",
        default="docs/rp_framework.mmd",
        help="Path to Mermaid input file.",
    )
    parser.add_argument(
        "--output",
        default="docs/rp_framework_kroki.png",
        help="Path to output PNG file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    in_file = (root / args.input).resolve()
    out_file = (root / args.output).resolve()

    if not in_file.exists():
        raise SystemExit(f"Missing input: {in_file}")

    text = in_file.read_text(encoding="utf-8")
    resp = requests.post(
        "https://kroki.io/mermaid/png",
        data=text.encode("utf-8"),
        timeout=60,
    )
    resp.raise_for_status()

    # Flatten transparent background to white for paper-ready export.
    image = Image.open(BytesIO(resp.content)).convert("RGBA")
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    merged = Image.alpha_composite(bg, image).convert("RGB")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_file, format="PNG")
    if out_file.stat().st_size == 0:
        raise SystemExit("Generated file is empty")

    print(f"Generated: {out_file}")


if __name__ == "__main__":
    main()

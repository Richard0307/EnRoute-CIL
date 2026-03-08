#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IN_FILE="$ROOT_DIR/docs/rp_framework.mmd"
OUT_FILE="$ROOT_DIR/docs/rp_framework_kroki.png"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required but not found." >&2
  exit 1
fi

curl -sS -X POST \
  -H 'Content-Type: text/plain' \
  --data-binary "@$IN_FILE" \
  https://kroki.io/mermaid/png > "$OUT_FILE"

if [ ! -s "$OUT_FILE" ]; then
  echo "Failed to generate diagram: output file is empty." >&2
  exit 1
fi

echo "Generated: $OUT_FILE"

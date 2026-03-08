#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-cockpit}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/benchmark_sota}"
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"

PYTHON_CMD=(conda run -n "$CONDA_ENV" python)

print_usage() {
  cat <<'USAGE'
Usage: ./training.sh

Staged benchmark order:
  1. cifar100: ours + l2p + coda_prompt
  2. verify benchmark_summary.json for all stage-1 runs
  3. cifar100: moe_adapters
  4. verify benchmark_summary.json for moe_adapters
  5. statefarm: ours + l2p + coda_prompt + moe_adapters
  6. verify benchmark_summary.json for all statefarm runs

Environment overrides:
  CONDA_ENV    default: cockpit
  OUTPUT_ROOT  default: output/benchmark_sota
  SEEDS        default: "42 43 44"
  EPOCHS       default: 5
  BATCH_SIZE   default: 64
  NUM_WORKERS  default: 8
  DEVICE       default: cuda
USAGE
}

run_stage() {
  local dataset="$1"
  shift
  local methods=("$@")

  echo "[training.sh] Running dataset=${dataset} methods=${methods[*]}"
  "${PYTHON_CMD[@]}" scripts/run_multiseed.py \
    --benchmark \
    --seeds ${SEEDS} \
    --methods "${methods[@]}" \
    --datasets "$dataset" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --fast_mode \
    --skip_existing \
    --output_root "$OUTPUT_ROOT"
}

verify_stage() {
  local dataset="$1"
  shift
  local methods=("$@")
  local methods_joined="${methods[*]}"

  echo "[training.sh] Verifying benchmark_summary.json for dataset=${dataset} methods=${methods[*]}"
  OUTPUT_ROOT_ENV="$OUTPUT_ROOT" \
  DATASET_ENV="$dataset" \
  METHODS_ENV="$methods_joined" \
  SEEDS_ENV="$SEEDS" \
  "${PYTHON_CMD[@]}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT_ENV"])
dataset = os.environ["DATASET_ENV"]
methods = os.environ["METHODS_ENV"].split()
seeds = [int(item) for item in os.environ["SEEDS_ENV"].split()]
required = {
    "method",
    "dataset",
    "seed",
    "aa",
    "af",
    "final_old_task_accuracy",
    "final_ood_auroc",
    "final_ood_fpr_at_95tpr",
    "total_params",
    "trainable_params",
    "trainable_ratio",
    "acc_matrix_path",
}

for method in methods:
    for seed in seeds:
        summary_path = root / dataset / method / f"seed_{seed}" / "benchmark_summary.json"
        if not summary_path.exists():
            raise SystemExit(f"Missing summary: {summary_path}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        missing = sorted(required - set(payload))
        if missing:
            raise SystemExit(f"Incomplete summary: {summary_path} missing {missing}")
        acc_matrix_path = Path(payload["acc_matrix_path"])
        if not acc_matrix_path.is_absolute():
            acc_matrix_path = (summary_path.parent / acc_matrix_path).resolve()
        if not acc_matrix_path.exists():
            raise SystemExit(f"Missing acc_matrix from summary: {acc_matrix_path}")
print("verification_ok")
PY
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage
    return 0
  fi

  local stage1_methods=(ours l2p coda_prompt)
  local stage2_methods=(moe_adapters)
  local statefarm_methods=(ours l2p coda_prompt moe_adapters)

  run_stage cifar100 "${stage1_methods[@]}"
  verify_stage cifar100 "${stage1_methods[@]}"

  run_stage cifar100 "${stage2_methods[@]}"
  verify_stage cifar100 "${stage2_methods[@]}"

  run_stage statefarm "${statefarm_methods[@]}"
  verify_stage statefarm "${statefarm_methods[@]}"

  echo "[training.sh] All stages completed successfully."
}

main "$@"

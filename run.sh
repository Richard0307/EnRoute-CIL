#!/usr/bin/env bash
bash -lc '
set -e

# 1) Baseline: Adapter + KD
python scripts/run_multiseed.py \
  --seeds 42 43 44 \
  --output_root output/paper_baseline_e5 \
  --no_update_readme \
  -- \
  --epochs 5 \
  --batch_size 32 \
  --der_alpha 0 \
  --no_wa \
  --no_online_exemplar_aug \
  --no_oversample \
  --no_scale_matched_head_init \
  --save_best

# 2) + MoE + Energy OOD + OP
python scripts/run_multiseed.py \
  --seeds 42 43 44 \
  --output_root output/paper_moe_ood_op_e5 \
  --no_update_readme \
  -- \
  --epochs 5 \
  --batch_size 32 \
  --use_moe \
  --use_energy_ood \
  --use_ortho_proj \
  --der_alpha 0 \
  --no_wa \
  --no_online_exemplar_aug \
  --no_oversample \
  --no_scale_matched_head_init \
  --save_best

# 3) + DER++ + WA
python scripts/run_multiseed.py \
  --seeds 42 43 44 \
  --output_root output/paper_der_wa_e5 \
  --no_update_readme \
  -- \
  --epochs 5 \
  --batch_size 32 \
  --use_moe \
  --use_energy_ood \
  --use_ortho_proj \
  --der_alpha 0.3 \
  --no_online_exemplar_aug \
  --no_oversample \
  --no_scale_matched_head_init \
  --save_best

# 4) + Online Aug + Oversample + Head Init
python scripts/run_multiseed.py \
  --seeds 42 43 44 \
  --output_root output/paper_full_e5 \
  --no_update_readme \
  -- \
  --epochs 5 \
  --batch_size 32 \
  --use_moe \
  --use_energy_ood \
  --use_ortho_proj \
  --der_alpha 0.3 \
  --save_best
'

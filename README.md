# ERO-MoE-CIL: Privacy-Preserving Open-World Continual Learning for Intelligent Cockpit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/1167839322.svg)](https://doi.org/10.5281/zenodo.18873097)

Official implementation of **ERO-MoE-CIL**, a parameter-efficient class-incremental learning framework for privacy-preserving intelligent cockpit personalization. On Split CIFAR-100 (`50 + 10 x 5`), the full pipeline achieves **86.84% AA** and **7.87% AF** (3-seed mean), reducing forgetting by **44.7%** relative to the 3-seed frozen-ViT `Adapter + KD` baseline.

## Abstract

In-cabin AI systems must continually adapt to new user-specific behaviors after deployment while preserving prior knowledge and protecting sensitive personal data. We propose ERO-MoE-CIL, a continual learning framework that combines a frozen Vision Transformer backbone with parameter-efficient Mixture-of-Experts adapters, exemplar replay with online augmentation, knowledge distillation, orthogonal gradient projection, and energy-based open-world detection. The system is designed for on-device learning under memory and compute constraints without cloud upload of raw cockpit data. Under a fair 4-group x 3-seed comparison on Split CIFAR-100, the full ERO-MoE-CIL pipeline improves Average Accuracy from 85.10% to 86.84% and reduces Average Forgetting from 14.23% to 7.87% relative to the 3-seed `Adapter + KD` baseline, while showing that replay-pipeline repair is the dominant factor behind the observed gain in this frozen-ViT replay-based setting.

## Method Overview
The framework consists of the following components:

| Module | Purpose | Implementation |
|---|---|---|
| A. Frozen ViT backbone | Preserve generic visual features and keep the trainable ratio low | `models/vit_adapter.py` |
| B. Energy-based OOD detector | Flag unseen behavior patterns in the open-world setting | `utils/energy_ood.py` |
| C. Energy-routed MoE adapters | Increase adaptation capacity with low overhead and dynamic expert growth | `models/moe_adapter.py` |
| D. Replay + KD | Mitigate forgetting with exemplar replay and old-model distillation | `trainer.py`, `utils/herding.py` |
| E. DER++ + WA + orthogonal projection | Stabilize incremental training and reduce old/new bias | `trainer.py`, `utils/orthogonal_projection.py` |
| F. Experiment management | Resume/save-best checkpoints, TensorBoard, auto plots, and multi-seed aggregation | `trainer.py`, `scripts/plot_results.py`, `scripts/run_multiseed.py` |

Current training objective:

`L_total = L_CE + lambda_kd * L_KD + alpha_der * L_DER + L_balance(MoE, optional)`

Implementation details that are already reflected in the current code:
- Exemplars are stored as raw PIL images and transformed online, so replay samples receive fresh augmentation every epoch.
- Exemplar replay can be oversampled to reduce the severe imbalance between new-task data and old-task memory.
- New classifier rows are initialized to match the scale of old classifier weights, reducing early-task bias after class expansion.
- Weight Aligning (WA) is applied temporarily at evaluation time after each task when enabled.
- Orthogonal projection uses a block-projection strategy for dynamically expanded parameters such as MoE router weights.

## Framework Figure
![Framework](docs/rp_framework_paper.png)

## Repository Structure
```text
Personalized-Cockpit-CIL/
├── main.py                      # CLI entry point
├── trainer.py                   # Incremental training loop
├── config.py                    # Experiment configuration
├── models/
│   ├── vit_adapter.py           # Frozen ViT + adapters + incremental head
│   └── moe_adapter.py           # MoE adapter layer and router
├── utils/
│   ├── data_utils.py            # CIFAR-100 loading and task split
│   ├── herding.py               # Exemplar selection
│   ├── energy_ood.py            # OOD scoring and metrics
│   ├── orthogonal_projection.py # Gradient projection
│   └── metrics.py               # AA/AF metrics
├── scripts/
│   ├── plot_results.py          # Figure generation for single run / aggregate run
│   └── run_multiseed.py         # Multi-seed experiment runner and README updater
├── docs/rp_framework_paper.png  # Main framework figure used in README

```

## Experimental Protocol

### Dataset and split
- Benchmark: CIFAR-100.
- Incremental schedule: `50 + 10 x 5` classes (`6` tasks total).
- Class order: deterministic random permutation controlled by `--seed`.
- Input pipeline: resize to `224 x 224`, ImageNet normalization.

### Evaluation metrics
- `AA` (Average Accuracy): mean final accuracy over all seen tasks.
- `AF` (Average Forgetting): mean performance drop on old tasks.
- OOD metrics (when enabled): `AUROC`, `FPR@95TPR`.
- Runtime metrics: max allocated memory, old-task accuracy, time/epoch, latency/batch.

## Environment and Installation
Tested locally on:
- Python `3.10.19`
- PyTorch `2.5.1` (CUDA `12.4`)
- timm `1.0.25`
- NumPy `2.0.1`
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU (16 GB)

Recommended setup:

```bash
conda create -n YOUR ENV NAME python=3.10 -y
conda activate YOUR ENV NAME
pip install -r requirements.txt
pip install tensorboard
```

Notes:
- `tensorboard` is required by `trainer.py` for live logging.
- The first run may download pre-trained ViT weights through `timm`, so a working network setup is required unless weights are already cached locally.

## Training Commands

### Quick start (single seed)
```bash
python main.py --epochs 5 --use_moe --use_energy_ood --use_ortho_proj \
  --der_alpha 0.3 --output_dir output/quick --save_best
```

### Multi-seed (recommended for reporting)
```bash
python scripts/run_multiseed.py \
  --seeds 42 43 44 \
  --output_root output/multiseed \
  -- \
  --epochs 5 --use_moe --use_energy_ood --use_ortho_proj \
  --der_alpha 0.3 --save_best
```

### Baseline (frozen ViT + Adapter + KD only)
```bash
python main.py --epochs 5 --output_dir output/baseline --save_best \
  --der_alpha 0 --no_wa \
  --no_online_exemplar_aug \
  --no_oversample \
  --no_scale_matched_head_init
```

### Ablation examples
```bash
# + MoE only
python main.py --epochs 5 --use_moe \
  --der_alpha 0 --no_wa \
  --no_online_exemplar_aug \
  --no_oversample \
  --no_scale_matched_head_init \
  --output_dir output/ablation_moe --save_best

# + MoE + OOD + OP (no replay fixes)
python main.py --epochs 5 --use_moe --use_energy_ood --use_ortho_proj \
  --der_alpha 0 --no_wa \
  --no_online_exemplar_aug \
  --no_oversample \
  --no_scale_matched_head_init \
  --output_dir output/ablation_moe_ood_op --save_best

# + DER++ + WA (still without replay-pipeline fixes)
python main.py --epochs 5 --use_moe --use_energy_ood --use_ortho_proj \
  --der_alpha 0.3 \
  --no_online_exemplar_aug \
  --no_oversample \
  --no_scale_matched_head_init \
  --output_dir output/ablation_der_wa_only --save_best

# Full method
python main.py --epochs 5 --use_moe --use_energy_ood --use_ortho_proj \
  --der_alpha 0.3 \
  --output_dir output/ablation_full --save_best
```

Available flags:
- `--no_wa`: disable weight aligning
- `--no_online_exemplar_aug`: replay exemplars are transformed once and then replayed as fixed tensors
- `--no_oversample`: disable exemplar oversampling
- `--no_scale_matched_head_init`: use default classifier-row initialization on class expansion
- `--der_alpha 0`: disable DER++

## Checkpointing and Resume
Automatic task-level checkpointing is enabled by default.

- Saved files: `<output_dir>/checkpoints/task_XXX.pt`, `latest.pt`, and optional `best.pt`.
- Continue interrupted training:

```bash
# Resume from latest checkpoint
python main.py --resume --output_dir output/full --save_best \
  --use_moe --use_energy_ood --use_ortho_proj

# Resume from an explicit checkpoint path
python main.py --resume --resume_path output/full/checkpoints/task_003.pt \
  --output_dir output/full --save_best \
  --use_moe --use_energy_ood --use_ortho_proj
```

## Monitoring
Each training run writes TensorBoard events to `<output_dir>/tensorboard`.

```bash
tensorboard --logdir output/YOUR_DIR/tensorboard --host 127.0.0.1 --port 6006
```

Per-batch scalars are logged every 10 batches for live monitoring, and per-epoch / per-task scalars are also recorded.

## Output Artifacts
After each run, the pipeline generates:
- `acc_matrix.npy`
- `training_metrics.npz` (runtime + OOD logs, if available)
- `tensorboard/` (live scalars for batch, epoch, task, and OOD metrics)
- `checkpoints/` with `task_XXX.pt`, `latest.pt`, and optional `best.pt`
- `task_accuracy_curves.png`
- `aa_af_progression.png`
- `accuracy_heatmap.png`
- `max_memory_allocated.png` (if runtime metrics available)
- `accuracy_on_old_tasks.png` (if runtime metrics available)
- `time_per_epoch_latency.png` (if runtime metrics available)
- `ood_metrics.png` (if OOD metrics available)

Manual plotting:

```bash
python scripts/plot_results.py \
  --matrix output/full/acc_matrix.npy \
  --output_dir output/full \
  --metrics output/full/training_metrics.npz
```

## Main Results

All experiments are evaluated on Split CIFAR-100 (`50 + 10 x 5`, 6 tasks). The main comparison below reports mean +/- std over 3 independent runs with seeds `{42, 43, 44}` for all four configurations.

### Fair Comparison (4 Groups x 3 Seeds)

| Configuration | AA (%) | AF (%) | Final Old-Task Acc (%) | Final OOD AUROC | Delta AA vs Baseline | Delta AF vs Baseline |
|:---|---:|---:|---:|---:|---:|---:|
| Baseline (`Adapter + KD`) | 85.10 ± 1.38 | 14.23 ± 1.73 | 82.54 ± 1.76 | - | 0.00 | 0.00 |
| + MoE + Energy OOD + OP | 83.17 ± 1.67 | 16.06 ± 1.83 | 80.39 ± 1.90 | 0.7844 ± 0.0122 | -1.93 | +1.83 |
| + DER++ + WA | 81.28 ± 1.79 | 18.72 ± 1.86 | 78.07 ± 2.15 | 0.7246 ± 0.0386 | -3.82 | +4.49 |
| **+ Online Aug + Oversample + Head Init (Full)** | **86.84 ± 0.89** | **7.87 ± 0.51** | **86.07 ± 0.79** | **0.7950 ± 0.0131** | **+1.74** | **-6.36** |

The fair 3-seed comparison supports a narrower but stronger claim than the previous draft: in this replay-based frozen-ViT setting, adding `MoE + Energy OOD + OP` and later `DER++ + WA` does not improve performance while the replay pipeline remains unrepaired, and both intermediate configurations underperform the `Adapter + KD` baseline. Once online exemplar augmentation, balanced oversampling, and scale-matched classifier expansion are enabled, the same framework becomes the strongest configuration, improving AA by **+1.74** points and reducing AF by **-6.36** points (**44.7% relative**) over the baseline.

### Per-Seed Breakdown

<!-- MULTISEED_RESULTS_START -->

| Seed | AA (%) | AF (%) |
|---:|---:|---:|
| 42 | 88.05 | 7.58 |
| 43 | 86.55 | 7.43 |
| 44 | 85.93 | 8.58 |
| **mean ± std** | **86.84 ± 0.89** | **7.87 ± 0.51** |

<!-- MULTISEED_RESULTS_END -->

### Stage-wise Effect of Component Groups

The stage-wise deltas below show what each group of components contributed when all rows are compared under the same 3-seed protocol:

| Transition | Delta AA | Delta AF | Delta Final Old-Task Acc |
|:---|---:|---:|---:|
| Baseline -> + MoE + Energy OOD + OP | -1.93 | +1.83 | -2.15 |
| + MoE + Energy OOD + OP -> + DER++ + WA | -1.88 | +2.66 | -2.32 |
| + DER++ + WA -> + Online Aug + Oversample + Head Init (Full) | **+5.56** | **-10.85** | **+8.00** |

Key observations:
- `MoE + Energy OOD + OP` alone does not provide a positive gain under the unrepaired replay pipeline.
- `DER++ + WA` on top of that setting further reduces AA and worsens AF.
- The decisive performance jump appears only after enabling online exemplar augmentation, balanced oversampling, and scale-matched classifier expansion.
- This README claim is intentionally bounded to the current replay-based frozen-ViT setting; it is not stated as a universal conclusion for all PEFT-CIL methods.

### Accuracy Matrix (Seed 42, Full Method)

Each cell `(i, j)` shows accuracy on task `j` after training on task `i`.

|  | T0 | T1 | T2 | T3 | T4 | T5 |
|---:|---:|---:|---:|---:|---:|---:|
| After T0 | 93.02 | — | — | — | — | — |
| After T1 | 90.32 | 97.10 | — | — | — | — |
| After T2 | 86.72 | 95.10 | 93.90 | — | — | — |
| After T3 | 82.66 | 92.30 | 95.20 | 96.30 | — | — |
| After T4 | 79.14 | 89.10 | 92.40 | 96.10 | 92.20 | — |
| After T5 | 75.72 | 87.70 | 88.30 | 92.30 | 91.90 | 92.40 |

## Reproducibility Notes
- All results can be reproduced using the commands in [Training Commands](#training-commands).
- `--seed` controls class order and random initialization. CUDNN deterministic mode is enabled in `main.py`.
- For fair comparisons, keep `--epochs`, `--batch_size`, and seed set fixed across ablations.

## Privacy and Ethics
- This codebase is designed for privacy-preserving continual learning where raw user behavior data remains local.
- No cloud data transmission is required by default.
- Deployment on real cockpit data should follow applicable data governance, consent, and safety regulations.

## Limitations
- Current benchmark is Split CIFAR-100; real in-cabin datasets are not yet integrated.
- OOD evaluation uses unseen future-task classes as a proxy for open-world unknowns.
- Experiments are conducted on a single GPU; distributed training is not yet supported.


<!-- ## Citation

If you find this repository useful, please cite:

```bibtex
@software{ero_moe_cil_2026,
  author    = {Richard},
  title     = {{ERO-MoE-CIL}: Privacy-Preserving Open-World Continual Learning for Intelligent Cockpit},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18873097},
  url       = {https://doi.org/10.5281/zenodo.18873097}
}
``` -->

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

# ERO-MoE-CIL: Privacy-Preserving Open-World Continual Learning for Intelligent Cockpit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/1167839322.svg)](https://doi.org/10.5281/zenodo.18873097)

Official implementation of **ERO-MoE-CIL**, a parameter-efficient class-incremental learning framework for privacy-preserving intelligent cockpit personalization. On Split CIFAR-100 (`50 + 10 x 5`), ERO-MoE-CIL achieves **87.01% AA** and **8.04% AF** (3-seed mean), reducing forgetting by **46%** relative to the frozen-ViT baseline.

## Abstract

In-cabin AI systems must continually adapt to new user-specific behaviors after deployment while preserving prior knowledge and protecting sensitive personal data. We propose ERO-MoE-CIL, a continual learning framework that combines a frozen Vision Transformer backbone with parameter-efficient Mixture-of-Experts adapters, exemplar replay with online augmentation, knowledge distillation, orthogonal gradient projection, and energy-based open-world detection. The system is designed for on-device learning under memory and compute constraints without cloud upload of raw cockpit data. On the Split CIFAR-100 benchmark, ERO-MoE-CIL reduces Average Forgetting from 14.88% to 8.04% while improving Average Accuracy from 85.03% to 87.01%, with stable performance across multiple random seeds.

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
  --der_alpha 0 --no_wa --no_oversample
```

### Ablation examples
```bash
# + MoE only
python main.py --epochs 5 --use_moe \
  --der_alpha 0 --no_wa --no_oversample \
  --output_dir output/ablation_moe --save_best

# + MoE + OOD + OP (no replay fixes)
python main.py --epochs 5 --use_moe --use_energy_ood --use_ortho_proj \
  --der_alpha 0 --no_wa --no_oversample \
  --output_dir output/ablation_moe_ood_op --save_best

# Full method without DER++
python main.py --epochs 5 --use_moe --use_energy_ood --use_ortho_proj \
  --der_alpha 0 --output_dir output/ablation_no_der --save_best
```

Available flags: `--no_wa` (disable weight aligning), `--no_oversample` (disable exemplar oversampling), `--der_alpha 0` (disable DER++).

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

All experiments are evaluated on Split CIFAR-100 (`50 + 10 x 5`, 6 tasks). Multi-seed results report mean ± std over 3 independent runs with seeds `{42, 43, 44}`. Single-seed baselines use seed `42`.

### Comparison with Baseline

| Method | Epochs | AA (%) | AF (%) | Seeds |
|:---|---:|---:|---:|---:|
| Frozen ViT + Adapter + KD (baseline) | 5 | 85.03 | 14.88 | 1 |
| + MoE + Energy OOD + OP | 5 | 84.89 | 14.54 | 1 |
| **ERO-MoE-CIL (full)** | **5** | **87.01 ± 0.75** | **8.04 ± 0.69** | **3** |

The full method improves AA by **+2.0%** and reduces AF by **-6.8%** over the baseline, indicating that the replay-level fixes (online exemplar augmentation, oversampling, classifier initialization) contribute more to forgetting reduction than the loss-level additions (DER++, WA) alone.

### Per-Seed Breakdown

<!-- MULTISEED_RESULTS_START -->

| Seed | AA (%) | AF (%) |
|---:|---:|---:|
| 42 | 88.06 | 7.58 |
| 43 | 86.53 | 7.54 |
| 44 | 86.43 | 9.02 |
| **mean ± std** | **87.01 ± 0.75** | **8.04 ± 0.69** |

<!-- MULTISEED_RESULTS_END -->

### Ablation Study

To isolate the contribution of each component, the following ablations are reported on seed `42` with 5 epochs per task:

| Configuration | AA (%) | AF (%) | Delta AA | Delta AF |
|:---|---:|---:|---:|---:|
| Baseline (Adapter + KD) | 85.03 | 14.88 | — | — |
| + MoE + OOD + OP | 84.89 | 14.54 | -0.14 | -0.34 |
| + DER++ + WA | 84.84 | 15.00 | -0.19 | +0.12 |
| + Online augmentation + oversampling + head init | 88.17 | 8.17 | **+3.14** | **-6.71** |

Key observations:
- Adding MoE, OOD, and orthogonal projection alone does not significantly reduce forgetting.
- DER++ and WA without replay-level fixes slightly *increase* AF, because the exemplar data pipeline was the dominant bottleneck.
- Online exemplar augmentation, balanced oversampling, and matched classifier initialization together account for the majority of improvement.

### Accuracy Matrix (Seed 42, Full Method)

Each cell `(i, j)` shows accuracy on task `j` after training on task `i`.

|  | T0 | T1 | T2 | T3 | T4 | T5 |
|---:|---:|---:|---:|---:|---:|---:|
| After T0 | 92.66 | — | — | — | — | — |
| After T1 | 90.48 | 97.10 | — | — | — | — |
| After T2 | 86.74 | 94.60 | 94.40 | — | — | — |
| After T3 | 81.60 | 90.50 | 94.50 | 97.60 | — | — |
| After T4 | 78.42 | 88.90 | 89.30 | 97.20 | 92.20 | — |
| After T5 | 75.06 | 87.80 | 86.40 | 93.60 | 93.30 | 92.20 |

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

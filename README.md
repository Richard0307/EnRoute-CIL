# ERO-MoE-CIL: Privacy-Preserving Open-World Continual Learning for Intelligent Cockpit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/1167839322.svg)](https://doi.org/10.5281/zenodo.18873097)

This software release provides a reference implementation of ERO-MoE-CIL, a privacy-preserving open-world class-incremental learning framework for intelligent cockpit behavior recognition. The codebase integrates a frozen Vision Transformer backbone, replay with knowledge distillation, optional Mixture-of-Experts adapters, energy-based out-of-distribution scoring, and orthogonal gradient projection. Experiments are configured on Split CIFAR-100 (50 + 10x5) and evaluated with Average Accuracy (AA), Average Forgetting (AF), and optional OOD/runtime diagnostics (AUROC, FPR@95TPR, memory, and latency). This release is intended for research reproducibility and educational use.

## Abstract
In-cabin AI systems must continually adapt to new user-specific behaviors after deployment while preserving prior knowledge and protecting sensitive personal data. This repository implements `ERO-MoE-CIL`, a parameter-efficient continual learning framework built on a frozen Vision Transformer (ViT) backbone with replay, distillation, energy-based open-world detection, Mixture-of-Experts (MoE) adapters, and orthogonal gradient projection. The system is designed for on-device learning under memory and compute constraints, without cloud upload of raw cockpit data.

## Method Summary
The framework consists of four modules:

| Module | Purpose | Implementation |
|---|---|---|
| A. Frozen backbone | Preserve generic visual features and reduce trainable parameters | `models/vit_adapter.py` (`timm` ViT-B/16, frozen) |
| B. Energy OOD detector | Flag unseen behavior patterns in open-world setting | `utils/energy_ood.py` |
| C. Energy-routed MoE adapters | Increase adaptation capacity with low overhead | `models/moe_adapter.py` |
| D. Anti-forgetting regularization | Reduce interference with old tasks | Replay + KD in `trainer.py`, optional orthogonal projection in `utils/orthogonal_projection.py` |

Core training objective:

`L_total = L_CE + lambda_kd * L_KD + L_balance(MoE, optional)`

Orthogonal projection is applied at the gradient-update level when `--use_ortho_proj` is enabled.

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
├── scripts/plot_results.py      # Figure generation
├── blueprint.md                 # Research/system blueprint
└── docs/PROJECT_GUIDE.md        # Extended project guide
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
```

## Training Commands

### Baseline
```bash
python main.py --epochs 20 --batch_size 32 \
  --output_dir output/baseline \
  --save_best
```

### Ablation settings
```bash
# + MoE
python main.py --epochs 20 --batch_size 32 \
  --use_moe \
  --output_dir output/moe \
  --save_best

# + MoE + Energy OOD
python main.py --epochs 20 --batch_size 32 \
  --use_moe --use_energy_ood \
  --output_dir output/moe_energy \
  --save_best

# + MoE + Orthogonal Projection
python main.py --epochs 20 --batch_size 32 \
  --use_moe --use_ortho_proj \
  --output_dir output/moe_ortho \
  --save_best

# Full ERO-MoE-CIL
python main.py --epochs 20 --batch_size 32 \
  --use_moe --use_energy_ood --use_ortho_proj \
  --output_dir output/full \
  --save_best
```

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

## Output Artifacts
After each run, the pipeline generates:
- `acc_matrix.npy`
- `training_metrics.npz` (runtime + OOD logs, if available)
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

## Current Results in Repository Artifacts
The following values are computed directly from committed experiment outputs:

| Artifact directory | Accuracy matrix | Final AA | Final AF |
|---|---:|---:|---:|
| `output/` | `6 x 6` | `85.03%` | `14.88%` |
| `output_e20/` | `6 x 6` | `85.71%` | `14.35%` |

To recompute from saved matrices:

```bash
python - <<'PY'
import numpy as np
for d in ["output", "output_e20"]:
    acc = np.load(f"{d}/acc_matrix.npy")
    T = acc.shape[0]
    aa = np.mean(acc[T-1, :T])
    af = np.mean([max(acc[s][i] for s in range(i, T-1)) - acc[T-1][i] for i in range(T-1)])
    print(d, f"AA={aa*100:.2f}%", f"AF={af*100:.2f}%")
PY
```

## Reproducibility Notes
- `--seed` controls class order and random initialization.
- CUDNN deterministic mode is enabled in `main.py`.
- For fair comparisons, keep `--epochs`, `--batch_size`, and `--output_dir` conventions fixed across ablations.
- Use a distinct output directory per experiment to avoid overwriting results.

## Privacy and Ethics
- This codebase is designed for privacy-preserving continual learning where raw user behavior data remains local.
- No cloud data transmission is required by default.
- Deployment on real cockpit data should follow applicable data governance, consent, and safety regulations.

## Limitations
- Current public benchmark is Split CIFAR-100; real in-cabin datasets are not yet integrated in this repository.
- Statistical reporting across multiple seeds is not yet automated.
- OOD evaluation currently uses unseen future-task classes as a proxy for open-world unknowns.



<!-- ## Citation
If you use this repository, please cite the software release:

```bibtex
@software{ero_moe_cil_2026,
  author    = {Richard},
  title     = {ERO-MoE-CIL: Privacy-Preserving Open-World Continual Learning for Intelligent Cockpit},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.xxxxxxx},
  url       = {https://doi.org/10.5281/zenodo.18873098}
}
``` -->


## License
MIT License. See [LICENSE](LICENSE).

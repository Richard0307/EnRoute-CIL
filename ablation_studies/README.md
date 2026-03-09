# Ablation Studies

This folder records the ablation logic behind the current public EnRoute-CIL benchmark.

## Current Public Ablation

The strongest public ablation currently available in the repository compares:

- `ours_baseline`: frozen ViT + standard adapter continual learning
- `ours`: full EnRoute-CIL with the routed adaptation stack

This is a **system-level ablation**. It is not a pure single-switch isolation of `L_skew`, but it captures the practical cost/benefit of enabling the routed adaptation design that `L_skew` stabilizes.

### CIFAR-100 (`50 + 10 x 5`, 3 seeds, 5 epochs/task)

| Configuration | AA (mean ± std) | AF (mean ± std) | Final Old-Task Acc |
|---|---:|---:|---:|
| `ours_baseline` | 85.10% ± 1.38% | 14.23% ± 1.73% | 82.54% ± 1.76% |
| `ours` | 86.02% ± 0.78% | 10.25% ± 0.70% | 84.55% ± 0.81% |

Derived deltas:

- `AA`: `+0.91` percentage points
- `AF`: `-3.98` percentage points
- Relative AF reduction: `27.94%`

## Why This Ablation Matters

The numerical pattern is the important part:

- absolute accuracy improves, but only modestly;
- forgetting improves much more strongly in relative terms;
- the net system effect is therefore about **stability under constrained adaptation**, not only about peak accuracy.

In other words, the public benchmark supports the claim that EnRoute-CIL is useful because it improves the **accuracy/forgetting trade-off**, not because it wins by a large AA margin alone.

## Artifact Sources

The values above are taken directly from:

- `output/benchmark_sota/cifar100/ours/multiseed_summary.json`
- `output/benchmark_sota/cifar100/ours_baseline/multiseed_summary.json`

## Next-Step Ablations

The next ablations worth adding are:

1. isolated `L_skew` on/off under the same MoE routing stack
2. OOD-guided routing on/off while keeping the same replay pipeline
3. visualization-backed comparison of feature-space collapse vs expert specialization

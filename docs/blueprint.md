# System Blueprint: Privacy-Preserving Open-World CIL for Intelligent Cockpit
It defines the production target architecture and an implementation path from the current repo baseline.

---

## 1. Problem Definition

### 1.1 Stream Setting
- Data arrives as incremental stages:
  - $\mathcal{D} = \{\mathcal{D}_1, \mathcal{D}_2, \ldots, \mathcal{D}_T\}$
  - $\mathcal{D}_t = \{(x_i^t, y_i^t)\}_{i=1}^{N_t}$, $y_i^t \in \mathcal{C}_t$
- Goal after stage $t$:
  - Maintain performance on $\bigcup_{i=1}^{t}\mathcal{C}_i$
  - Minimize catastrophic forgetting on old classes

### 1.2 Hard Constraints
- Privacy: no raw in-cabin stream upload to cloud.
- Memory: only local, bounded replay memory $\mathcal{M}$ is allowed.
- Compute: edge-feasible training/inference (single GPU class budget, proposal uses RTX 4090 as proxy).
- Open-world: system must detect OOD behaviors instead of forcing closed-set prediction.

---

## 2. Target Architecture (ERO-MoE-CIL)

### Module A. Frozen Perception Backbone
- Backbone: pre-trained ViT (e.g., `ViT-B/16`).
- Rule: backbone parameters are frozen (`requires_grad=False`).
- Output: feature embedding $h=f(x)\in\mathbb{R}^d$ (e.g., $d=768$).

### Module B. Energy-Based Open-World Detector
- Compute energy score on classifier logits $f_i(h)$:
  $$
  E(h) = -T \log \sum_{i=1}^{C}\exp\left(\frac{f_i(h)}{T}\right)
  $$
- Routing decision:
  - If $E(h)\le\tau$: process with existing experts/classes.
  - If $E(h)>\tau$: mark as OOD and trigger capacity expansion (new expert path).
- Threshold policy:
  - Dynamic $\tau$ from validation percentile per stage (recommended), not fixed global constant.

### Module C. Energy-Routed MoE PEFT Updater
- Replace monolithic adapter-only update with Mixture-of-Experts (MoE) PEFT:
  - Shared frozen ViT backbone.
  - Multiple lightweight experts (LoRA/Adapter style) with routing.
- Macro anti-forgetting:
  - Route novel/OOD behavior to dormant or low-conflict experts.
  - Avoid overwriting one shared adapter for all tasks.
- Edge requirement:
  - Keep inference growth controlled; activate minimal experts per sample.

### Module D. Orthogonal Projection + Replay + Distillation
- Replay memory:
  - Herding selection, at most $K$ exemplars/class.
- Distillation:
  - Keep old model outputs as soft targets for seen classes.
- Orthogonal projection update (OPLoRA idea):
  - Project raw update gradient to null space of historical basis $U_{\text{old}}$:
  $$
  \nabla W_{\text{proj}} = (I - U_{\text{old}}U_{\text{old}}^\top)\nabla W_{\text{new}}
  $$
  - Purpose: suppress intra-expert interference during incremental updates.

---

## 3. Training Objective and Optimization

### 3.1 Core Loss
Use the proposal-consistent primary objective:
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{CE} + \lambda_{KD}\mathcal{L}_{KD}
$$
- $\mathcal{L}_{CE}$: supervised loss on current task labels.
- $\mathcal{L}_{KD}$: distillation on old classes.
- Orthogonality is enforced in the update rule via projected gradients (Module D), not by default as an extra loss term.
- Optional ablation-only variant may add $\mathcal{L}_{OP}$ to compare projected-vs-penalized behavior.

### 3.2 Stable Optimization
- Differential learning rates for modules (proposal guidance):
  - $\eta_{adapter/expert} \gg \eta_{top\_vit}$
- Warmup + cosine annealing scheduler.
- Gradient clipping (`L2` norm).
- Mixed precision allowed if numerically stable.

---

## 4. End-to-End Stage Lifecycle

For each stage $t$:
1. Build training set from new classes + replay exemplars.
2. Run open-world energy scoring and routing.
3. Train PEFT experts/head with CE + KD + OP constraints.
4. Update replay memory via herding.
5. Evaluate on all seen tasks and record accuracy matrix.
6. Save checkpoint (`task_xxx.pt`, `latest.pt`, optional `best.pt`).
7. Generate plots and logs (AA/AF curves + heatmap).

---

## 5. Evaluation Protocol

### 5.1 Primary CIL Metrics
- Average Accuracy (AA):
$$
AA = \frac{1}{T}\sum_{i=1}^{T}A_{T,i}
$$
- Average Forgetting (AF):
$$
AF = \frac{1}{T-1}\sum_{i=1}^{T-1}\max_{t\in\{i,\ldots,T-1\}}(A_{t,i}-A_{T,i})
$$

### 5.2 Open-World and Routing Diagnostics
- OOD detection quality:
  - AUROC, FPR@95TPR, in-distribution accuracy drop.
- Routing quality:
  - Expert utilization entropy.
  - Dormant-expert activation rate.
  - Conflict rate (samples routed to overloaded experts).

### 5.3 Resource Metrics
- Trainable parameter ratio.
- Peak VRAM and training throughput.
- Inference latency change after incremental stages.

---

## 6. Dataset and Benchmark Plan

- Algorithm benchmark: Split-CIFAR-100 (`50 + 10x5`).
- Cockpit scenario validation: State Farm Distracted Driver dataset.
- Baselines to keep:
  - Naive fine-tuning.
  - iCaRL-style replay + KD baseline.
  - Current repo adapter baseline.

---

## 7. Implementation Roadmap (Repo-Aligned)

### Phase 0 (already in repo)
- Frozen ViT + Adapter + incremental head.
- Replay herding + KD.
- AA/AF computation + plotting.
- Resume/checkpoint (`latest`, `task_xxx`, `best`).

### Phase 1
- Add Module B energy scoring API and threshold calibration.
- Add OOD metrics logging.

### Phase 2
- Introduce MoE expert container (LoRA/Adapter experts).
- Add routing gate and dormant expert activation policy.

### Phase 3
- Implement orthogonal projection updates for expert parameters.
- Add ablation toggles: `w/ OP`, `w/o OP`, `single expert`, `multi expert`.

### Phase 4
- Port and validate on State Farm dataset with privacy-preserving data handling.

---

## 8. Deliverables Checklist

- `acc_matrix.npy`, AA/AF plots, and run configs for every experiment.
- Checkpoints with replay state and resume support.
- OOD/routing diagnostics report.
- Ablation table:
  - Baseline Adapter
  - + Energy routing
  - + MoE
  - + OPLoRA
  - Full ERO-MoE pipeline

---

## 9. Non-Negotiable Rules

- Never unfreeze full backbone for routine incremental updates.
- Never rely on cloud storage of raw in-cabin streams.
- Never evaluate only final-task accuracy; always report AA + AF.
- Never claim open-world capability without OOD metrics.

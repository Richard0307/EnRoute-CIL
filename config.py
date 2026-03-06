"""
Configuration for Open-World CIL experiments.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ── Dataset ──
    dataset: str = "cifar100"
    data_root: str = "./data/raw"
    num_workers: int = 4

    # ── Incremental Schedule ──
    # Base phase has `init_classes` classes; each subsequent phase adds `inc_classes`.
    init_classes: int = 50
    inc_classes: int = 10

    # ── Model ──
    backbone: str = "vit_base_patch16_224"
    pretrained: bool = True
    feature_dim: int = 768          # ViT-B output dimension
    adapter_bottleneck: int = 64    # Adapter down-projection dimension

    # ── Memory ──
    exemplars_per_class: int = 20   # K in herding selection
    oversample_exemplars: bool = True  # Oversample exemplars to balance with new-class data

    # ── Training ──
    epochs: int = 20
    batch_size: int = 64
    lr_adapter: float = 1e-3        # LR for adapters & classifier head
    lr_backbone_top: float = 1e-5   # LR for unfrozen top blocks (if any)
    weight_decay: float = 5e-4
    warmup_epochs: int = 3
    grad_clip_norm: float = 1.0

    # ── Knowledge Distillation ──
    kd_lambda: float = 1.0          # λ weight for distillation loss
    kd_temperature: float = 2.0     # T for soft-label distillation

    # ── MoE (Module C) ──
    use_moe: bool = False           # Enable MoE adapters instead of single adapter
    num_experts: int = 2            # Initial number of experts per MoE layer
    top_k: int = 1                  # Number of experts activated per sample
    add_expert_per_task: bool = True # Add a new expert when learning each new task

    # ── Energy OOD (Module B) ──
    use_energy_ood: bool = False    # Enable energy-based OOD detection
    energy_temperature: float = 1.0 # Temperature for energy score computation
    ood_percentile: float = 95.0    # Percentile for threshold calibration

    # ── Orthogonal Projection (Module D extension) ──
    use_ortho_proj: bool = False    # Enable orthogonal gradient projection
    ortho_max_rank: int = 20        # Max rank of historical subspace per param

    # ── DER++ (Dark Experience Replay++) ──
    der_alpha: float = 0.1          # MSE loss coefficient for stored-logit replay

    # ── Weight Aligning (WA) ──
    use_wa: bool = True             # Align old/new classifier weight norms after each task

    # ── Device ──
    device: str = "cuda"

    # ── Logging ──
    output_dir: str = "./output"
    seed: int = 42

    # ── Checkpoint / Resume ──
    checkpoint_dir: str = ""         # Empty => <output_dir>/checkpoints
    auto_checkpoint: bool = True     # Save one checkpoint after each finished task
    save_best: bool = False          # Save extra best checkpoint by AA
    resume: bool = False             # Resume from checkpoint
    resume_path: str = ""            # Explicit checkpoint path; empty => latest

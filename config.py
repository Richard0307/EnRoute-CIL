"""Configuration for CIL experiments."""

from dataclasses import dataclass


@dataclass
class Config:
    # Dataset
    dataset: str = "cifar100"
    data_root: str = "./data/raw"
    class_order_path: str = ""
    num_workers: int = 4

    # Incremental schedule
    init_classes: int = 50
    inc_classes: int = 10
    max_tasks: int = -1

    # Model
    backbone: str = "vit_base_patch16_224"
    pretrained: bool = True
    feature_dim: int = 768
    adapter_bottleneck: int = 64

    # Memory
    exemplars_per_class: int = 20
    online_exemplar_augmentation: bool = True
    oversample_exemplars: bool = True

    # Training
    epochs: int = 20
    batch_size: int = 64
    lr_adapter: float = 1e-3
    lr_backbone_top: float = 1e-5
    weight_decay: float = 5e-4
    warmup_epochs: int = 3
    grad_clip_norm: float = 1.0

    # Knowledge distillation
    kd_lambda: float = 1.0
    kd_temperature: float = 2.0

    # MoE
    use_moe: bool = False
    num_experts: int = 2
    top_k: int = 1
    add_expert_per_task: bool = True

    # Energy OOD
    use_energy_ood: bool = False
    energy_temperature: float = 1.0
    ood_percentile: float = 95.0
    use_ood_expert_routing: bool = False
    ood_router_lambda: float = 0.2
    ood_router_temperature: float = 1.0
    ood_trigger_min_count: int = 20
    ood_trigger_min_ratio: float = 0.05
    ood_cache_max_per_task: int = 128

    # Orthogonal projection
    use_ortho_proj: bool = False
    ortho_max_rank: int = 20

    # DER++
    der_alpha: float = 0.1

    # Weight Aligning
    use_wa: bool = True
    scale_matched_head_init: bool = True

    # Device
    device: str = "cuda"

    # Logging
    output_dir: str = "./output"
    seed: int = 42
    auto_plots: bool = True

    # Checkpoint / Resume
    checkpoint_dir: str = ""
    auto_checkpoint: bool = True
    save_best: bool = False
    resume: bool = False
    resume_path: str = ""

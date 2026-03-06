"""
Entry point for Open-World CIL experiments (ERO-MoE-CIL).

Usage:
    python main.py                                  # Baseline adapter + KD
    python main.py --use_moe                        # + MoE routing
    python main.py --use_moe --use_energy_ood       # + Energy OOD detection
    python main.py --use_moe --use_ortho_proj       # + Orthogonal projection
    python main.py --use_moe --use_energy_ood --use_ortho_proj  # Full pipeline
    python main.py --resume                          # Resume from latest checkpoint
    python main.py --save_best                       # Save best checkpoint by AA
"""

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from config import Config
from trainer import CILTrainer
from utils.data_utils import build_cifar100_tasks


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="ERO-MoE-CIL for Intelligent Cockpit")
    cfg = Config()

    # Dataset & schedule
    parser.add_argument("--dataset", type=str, default=cfg.dataset)
    parser.add_argument("--data_root", type=str, default=cfg.data_root)
    parser.add_argument("--init_classes", type=int, default=cfg.init_classes)
    parser.add_argument("--inc_classes", type=int, default=cfg.inc_classes)

    # Model
    parser.add_argument("--backbone", type=str, default=cfg.backbone)
    parser.add_argument("--adapter_bottleneck", type=int, default=cfg.adapter_bottleneck)
    parser.add_argument("--exemplars_per_class", type=int, default=cfg.exemplars_per_class)

    # Training
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr_adapter", type=float, default=cfg.lr_adapter)
    parser.add_argument("--kd_lambda", type=float, default=cfg.kd_lambda)
    parser.add_argument("--kd_temperature", type=float, default=cfg.kd_temperature)

    # Module B: Energy OOD
    parser.add_argument("--use_energy_ood", action="store_true",
                        help="Enable energy-based OOD detection (Module B)")
    parser.add_argument("--energy_temperature", type=float, default=cfg.energy_temperature)
    parser.add_argument("--ood_percentile", type=float, default=cfg.ood_percentile)

    # Module C: MoE
    parser.add_argument("--use_moe", action="store_true",
                        help="Enable MoE adapters (Module C)")
    parser.add_argument("--num_experts", type=int, default=cfg.num_experts)
    parser.add_argument("--top_k", type=int, default=cfg.top_k)
    parser.add_argument("--no_add_expert", action="store_true",
                        help="Disable adding new expert per task")

    # Module D extension: Orthogonal Projection
    parser.add_argument("--use_ortho_proj", action="store_true",
                        help="Enable orthogonal gradient projection (Module D ext)")
    parser.add_argument("--ortho_max_rank", type=int, default=cfg.ortho_max_rank)

    # DER++ & WA
    parser.add_argument("--der_alpha", type=float, default=cfg.der_alpha,
                        help="DER++ MSE loss coefficient (0 to disable)")
    parser.add_argument("--no_wa", action="store_true",
                        help="Disable Weight Aligning")
    parser.add_argument("--no_oversample", action="store_true",
                        help="Disable exemplar oversampling")

    # System
    parser.add_argument("--device", type=str, default=cfg.device)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--output_dir", type=str, default=cfg.output_dir)
    parser.add_argument("--checkpoint_dir", type=str, default=cfg.checkpoint_dir)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from a checkpoint")
    parser.add_argument("--resume_path", type=str, default=cfg.resume_path,
                        help="Path to checkpoint .pt file (default: latest checkpoint)")
    parser.add_argument("--save_best", action="store_true",
                        help="Save best checkpoint by AA as <checkpoint_dir>/best.pt")
    parser.add_argument("--no_auto_checkpoint", action="store_true",
                        help="Disable automatic checkpoint save after each task")

    args = parser.parse_args()
    args_dict = vars(args)
    disable_auto_checkpoint = args_dict.pop("no_auto_checkpoint")
    no_wa = args_dict.pop("no_wa")
    no_oversample = args_dict.pop("no_oversample")

    # Update config with parsed arguments
    for key, value in args_dict.items():
        if key == "no_add_expert":
            continue
        setattr(cfg, key, value)

    if args.no_add_expert:
        cfg.add_expert_per_task = False
    cfg.auto_checkpoint = not disable_auto_checkpoint
    if no_wa:
        cfg.use_wa = False
    if no_oversample:
        cfg.oversample_exemplars = False

    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    print("=" * 60)
    print("ERO-MoE-CIL for Intelligent Cockpit")
    print("=" * 60)
    print(f"Config: {cfg}")
    print()

    trainer = CILTrainer(cfg)
    resume_checkpoint = None
    if cfg.resume:
        try:
            resume_checkpoint = trainer.resolve_resume_path(cfg.resume_path)
            print(f"Resuming from checkpoint: {resume_checkpoint}")
        except FileNotFoundError as exc:
            raise SystemExit(f"[Error] {exc}") from exc

    # Build incremental task schedule
    task_classes, train_dataset, test_dataset = build_cifar100_tasks(
        data_root=cfg.data_root,
        init_classes=cfg.init_classes,
        inc_classes=cfg.inc_classes,
        seed=cfg.seed,
    )

    print(f"Total tasks: {len(task_classes)}")
    for i, classes in enumerate(task_classes):
        print(f"  Task {i}: {len(classes)} classes → {classes[:5]}{'...' if len(classes) > 5 else ''}")
    print()

    # Run CIL training
    try:
        acc_matrix = trainer.run(
            task_classes,
            train_dataset,
            test_dataset,
            resume_checkpoint=resume_checkpoint,
        )
    except ValueError as exc:
        raise SystemExit(f"[Error] {exc}") from exc

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    from utils.metrics import compute_average_accuracy, compute_average_forgetting
    print(f"Average Accuracy (AA): {compute_average_accuracy(acc_matrix):.4f}")
    print(f"Average Forgetting (AF): {compute_average_forgetting(acc_matrix):.4f}")
    print(f"\nFull accuracy matrix saved to: {cfg.output_dir}/acc_matrix.npy")

    runtime_metrics_path = Path(cfg.output_dir) / "training_metrics.npz"
    try:
        from scripts.plot_results import generate_all_plots
        plot_paths = generate_all_plots(
            acc_matrix=acc_matrix,
            output_dir=cfg.output_dir,
            runtime_metrics_path=runtime_metrics_path,
            verbose=False,
        )
    except Exception as exc:
        print(f"\n[Warning] Failed to generate plots automatically: {exc}")
    else:
        print("\nGenerated plots:")
        for plot_name, plot_path in plot_paths.items():
            print(f"  {plot_name}: {plot_path}")


if __name__ == "__main__":
    main()

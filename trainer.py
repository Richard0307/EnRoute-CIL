"""CIL Trainer: core training loop for ERO-MoE-CIL pipeline."""

from dataclasses import asdict
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from config import Config
from models import CILModel
from utils.data_utils import (
    ClassSubset,
    get_task_dataloaders,
    TRAIN_TRANSFORM,
    TEST_TRANSFORM,
)
from utils.energy_ood import (
    collect_energy_scores,
    calibrate_threshold,
    evaluate_ood,
)
from utils.herding import extract_class_features, herding_select
from utils.metrics import print_metrics
from utils.orthogonal_projection import OrthogonalProjector


class CILTrainer:
    """Orchestrates the full ERO-MoE-CIL pipeline."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model: Optional[CILModel] = None
        self.old_model: Optional[CILModel] = None
        self.exemplar_sets: dict = {}
        self.exemplar_logits: dict = {}
        self.acc_matrix: Optional[np.ndarray] = None
        self.seen_classes: List[int] = []
        self.label_map: dict = {}

        self.projector: Optional[OrthogonalProjector] = None
        if cfg.use_ortho_proj:
            self.projector = OrthogonalProjector(max_rank=cfg.ortho_max_rank)

        self.ood_metrics_log: List[Dict[str, float]] = []
        self.energy_thresholds: List[float] = []

        self.task_old_task_accuracy: List[float] = []
        self.task_max_memory_mb: List[float] = []
        self.task_avg_epoch_time_sec: List[float] = []
        self.task_avg_batch_latency_ms: List[float] = []
        self.epoch_records: List[Dict[str, float]] = []

        self.best_aa: float = float("-inf")
        self.best_task_id: int = -1

        tb_dir = Path(cfg.output_dir) / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"TensorBoard log dir: {tb_dir}")

        self._global_step: int = 0
        self._global_batch_step: int = 0
        self._tb_batch_log_interval: int = 10

    def _train_one_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        num_old_classes: int,
    ) -> Dict[str, List[float]]:
        """Train the model for one incremental task."""
        self.model.train()

        if self.projector is not None and task_id > 0:
            self.projector.snapshot_params(self.model)

        trainable_params = self.model.get_trainable_params()
        optimizer = AdamW(trainable_params, lr=self.cfg.lr_adapter,
                          weight_decay=self.cfg.weight_decay)

        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=self.cfg.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer,
                                   T_max=self.cfg.epochs - self.cfg.warmup_epochs)
        scheduler = SequentialLR(optimizer, [warmup, cosine],
                                 milestones=[self.cfg.warmup_epochs])

        ce_criterion = nn.CrossEntropyLoss()
        epoch_time_sec: List[float] = []
        epoch_avg_batch_latency_ms: List[float] = []
        epoch_peak_memory_mb: List[float] = []
        num_batches = max(1, len(train_loader))

        der_loader = None
        der_iter = None
        if self.cfg.der_alpha > 0 and task_id > 0:
            der_loader = self._build_der_replay_loader()

        for epoch in range(self.cfg.epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            epoch_start = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            if der_loader is not None:
                der_iter = iter(der_loader)

            pbar = tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch+1}/{self.cfg.epochs}")
            for batch_idx, (images, labels) in enumerate(pbar, start=1):
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)

                loss_ce = ce_criterion(logits, labels)

                loss_kd = torch.tensor(0.0, device=self.device)
                if self.old_model is not None and num_old_classes > 0:
                    with torch.no_grad():
                        old_logits = self.old_model(images)[:, :num_old_classes]
                    new_logits = logits[:, :num_old_classes]
                    T = self.cfg.kd_temperature
                    loss_kd = F.kl_div(
                        F.log_softmax(new_logits / T, dim=1),
                        F.softmax(old_logits / T, dim=1),
                        reduction="batchmean",
                    ) * (T * T)

                loss_balance = torch.tensor(0.0, device=self.device)
                if self.cfg.use_moe:
                    loss_balance = self.model.get_moe_balance_loss().to(self.device)

                loss_der = torch.tensor(0.0, device=self.device)
                if der_iter is not None:
                    try:
                        der_imgs, _der_lbls, der_stored = next(der_iter)
                    except StopIteration:
                        der_iter = iter(der_loader)
                        der_imgs, _der_lbls, der_stored = next(der_iter)
                    der_imgs = der_imgs.to(self.device)
                    der_stored = der_stored.to(self.device)
                    der_current = self.model(der_imgs)
                    n_stored = der_stored.shape[1]
                    loss_der = F.mse_loss(der_current[:, :n_stored], der_stored)

                loss = (loss_ce + self.cfg.kd_lambda * loss_kd
                        + loss_balance + self.cfg.der_alpha * loss_der)

                optimizer.zero_grad()
                loss.backward()

                if self.projector is not None and self.projector.has_basis():
                    self.projector.project_gradients(self.model)

                torch.nn.utils.clip_grad_norm_(trainable_params,
                                               self.cfg.grad_clip_norm)
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{100.*correct/total:.1f}%",
                )

                self._global_batch_step += 1
                if (
                    batch_idx % self._tb_batch_log_interval == 0
                    or batch_idx == num_batches
                ):
                    running_loss = total_loss / max(total, 1)
                    running_acc = 100.0 * correct / max(total, 1)
                    self.writer.add_scalar("Batch/Training_Loss", loss.item(), self._global_batch_step)
                    self.writer.add_scalar("Batch/Running_Loss", running_loss, self._global_batch_step)
                    self.writer.add_scalar("Batch/Running_Accuracy", running_acc, self._global_batch_step)
                    self.writer.add_scalar("Batch/Learning_Rate", optimizer.param_groups[0]["lr"], self._global_batch_step)
                    if self.device.type == "cuda":
                        current_mem_mb = torch.cuda.memory_allocated(self.device) / (1024.0 ** 2)
                        self.writer.add_scalar("Batch/Current_GPU_Memory_MB", current_mem_mb, self._global_batch_step)
                    self.writer.flush()

            scheduler.step()
            epoch_duration = time.perf_counter() - epoch_start
            avg_batch_latency_ms = (epoch_duration / num_batches) * 1000.0
            if self.device.type == "cuda":
                peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
            else:
                peak_mem_mb = 0.0

            epoch_time_sec.append(epoch_duration)
            epoch_avg_batch_latency_ms.append(avg_batch_latency_ms)
            epoch_peak_memory_mb.append(peak_mem_mb)
            self.epoch_records.append({
                "task_id": float(task_id),
                "epoch": float(epoch + 1),
                "epoch_time_sec": float(epoch_duration),
                "avg_batch_latency_ms": float(avg_batch_latency_ms),
                "peak_memory_mb": float(peak_mem_mb),
            })

            epoch_loss = total_loss / total
            epoch_acc = 100.0 * correct / total

            self._global_step += 1
            self.writer.add_scalar("Epoch/Training_Loss", epoch_loss, self._global_step)
            self.writer.add_scalar("Epoch/Training_Accuracy", epoch_acc, self._global_step)
            self.writer.add_scalar("Epoch/Learning_Rate", optimizer.param_groups[0]["lr"], self._global_step)
            self.writer.add_scalar("Epoch/Peak_GPU_Memory_MB", peak_mem_mb, self._global_step)
            if self.device.type == "cuda":
                try:
                    gpu_util = torch.cuda.utilization(self.device)
                    self.writer.add_scalar("Epoch/GPU_Utilization_Pct", gpu_util, self._global_step)
                except Exception:
                    pass
            self.writer.add_scalar("Epoch/Wall_Clock_Time_s", epoch_duration, self._global_step)
            self.writer.flush()

            print(
                f"  Epoch {epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.1f}%, "
                f"time={epoch_duration:.2f}s, latency={avg_batch_latency_ms:.2f}ms/batch, "
                f"peak_mem={peak_mem_mb:.1f}MB"
            )

        if self.projector is not None and task_id > 0:
            self.projector.update_bases(self.model)
            stats = self.projector.get_stats()
            total_rank = sum(stats.values())
            print(f"  Orthogonal projection: {len(stats)} params, total rank={total_rank}")

        return {
            "epoch_time_sec": epoch_time_sec,
            "epoch_avg_batch_latency_ms": epoch_avg_batch_latency_ms,
            "epoch_peak_memory_mb": epoch_peak_memory_mb,
        }

    @torch.no_grad()
    def _wa_compute_gamma(self, num_old_classes: int) -> float:
        """Compute WA scaling factor: γ = mean_norm_new / mean_norm_old."""
        fc = self.model.classifier.fc
        norm_per_class = fc.weight.data.norm(dim=1)
        old_mean = norm_per_class[:num_old_classes].mean().item()
        new_mean = norm_per_class[num_old_classes:].mean().item()
        if old_mean < 1e-8:
            return 1.0
        gamma = new_mean / old_mean
        print(f"  WA: γ={gamma:.4f} (old_norm={old_mean:.4f}, new_norm={new_mean:.4f})")
        return gamma

    @torch.no_grad()
    def _wa_apply(self, num_old_classes: int, gamma: float) -> None:
        fc = self.model.classifier.fc
        fc.weight.data[:num_old_classes] *= gamma
        fc.bias.data[:num_old_classes] *= gamma

    @torch.no_grad()
    def _wa_revert(self, num_old_classes: int, gamma: float) -> None:
        fc = self.model.classifier.fc
        fc.weight.data[:num_old_classes] /= gamma
        fc.bias.data[:num_old_classes] /= gamma

    @torch.no_grad()
    def _evaluate_task(
        self,
        test_dataset,
        class_ids: List[int],
    ) -> float:
        self.model.eval()
        test_subset = ClassSubset(test_dataset, class_ids, transform=TEST_TRANSFORM,
                                  label_map=self.label_map)
        loader = DataLoader(test_subset, batch_size=self.cfg.batch_size,
                            shuffle=False, num_workers=self.cfg.num_workers,
                            pin_memory=True)
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return correct / total if total > 0 else 0.0

    def _evaluate_ood(
        self,
        test_dataset,
        seen_class_ids: List[int],
        unseen_class_ids: List[int],
        task_id: int,
    ) -> Dict[str, float]:
        """Evaluate OOD detection: seen classes as ID, unseen as OOD."""
        self.model.eval()

        id_subset = ClassSubset(test_dataset, seen_class_ids,
                                transform=TEST_TRANSFORM, label_map=self.label_map)
        id_loader = DataLoader(id_subset, batch_size=self.cfg.batch_size,
                               shuffle=False, num_workers=self.cfg.num_workers,
                               pin_memory=True)

        ood_subset = ClassSubset(test_dataset, unseen_class_ids,
                                 transform=TEST_TRANSFORM, label_map=None)
        ood_loader = DataLoader(ood_subset, batch_size=self.cfg.batch_size,
                                shuffle=False, num_workers=self.cfg.num_workers,
                                pin_memory=True)

        id_scores = collect_energy_scores(
            self.model, id_loader,
            temperature=self.cfg.energy_temperature,
            device=str(self.device),
        )
        ood_scores = collect_energy_scores(
            self.model, ood_loader,
            temperature=self.cfg.energy_temperature,
            device=str(self.device),
        )

        tau = calibrate_threshold(id_scores, percentile=self.cfg.ood_percentile)
        self.energy_thresholds.append(tau)

        ood_results = evaluate_ood(id_scores, ood_scores)
        ood_results["threshold"] = tau
        ood_results["task_id"] = task_id
        ood_results["id_mean_energy"] = float(np.mean(id_scores))
        ood_results["ood_mean_energy"] = float(np.mean(ood_scores))

        return ood_results

    def _update_exemplars(
        self,
        train_dataset,
        new_class_ids: List[int],
    ) -> None:
        """Select exemplars via herding and store DER++ logits."""
        print("  Updating exemplar memory via herding...")
        train_subset = ClassSubset(train_dataset, new_class_ids,
                                   transform=TRAIN_TRANSFORM,
                                   label_map=self.label_map)

        for class_id in new_class_ids:
            features, samples = extract_class_features(
                self.model, train_subset, class_id,
                device=str(self.device),
                batch_size=self.cfg.batch_size,
                label_map=self.label_map,
            )
            selected = herding_select(features, samples,
                                      k=self.cfg.exemplars_per_class)
            self.exemplar_sets[class_id] = selected
            print(f"    Class {class_id}: {len(selected)} exemplars selected")

        if self.cfg.der_alpha > 0:
            self._store_exemplar_logits()

    @torch.no_grad()
    def _store_exemplar_logits(self) -> None:
        """Cache current model logits for all exemplars (DER++)."""
        self.model.eval()
        self.exemplar_logits = {}
        for class_id, samples in self.exemplar_sets.items():
            images = torch.stack([TEST_TRANSFORM(img) for img, _ in samples]).to(self.device)
            logits_list = []
            for i in range(0, len(images), self.cfg.batch_size):
                batch = images[i:i + self.cfg.batch_size]
                logits_list.append(self.model(batch).cpu())
            self.exemplar_logits[class_id] = torch.cat(logits_list, dim=0)
        self.model.train()

    def _build_der_replay_loader(self) -> Optional[DataLoader]:
        """Build a DataLoader of (image, label, stored_logits) for DER++ replay."""
        if not self.exemplar_logits:
            return None
        images, labels, logits = [], [], []
        for class_id in sorted(self.exemplar_sets.keys()):
            stored_logits = self.exemplar_logits[class_id]
            for idx, (img, lbl) in enumerate(self.exemplar_sets[class_id]):
                images.append(TEST_TRANSFORM(img))
                labels.append(lbl)
                logits.append(stored_logits[idx])
        dataset = torch.utils.data.TensorDataset(
            torch.stack(images),
            torch.tensor(labels, dtype=torch.long),
            torch.stack(logits),
        )
        return DataLoader(dataset, batch_size=self.cfg.batch_size,
                          shuffle=True, drop_last=False)

    def _get_all_exemplars(self) -> List[Tuple]:
        all_exemplars = []
        for class_id in sorted(self.exemplar_sets.keys()):
            all_exemplars.extend(self.exemplar_sets[class_id])
        return all_exemplars

    def _save_runtime_metrics(self, output_dir: Path) -> Path:
        metrics_path = output_dir / "training_metrics.npz"

        task_ids = np.arange(len(self.task_old_task_accuracy), dtype=int)
        old_task_acc = np.array(self.task_old_task_accuracy, dtype=float)
        max_memory_mb = np.array(self.task_max_memory_mb, dtype=float)
        avg_epoch_time_sec = np.array(self.task_avg_epoch_time_sec, dtype=float)
        avg_batch_latency_ms = np.array(self.task_avg_batch_latency_ms, dtype=float)

        epoch_task_ids = np.array([int(r["task_id"]) for r in self.epoch_records], dtype=int)
        epoch_indices = np.array([int(r["epoch"]) for r in self.epoch_records], dtype=int)
        epoch_time_sec = np.array([r["epoch_time_sec"] for r in self.epoch_records], dtype=float)
        epoch_latency_ms = np.array([r["avg_batch_latency_ms"] for r in self.epoch_records], dtype=float)
        epoch_peak_memory_mb = np.array([r["peak_memory_mb"] for r in self.epoch_records], dtype=float)

        save_dict = dict(
            task_ids=task_ids,
            task_old_task_accuracy=old_task_acc,
            task_max_memory_mb=max_memory_mb,
            task_avg_epoch_time_sec=avg_epoch_time_sec,
            task_avg_batch_latency_ms=avg_batch_latency_ms,
            epoch_task_ids=epoch_task_ids,
            epoch_indices=epoch_indices,
            epoch_time_sec=epoch_time_sec,
            epoch_avg_batch_latency_ms=epoch_latency_ms,
            epoch_peak_memory_mb=epoch_peak_memory_mb,
        )

        if self.ood_metrics_log:
            ood_auroc = np.array([m.get("auroc", 0.0) for m in self.ood_metrics_log])
            ood_fpr = np.array([m.get("fpr_at_95tpr", 1.0) for m in self.ood_metrics_log])
            ood_task_ids = np.array([int(m.get("task_id", i)) for i, m in enumerate(self.ood_metrics_log)])
            save_dict["ood_auroc"] = ood_auroc
            save_dict["ood_fpr_at_95tpr"] = ood_fpr
            save_dict["ood_task_ids"] = ood_task_ids

        np.savez(metrics_path, **save_dict)

        global_max_mem = float(np.max(max_memory_mb)) if max_memory_mb.size > 0 else 0.0
        valid_old_acc = old_task_acc[np.isfinite(old_task_acc)]
        final_old_acc = float(valid_old_acc[-1]) if valid_old_acc.size > 0 else float("nan")
        overall_epoch_time = float(np.mean(avg_epoch_time_sec)) if avg_epoch_time_sec.size > 0 else 0.0
        overall_latency = float(np.mean(avg_batch_latency_ms)) if avg_batch_latency_ms.size > 0 else 0.0

        print("\nRuntime Metrics Summary")
        print(f"  Max Memory Allocated: {global_max_mem:.1f} MB")
        if np.isfinite(final_old_acc):
            print(f"  Accuracy on Old Tasks (final): {final_old_acc * 100:.2f}%")
        else:
            print("  Accuracy on Old Tasks (final): N/A (only base task trained)")
        print(f"  Avg Time per Epoch: {overall_epoch_time:.2f} s")
        print(f"  Avg Latency: {overall_latency:.2f} ms/batch")

        if self.ood_metrics_log:
            last_ood = self.ood_metrics_log[-1]
            print(f"  OOD AUROC (final): {last_ood.get('auroc', 0):.4f}")
            print(f"  OOD FPR@95TPR (final): {last_ood.get('fpr_at_95tpr', 1):.4f}")

        print(f"  Runtime metrics saved to {metrics_path}")

        return metrics_path

    def _get_checkpoint_dir(self) -> Path:
        if self.cfg.checkpoint_dir:
            return Path(self.cfg.checkpoint_dir)
        return Path(self.cfg.output_dir) / "checkpoints"

    def resolve_resume_path(self, resume_path: str = "") -> Path:
        if resume_path:
            candidate = Path(resume_path)
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"Checkpoint file not found: {candidate}")

        checkpoint_dir = self._get_checkpoint_dir()
        latest = checkpoint_dir / "latest.pt"
        if latest.exists():
            return latest

        task_checkpoints = sorted(checkpoint_dir.glob("task_*.pt"))
        if task_checkpoints:
            return task_checkpoints[-1]

        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}. "
            "Run training once or provide --resume_path."
        )

    def _serialize_exemplar_sets(self) -> Dict[int, List[Tuple[torch.Tensor, int]]]:
        from torchvision.transforms.functional import to_tensor
        serialized = {}
        for class_id, samples in self.exemplar_sets.items():
            packed = []
            for image, label in samples:
                if torch.is_tensor(image):
                    image_tensor = image.detach().cpu()
                else:
                    # PIL image → raw tensor (no normalization, just pixel values)
                    image_tensor = to_tensor(image).cpu()
                packed.append((image_tensor, int(label)))
            serialized[int(class_id)] = packed
        return serialized

    def _deserialize_exemplar_sets(self, payload: Dict) -> Dict[int, List[Tuple]]:
        from torchvision.transforms.functional import to_pil_image
        restored = {}
        for class_id, samples in payload.items():
            normalized = []
            for image, label in samples:
                if not torch.is_tensor(image):
                    image = torch.as_tensor(image)
                # Convert tensor back to PIL for on-the-fly augmentation
                pil_image = to_pil_image(image.cpu())
                normalized.append((pil_image, int(label)))
            restored[int(class_id)] = normalized
        return restored

    def _serialize_projector_bases(self) -> Dict[str, torch.Tensor]:
        if self.projector is None:
            return {}
        return {
            name: basis.detach().cpu()
            for name, basis in self.projector._bases.items()
        }

    def _infer_num_experts_from_state(self, model_state: Dict[str, torch.Tensor]) -> int:
        for key, tensor in model_state.items():
            if key.endswith("moe_adapter.router.gate.weight"):
                return int(tensor.shape[0])
        return int(self.cfg.num_experts)

    def _save_checkpoint(
        self,
        task_id: int,
        task_classes: List[List[int]],
        checkpoint_name: Optional[str] = None,
        update_latest: bool = True,
        is_best: bool = False,
    ) -> Path:
        if self.model is None:
            raise RuntimeError("Cannot save checkpoint: model is not initialized.")
        if self.acc_matrix is None:
            raise RuntimeError("Cannot save checkpoint: acc_matrix is not initialized.")

        checkpoint_dir = self._get_checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_state_cpu = {
            key: value.detach().cpu()
            for key, value in self.model.state_dict().items()
        }

        runtime_metrics = {
            "task_old_task_accuracy": list(self.task_old_task_accuracy),
            "task_max_memory_mb": list(self.task_max_memory_mb),
            "task_avg_epoch_time_sec": list(self.task_avg_epoch_time_sec),
            "task_avg_batch_latency_ms": list(self.task_avg_batch_latency_ms),
            "epoch_records": list(self.epoch_records),
        }

        checkpoint = {
            "version": 2,
            "completed_task_id": int(task_id),
            "model_state_dict": model_state_cpu,
            "seen_classes": [int(cls_id) for cls_id in self.seen_classes],
            "label_map": {int(k): int(v) for k, v in self.label_map.items()},
            "exemplar_sets": self._serialize_exemplar_sets(),
            "acc_matrix": self.acc_matrix.copy(),
            "task_classes": [[int(cls_id) for cls_id in task] for task in task_classes],
            "config": asdict(self.cfg),
            "model_use_moe": bool(self.cfg.use_moe),
            "model_num_experts": int(self._infer_num_experts_from_state(model_state_cpu)),
            "model_top_k": int(self.cfg.top_k),
            "projector_bases": self._serialize_projector_bases(),
            "exemplar_logits": {
                int(k): v.detach().cpu()
                for k, v in self.exemplar_logits.items()
            },
            "ood_metrics_log": list(self.ood_metrics_log),
            "energy_thresholds": list(self.energy_thresholds),
            "runtime_metrics": runtime_metrics,
            "best_aa_so_far": float(self.best_aa),
            "best_task_id": int(self.best_task_id),
            "is_best": bool(is_best),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

        if checkpoint_name is None:
            checkpoint_path = checkpoint_dir / f"task_{task_id:03d}.pt"
        else:
            checkpoint_path = checkpoint_dir / checkpoint_name

        torch.save(checkpoint, checkpoint_path)
        if update_latest:
            torch.save(checkpoint, checkpoint_dir / "latest.pt")

        print(f"  Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def _load_checkpoint(self, checkpoint_path: Path, task_classes: List[List[int]]) -> int:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        completed_task_id = int(checkpoint.get("completed_task_id", -1))
        total_tasks = len(task_classes)
        if completed_task_id < -1 or completed_task_id >= total_tasks:
            raise ValueError(
                f"Invalid completed_task_id={completed_task_id} for total_tasks={total_tasks}"
            )

        saved_task_classes = checkpoint.get("task_classes")
        current_task_classes = [[int(c) for c in task] for task in task_classes]
        if saved_task_classes is not None:
            normalized_saved = [[int(c) for c in task] for task in saved_task_classes]
            if normalized_saved != current_task_classes:
                raise ValueError(
                    "Checkpoint task schedule does not match current run config. "
                    "Use matching seed/init/inc class settings."
                )

        saved_use_moe = bool(checkpoint.get("model_use_moe", self.cfg.use_moe))
        if saved_use_moe != bool(self.cfg.use_moe):
            raise ValueError(
                "Checkpoint use_moe flag does not match current run. "
                f"(ckpt={saved_use_moe}, current={self.cfg.use_moe})"
            )

        saved_cfg = checkpoint.get("config", {})
        saved_use_ortho = bool(saved_cfg.get("use_ortho_proj", self.cfg.use_ortho_proj))
        if saved_use_ortho != bool(self.cfg.use_ortho_proj):
            raise ValueError(
                "Checkpoint use_ortho_proj flag does not match current run. "
                f"(ckpt={saved_use_ortho}, current={self.cfg.use_ortho_proj})"
            )

        model_state = checkpoint.get("model_state_dict")
        if model_state is None:
            raise ValueError("Checkpoint missing model_state_dict.")
        if "classifier.fc.weight" not in model_state:
            raise ValueError("Checkpoint is incompatible: missing classifier.fc.weight.")

        num_model_classes = int(model_state["classifier.fc.weight"].shape[0])
        num_experts = int(
            checkpoint.get("model_num_experts", self._infer_num_experts_from_state(model_state))
        )
        top_k = int(checkpoint.get("model_top_k", self.cfg.top_k))

        self.model = CILModel(
            backbone_name=self.cfg.backbone,
            pretrained=False,
            feature_dim=self.cfg.feature_dim,
            adapter_bottleneck=self.cfg.adapter_bottleneck,
            init_classes=num_model_classes,
            use_moe=saved_use_moe,
            num_experts=num_experts,
            top_k=top_k,
        ).to(self.device)
        self.model.load_state_dict(model_state, strict=True)
        self.old_model = None

        self.seen_classes = [int(cls_id) for cls_id in checkpoint.get("seen_classes", [])]
        self.label_map = {int(k): int(v) for k, v in checkpoint.get("label_map", {}).items()}
        self.exemplar_sets = self._deserialize_exemplar_sets(checkpoint.get("exemplar_sets", {}))

        saved_logits = checkpoint.get("exemplar_logits", {})
        self.exemplar_logits = {
            int(k): v.cpu() if torch.is_tensor(v) else torch.as_tensor(v)
            for k, v in saved_logits.items()
        }

        self.acc_matrix = np.zeros((total_tasks, total_tasks))
        saved_matrix = checkpoint.get("acc_matrix")
        if saved_matrix is not None:
            saved_matrix = np.asarray(saved_matrix, dtype=float)
            rows = min(saved_matrix.shape[0], total_tasks)
            cols = min(saved_matrix.shape[1], total_tasks)
            self.acc_matrix[:rows, :cols] = saved_matrix[:rows, :cols]

        self.ood_metrics_log = list(checkpoint.get("ood_metrics_log", []))
        self.energy_thresholds = list(checkpoint.get("energy_thresholds", []))

        runtime_metrics = checkpoint.get("runtime_metrics", {})
        self.task_old_task_accuracy = list(runtime_metrics.get("task_old_task_accuracy", []))
        self.task_max_memory_mb = list(runtime_metrics.get("task_max_memory_mb", []))
        self.task_avg_epoch_time_sec = list(runtime_metrics.get("task_avg_epoch_time_sec", []))
        self.task_avg_batch_latency_ms = list(runtime_metrics.get("task_avg_batch_latency_ms", []))
        self.epoch_records = list(runtime_metrics.get("epoch_records", []))
        self._global_step = len(self.epoch_records)

        if "best_aa_so_far" in checkpoint and "best_task_id" in checkpoint:
            self.best_aa = float(checkpoint["best_aa_so_far"])
            self.best_task_id = int(checkpoint["best_task_id"])
        else:
            self.best_aa = float("-inf")
            self.best_task_id = -1
            for t in range(completed_task_id + 1):
                aa_t = float(np.mean(self.acc_matrix[t, :t + 1]))
                if aa_t > self.best_aa:
                    self.best_aa = aa_t
                    self.best_task_id = t

        if self.projector is not None:
            projector_bases = checkpoint.get("projector_bases", {})
            self.projector._bases = {
                name: basis.cpu() for name, basis in projector_bases.items()
            }
        elif checkpoint.get("projector_bases"):
            raise ValueError(
                "Checkpoint contains orthogonal projector bases but current run has "
                "use_ortho_proj=False."
            )

        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.random.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng_state["cuda"])

        next_task_id = completed_task_id + 1
        print(
            f"Loaded checkpoint: completed task {completed_task_id}, "
            f"next task {next_task_id}, best AA {self.best_aa:.4f} (task {self.best_task_id})"
        )
        return next_task_id

    def run(
        self,
        task_classes: List[List[int]],
        train_dataset,
        test_dataset,
        resume_checkpoint: Optional[Path] = None,
    ) -> np.ndarray:
        """Run the full CIL pipeline across all tasks."""
        T = len(task_classes)
        if resume_checkpoint is not None:
            start_task_id = self._load_checkpoint(Path(resume_checkpoint), task_classes)
        else:
            start_task_id = 0
            self.model = None
            self.old_model = None
            self.exemplar_sets = {}
            self.exemplar_logits = {}
            self.seen_classes = []
            self.label_map = {}
            self.acc_matrix = np.zeros((T, T))
            self.task_old_task_accuracy = []
            self.task_max_memory_mb = []
            self.task_avg_epoch_time_sec = []
            self.task_avg_batch_latency_ms = []
            self.epoch_records = []
            self.ood_metrics_log = []
            self.energy_thresholds = []
            self.best_aa = float("-inf")
            self.best_task_id = -1
            if self.projector is not None:
                self.projector._bases = {}

        if start_task_id >= T:
            print("Checkpoint already contains all tasks. Skipping training loop.")

        all_class_ids = []
        for classes in task_classes:
            all_class_ids.extend(classes)

        modules = ["Adapter+KD (baseline)"]
        if self.cfg.use_moe:
            modules.append(f"MoE ({self.cfg.num_experts} experts, top-{self.cfg.top_k})")
        if self.cfg.use_energy_ood:
            modules.append("Energy OOD")
        if self.cfg.use_ortho_proj:
            modules.append(f"Orthogonal Projection (rank≤{self.cfg.ortho_max_rank})")
        if self.cfg.der_alpha > 0:
            modules.append(f"DER++ (α={self.cfg.der_alpha})")
        if self.cfg.use_wa:
            modules.append("Weight Aligning")
        print(f"Enabled modules: {' + '.join(modules)}")

        for task_id in range(start_task_id, T):
            new_classes = task_classes[task_id]
            print(f"\n{'='*60}")
            print(f"Task {task_id}: Learning classes {new_classes}")
            print(f"{'='*60}")

            num_old_classes = len(self.seen_classes)

            for cls_id in new_classes:
                if cls_id not in self.label_map:
                    self.label_map[cls_id] = len(self.label_map)

            if self.model is None and len(self.seen_classes) == 0:
                self.model = CILModel(
                    backbone_name=self.cfg.backbone,
                    pretrained=self.cfg.pretrained,
                    feature_dim=self.cfg.feature_dim,
                    adapter_bottleneck=self.cfg.adapter_bottleneck,
                    init_classes=len(new_classes),
                    use_moe=self.cfg.use_moe,
                    num_experts=self.cfg.num_experts,
                    top_k=self.cfg.top_k,
                ).to(self.device)

                total_params = sum(p.numel() for p in self.model.parameters())
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"  Total params: {total_params:,} | Trainable: {trainable:,} ({100*trainable/total_params:.1f}%)")
            else:
                if self.model is None:
                    raise RuntimeError("Model state is missing before incremental expansion.")
                self.old_model = self.model.freeze_copy()
                self.model.expand_classes(
                    len(new_classes),
                    scale_matched_head_init=self.cfg.scale_matched_head_init,
                )
                if self.cfg.use_moe and self.cfg.add_expert_per_task:
                    self.model.add_moe_experts(num_new=1)
                    print(f"  Added 1 new expert per MoE layer")

            exemplars = self._get_all_exemplars() if task_id > 0 else None
            train_loader, _ = get_task_dataloaders(
                train_dataset, test_dataset, new_classes,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                exemplar_data=exemplars,
                label_map=self.label_map,
                online_exemplar_augmentation=self.cfg.online_exemplar_augmentation,
                oversample_exemplars=self.cfg.oversample_exemplars,
            )

            train_stats = self._train_one_task(train_loader, task_id, num_old_classes)
            peak_mem = max(train_stats["epoch_peak_memory_mb"]) if train_stats["epoch_peak_memory_mb"] else 0.0
            avg_epoch_time = (
                float(np.mean(train_stats["epoch_time_sec"]))
                if train_stats["epoch_time_sec"] else 0.0
            )
            avg_latency = (
                float(np.mean(train_stats["epoch_avg_batch_latency_ms"]))
                if train_stats["epoch_avg_batch_latency_ms"] else 0.0
            )
            self.task_max_memory_mb.append(float(peak_mem))
            self.task_avg_epoch_time_sec.append(avg_epoch_time)
            self.task_avg_batch_latency_ms.append(avg_latency)

            wa_gamma = 1.0
            if self.cfg.use_wa and num_old_classes > 0:
                wa_gamma = self._wa_compute_gamma(num_old_classes)

            self.seen_classes.extend(new_classes)
            self._update_exemplars(train_dataset, new_classes)

            if wa_gamma != 1.0:
                self._wa_apply(num_old_classes, wa_gamma)

            print("  Evaluating on all seen tasks...")
            for past_id in range(task_id + 1):
                acc = self._evaluate_task(test_dataset, task_classes[past_id])
                self.acc_matrix[task_id][past_id] = acc
                print(f"    Task {past_id} accuracy: {acc:.4f}")

            if task_id == 0:
                old_task_acc = float("nan")
            else:
                old_task_acc = float(np.mean(self.acc_matrix[task_id][:task_id]))
            self.task_old_task_accuracy.append(old_task_acc)
            if np.isfinite(old_task_acc):
                print(f"  Old-task accuracy (mean): {old_task_acc * 100:.2f}%")
            else:
                print("  Old-task accuracy (mean): N/A (base task)")

            if self.cfg.use_energy_ood and task_id < T - 1:
                unseen_ids = []
                for future_task in task_classes[task_id + 1:]:
                    unseen_ids.extend(future_task)
                if unseen_ids:
                    print("  Evaluating OOD detection (Energy)...")
                    ood_results = self._evaluate_ood(
                        test_dataset,
                        seen_class_ids=list(self.seen_classes),
                        unseen_class_ids=unseen_ids,
                        task_id=task_id,
                    )
                    self.ood_metrics_log.append(ood_results)
                    self.writer.add_scalar("OOD/AUROC", ood_results["auroc"], task_id)
                    self.writer.add_scalar("OOD/FPR_at_95TPR", ood_results["fpr_at_95tpr"], task_id)
                    print(f"    AUROC: {ood_results['auroc']:.4f} | "
                          f"FPR@95TPR: {ood_results['fpr_at_95tpr']:.4f} | "
                          f"Threshold: {ood_results['threshold']:.4f}")
                    print(f"    ID energy (mean): {ood_results['id_mean_energy']:.4f} | "
                          f"OOD energy (mean): {ood_results['ood_mean_energy']:.4f}")

            print_metrics(self.acc_matrix, task_id)

            current_aa = float(np.mean(self.acc_matrix[task_id, :task_id + 1]))
            for past_id in range(task_id + 1):
                self.writer.add_scalar(
                    f"Per_Task_Accuracy/Task_{past_id}",
                    self.acc_matrix[task_id][past_id],
                    task_id,
                )
            self.writer.add_scalar("CIL/Average_Accuracy", current_aa, task_id)
            if task_id > 0:
                forgetting_vals = [
                    max(self.acc_matrix[:task_id+1, j]) - self.acc_matrix[task_id, j]
                    for j in range(task_id)
                ]
                af = float(np.mean(forgetting_vals))
                self.writer.add_scalar("CIL/Average_Forgetting", af, task_id)
            self.writer.add_scalar("Resource/Peak_GPU_Memory_MB", peak_mem, task_id)
            self.writer.add_scalar("Resource/Task_Wall_Clock_Time_s",
                                   sum(train_stats["epoch_time_sec"]), task_id)
            self.writer.flush()
            is_new_best = current_aa > self.best_aa
            if is_new_best:
                self.best_aa = current_aa
                self.best_task_id = task_id
                print(f"  New best AA: {self.best_aa:.4f} at task {task_id}")

            if self.cfg.auto_checkpoint:
                self._save_checkpoint(task_id, task_classes)
            if self.cfg.save_best and is_new_best:
                self._save_checkpoint(
                    task_id,
                    task_classes,
                    checkpoint_name="best.pt",
                    update_latest=False,
                    is_best=True,
                )

            if wa_gamma != 1.0:
                self._wa_revert(num_old_classes, wa_gamma)

        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "acc_matrix.npy", self.acc_matrix)
        self._save_runtime_metrics(output_dir)
        print(f"\nAccuracy matrix saved to {output_dir / 'acc_matrix.npy'}")

        self.writer.close()

        return self.acc_matrix

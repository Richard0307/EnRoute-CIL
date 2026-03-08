
import os
import json
import hydra
import logging
from omegaconf import DictConfig

from tqdm import tqdm

import torch
import statistics
from torch.utils.data import DataLoader
from continuum.metrics import Logger
import numpy as np

from continual_clip import utils
from continual_clip.models import load_model
from continual_clip.datasets import build_cl_scenarios


def compute_energy_scores(logits, temperature=1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def evaluate_ood_scores(id_scores, ood_scores):
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return {"auroc": None, "fpr_at_95tpr": None}

    tp = 0
    fp = 0
    tpr_list = []
    fpr_list = []
    for lab in sorted_labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    tpr_arr = np.array(tpr_list, dtype=float)
    fpr_arr = np.array(fpr_list, dtype=float)
    idx_95 = np.searchsorted(tpr_arr, 0.95)
    idx_95 = min(idx_95, len(fpr_arr) - 1)
    return {
        "auroc": float(np.trapz(tpr_arr, fpr_arr)),
        "fpr_at_95tpr": float(fpr_arr[idx_95]),
    }


@torch.no_grad()
def collect_energy_scores(model, dataloader):
    scores = []
    for inputs, _, _ in tqdm(dataloader):
        inputs = inputs.to(model.device)
        logits = model.forward_logits(inputs)
        scores.append(compute_energy_scores(logits).cpu().numpy())
    if not scores:
        return np.zeros(0, dtype=float)
    return np.concatenate(scores, axis=0)


@hydra.main(config_path=None, config_name=None, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:

    cfg.workdir = utils.get_workdir(path=os.getcwd())
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)

    utils.save_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.seed_all(int(getattr(cfg, "seed", 42)))
    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    model  = load_model(cfg, device)
    model_stats = {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    model_stats["trainable_ratio"] = float(
        model_stats["trainable_params"] / max(model_stats["total_params"], 1)
    )
    with open("model_stats.json", "w", encoding="utf-8") as handle:
        json.dump(model_stats, handle, indent=2)

    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.transforms
    )
    print(eval_dataset, eval_dataset)
    # print('eval_classname', classes_names)
    train_dataset, train_classes_names = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms
    )
    # print('train_classes_names', train_classes_names)
    model.classes_names = classes_names

    with open(cfg.log_path, 'w+') as f: 
        pass

    acc_list = []
    metric_logger = Logger(list_subsets=["test"])
    ood_history = []

    # test
    for task_id, _ in enumerate(eval_dataset):
        # breakpoint()
        logging.info(f"Evaluation for task {task_id} has started.")
        # breakpoint()
        model.adaptation(task_id, cfg, train_dataset, train_classes_names)  # task id 已经传入model

        eval_loader = DataLoader(
            eval_dataset[:task_id + 1],
            batch_size=cfg.batch_size,
            num_workers=int(getattr(cfg, "workers", 8)),
        )
        # breakpoint()
        for inputs, targets, task_ids in tqdm(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, task_ids)
            metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")

        acc_list.append(100 * metric_logger.accuracy)
        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'acc': round(100 * metric_logger.accuracy, 2),
                'avg_acc': round(100 * metric_logger.average_incremental_accuracy, 2),
                'forgetting': round(100 * metric_logger.forgetting, 6),
                'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
                'bwt': round(100 * metric_logger.backward_transfer, 2),
                'fwt': round(100 * metric_logger.forward_transfer, 2),
            }) + '\n')
            metric_logger.end_task()
        if task_id < len(eval_dataset) - 1:
            id_loader = DataLoader(
                eval_dataset[:task_id + 1],
                batch_size=cfg.batch_size,
                num_workers=int(getattr(cfg, "workers", 8)),
            )
            ood_loader = DataLoader(
                eval_dataset[task_id + 1:],
                batch_size=cfg.batch_size,
                num_workers=int(getattr(cfg, "workers", 8)),
            )
            id_scores = collect_energy_scores(model, id_loader)
            ood_scores = collect_energy_scores(model, ood_loader)
            ood_metrics = evaluate_ood_scores(id_scores, ood_scores)
            ood_metrics.update({
                "task": task_id,
                "id_mean_energy": float(np.mean(id_scores)),
                "ood_mean_energy": float(np.mean(ood_scores)),
            })
            ood_history.append(ood_metrics)
        # assert 1 == 2
    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'last': round(acc_list[-1], 2), 
            'avg': round(statistics.mean(acc_list), 2)
        }) + '\n')
    with open("ood_metrics.json", "w", encoding="utf-8") as handle:
        json.dump({
            "per_task": ood_history,
            "final": ood_history[-1] if ood_history else None,
        }, handle, indent=2)

        



if __name__ == "__main__":
    continual_clip()

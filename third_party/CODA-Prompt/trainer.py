import json
import os
import sys
import argparse
import torch
import numpy as np
import random
import copy
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import learners
import yaml
from PIL import Image


class ArchiveViewDataset(Dataset):
    """Lightweight view over archived task samples."""

    def __init__(self, data, targets, transform, class_mapping):
        self.data = np.asarray(data)
        self.targets = np.asarray(targets)
        self.transform = transform
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_mapping[int(self.targets[index])], -1


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

class Trainer:

    def __init__(self, args, seed, metric_keys, save_keys, repeat_idx=0):

        # process inputs
        self.seed = seed
        self.repeat_idx = repeat_idx
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        
        # model load directory
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'StateFarm':
            Dataset = dataloaders.iSTATE_FARM
            num_classes = 10
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.dataset == 'ImageNet_R':
            Dataset = dataloaders.iIMAGENET_R
            num_classes = 200
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.dataset == 'DomainNet':
            Dataset = dataloaders.iDOMAIN_NET
            num_classes = 345
            self.dataset_size = [224,224,3]
            self.top_k = 1
        else:
            raise ValueError('Dataset not implemented!')

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if getattr(args, 'class_order_path', ''):
            with open(args.class_order_path, 'r', encoding='utf-8') as handle:
                if args.class_order_path.endswith('.json'):
                    payload = json.load(handle)
                    class_order = payload['class_order'] if isinstance(payload, dict) else payload
                else:
                    payload = yaml.load(handle, Loader=yaml.Loader)
                    class_order = payload['class_order'] if isinstance(payload, dict) else payload
        elif self.seed > 0 and args.rand_split:
            print('=============================================')
            print('Shuffling....')
            print('pre-shuffle:' + str(class_order))
            random.seed(self.seed)
            random.shuffle(class_order)
            print('post-shuffle:' + str(class_order))
            print('=============================================')
        self.tasks = []
        self.tasks_logits = []
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p+inc])
            self.tasks_logits.append(class_order_logits[p:p+inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i+1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        k = 1 # number of transforms per image
        if args.model_name.startswith('vit'):
            resize_imnet = True
        else:
            resize_imnet = False
        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=resize_imnet)
        test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, resize_imnet=resize_imnet)
        self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                            download_flag=True, transform=train_transform, 
                            seed=self.seed, rand_split=args.rand_split, validation=args.validation)
        self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                download_flag=False, transform=test_transform, 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation)

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                        'lr': args.lr,
                        'debug_mode': args.debug_mode == 1,
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay,
                        'schedule': args.schedule,
                        'schedule_type': args.schedule_type,
                        'model_type': args.model_type,
                        'model_name': args.model_name,
                        'optimizer': args.optimizer,
                        'gpuid': args.gpuid,
                        'memory': args.memory,
                        'temp': args.temp,
                        'out_dim': num_classes,
                        'overwrite': args.overwrite == 1,
                        'DW': args.DW,
                        'batch_size': args.batch_size,
                        'upper_bound_flag': args.upper_bound_flag,
                        'tasks': self.tasks_logits,
                        'top_k': self.top_k,
                        'prompt_param':[self.num_tasks,args.prompt_param]
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
        trainable_params = 0
        if hasattr(self.learner, 'optimizer'):
            seen = set()
            for group in self.learner.optimizer.param_groups:
                for param in group['params']:
                    if id(param) in seen:
                        continue
                    seen.add(id(param))
                    trainable_params += param.numel()
        model_stats = {
            'total_params': int(self.learner.count_parameter()),
            'trainable_params': int(trainable_params),
            'trainable_ratio': float(trainable_params / max(self.learner.count_parameter(), 1)),
        }
        with open(os.path.join(self.log_dir, 'model_stats.json'), 'w', encoding='utf-8') as handle:
            json.dump(model_stats, handle, indent=2)

    def task_eval(self, t_index, local=False, task='acc'):

        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        
        # eval
        self.test_dataset.load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        if local:
            return self.learner.validation(test_loader, task_in = self.tasks_logits[t_index], task_metric=task)
        else:
            return self.learner.validation(test_loader, task_metric=task)

    def _build_archive_loader(self, task_indices):
        data = np.concatenate([self.test_dataset.archive[idx][0] for idx in task_indices], axis=0)
        targets = np.concatenate([self.test_dataset.archive[idx][1] for idx in task_indices], axis=0)
        dataset = ArchiveViewDataset(
            data=data,
            targets=targets,
            transform=self.test_dataset.transform,
            class_mapping=self.test_dataset.class_mapping,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.workers,
        )

    @torch.no_grad()
    def _collect_energy_scores(self, dataloader):
        model = self.learner.model
        orig_mode = model.training
        model.eval()
        all_scores = []
        for inputs, _, _ in dataloader:
            if self.learner.gpu:
                inputs = inputs.cuda()
            logits = model.forward(inputs)[:, :self.learner.valid_out_dim]
            scores = compute_energy_scores(logits)
            all_scores.append(scores.cpu().numpy())
        model.train(orig_mode)
        if not all_scores:
            return np.zeros(0, dtype=float)
        return np.concatenate(all_scores, axis=0)

    def _load_task_model_for_eval(self, task_index):
        if task_index > 0:
            try:
                if self.learner.model.module.prompt is not None:
                    self.learner.model.module.prompt.process_task_count()
            except:
                if self.learner.model.prompt is not None:
                    self.learner.model.prompt.process_task_count()

        model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.repeat_idx+1)+'/task-'+self.task_names[task_index]+'/'
        self.learner.task_count = task_index
        self.learner.add_valid_output_dim(len(self.tasks_logits[task_index]))
        self.learner.pre_steps()
        self.learner.load_model(model_save_dir)

        try:
            self.learner.model.module.task_id = task_index
        except:
            self.learner.model.task_id = task_index

    def evaluate_ood(self):
        if self.max_task <= 1:
            return None

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
        ood_history = []

        for task_index in range(self.max_task - 1):
            self._load_task_model_for_eval(task_index)
            id_loader = self._build_archive_loader(range(task_index + 1))
            ood_loader = self._build_archive_loader(range(task_index + 1, self.max_task))
            id_scores = self._collect_energy_scores(id_loader)
            ood_scores = self._collect_energy_scores(ood_loader)
            metrics = evaluate_ood_scores(id_scores, ood_scores)
            metrics.update({
                "task": task_index,
                "id_mean_energy": float(np.mean(id_scores)),
                "ood_mean_energy": float(np.mean(ood_scores)),
            })
            ood_history.append(metrics)

        payload = {
            "per_task": ood_history,
            "final": ood_history[-1] if ood_history else None,
        }
        with open(os.path.join(self.log_dir, 'ood_metrics.json'), 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        return payload

    def train(self, avg_metrics):
    
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: temp_table[mkey] = []
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # for each task
        for i in range(self.max_task):

            # save current task index
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task
            task = self.tasks_logits[i]
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            self.train_dataset.append_coreset(only=False)

            # load dataloader
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # learn
            self.test_dataset.load_dataset(i, train=False)
            test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.repeat_idx+1)+'/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir, test_loader)

            # save model
            self.learner.save_model(model_save_dir)
            
            # evaluate acc
            acc_table = []
            acc_table_ssl = []
            self.reset_cluster_labels = True
            for j in range(i+1):
                acc_table.append(self.task_eval(j))
            temp_table['acc'].append(np.mean(np.asarray(acc_table)))

            # save temporary acc results
            for mkey in ['acc']:
                save_file = temp_dir + mkey + '.csv'
                np.savetxt(save_file, np.asarray(temp_table[mkey]), delimiter=",", fmt='%.2f')  

            if avg_train_time is not None: avg_metrics['time']['global'][i] = avg_train_time

        return avg_metrics 
    
    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.repeat_idx] = acc_table[val_name][train_name]
                avg_acc_pt_local[j,i,self.repeat_idx] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        # Gather the final avg accuracy
        avg_acc_all[:,self.repeat_idx] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}

    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}
            
        for i in range(self.max_task):

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # load model
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.repeat_idx+1)+'/task-'+self.task_names[i]+'/'
            self.learner.task_count = i 
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # evaluate acc
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j)
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table_local['acc'][val_name][self.task_names[i]] = self.task_eval(j, local=True)

        # summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'])

        return avg_metrics

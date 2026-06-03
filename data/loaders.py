"""Replay-sample selection and DataLoader construction.

* :func:`set_replay_samples`  - importance-weighted reservoir selection of the
  replay buffer indices (shared by both methods).
* :func:`set_train_loader`    - continual-training loader (current task classes
  + replay buffer), with the per-task BatchSchedulerSampler.
* :func:`set_eval_loaders`    - linear-probe train/val loaders over all classes
  seen so far.

The per-dataset logic lives entirely in :mod:`data.datasets`; everything here
is dataset-agnostic.
"""
import copy
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.dataset import ConcatDataset
from torchvision import datasets as tvdatasets
import torchvision.transforms as transforms

from data.datasets import (build_dataset, get_targets, build_train_transform,
                           build_eval_transform, WeightedSubset)
from utils.util import BatchSchedulerSampler


def _sample_size(float_size):
    """Round a fractional per-class budget up or down probabilistically."""
    p = float_size - math.floor(float_size)
    return math.ceil(float_size) if random.random() < p else math.floor(float_size)


def set_replay_samples(opt, prev_indices=None, prev_importance_weight=None, prev_score=None):
    """Select replay-buffer indices and their normalised importance weights."""
    # Only the labels are needed here, so a bare ToTensor transform suffices.
    val_dataset = build_dataset(opt, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    val_targets = get_targets(val_dataset)

    prev_indices_len = 0
    if prev_indices is None:
        prev_indices, prev_importance_weight = [], []
        observed_classes = list(range(0, opt.target_task * opt.cls_per_task))
    else:
        shrink_size = (opt.target_task - 1) * opt.mem_size / opt.target_task
        if len(prev_indices) > 0:
            unique_cls = np.unique(val_targets[prev_indices])
            _prev_indices, prev_indices_len = prev_indices, len(prev_indices)
            prev_indices, prev_importance_weight = [], []
            prev_score_t = torch.tensor(prev_score[:prev_indices_len])

            for c in unique_cls:
                mask = val_targets[_prev_indices] == c
                size_for_c = _sample_size(shrink_size / len(unique_cls))
                cls_scores = prev_score_t[mask]
                store_index = torch.multinomial(cls_scores, min(len(cls_scores), size_for_c),
                                                replacement=False)
                prev_indices += torch.tensor(_prev_indices)[mask][store_index].tolist()
                prev_importance_weight += (cls_scores / cls_scores.sum())[store_index].tolist()

            print(np.unique(val_targets[prev_indices], return_counts=True))
        observed_classes = list(range(max(opt.target_task - 1, 0) * opt.cls_per_task,
                                      opt.target_task * opt.cls_per_task))

    if len(observed_classes) == 0:
        return prev_indices, prev_importance_weight, val_targets

    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()

    val_unique_cls = np.unique(val_targets[observed_indices])
    obs_score = torch.tensor(prev_score[prev_indices_len:])

    selected_indices, selected_weight = [], []
    for c_idx, c in enumerate(val_unique_cls):
        remaining = opt.mem_size - len(prev_indices) - len(selected_indices)
        size_for_c = _sample_size(remaining / (len(val_unique_cls) - c_idx))
        mask = val_targets[observed_indices] == c
        cls_scores = obs_score[mask]
        store_index = torch.multinomial(cls_scores, size_for_c, replacement=False)
        selected_indices += torch.tensor(observed_indices)[mask][store_index].tolist()
        selected_weight += (cls_scores / cls_scores.sum())[store_index].tolist()

    print(np.unique(val_targets[selected_indices], return_counts=True))
    print(selected_weight)
    return prev_indices + selected_indices, prev_importance_weight + selected_weight, val_targets


def _loader_kwargs(opt):
    kwargs = dict(num_workers=opt.num_workers, pin_memory=True,
                  persistent_workers=opt.num_workers > 0)
    if opt.num_workers > 0:
        kwargs['prefetch_factor'] = 4
    return kwargs


def set_train_loader(opt, replay_indices, importance_weight, training=True):
    """Build the continual-training loader for the current task + replay buffer.

    Returns ``(train_loader, subset_indices, replay_sample_num)`` where
    ``subset_indices`` are the dataset indices in the loader (current + replay)
    and ``replay_sample_num`` the per-class counts (ascending class order).
    """
    train_transform = build_train_transform(opt)

    if opt.dataset == 'path':
        train_dataset = tvdatasets.ImageFolder(root=opt.data_folder, transform=train_transform)
        return DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                          **_loader_kwargs(opt)), None, None

    base = build_dataset(opt, train=True, transform=train_transform)
    targets = get_targets(base)

    cur_indices, cur_weight = [], []
    for tc in range(opt.target_task * opt.cls_per_task, (opt.target_task + 1) * opt.cls_per_task):
        idx = np.where(targets == tc)[0]
        cur_indices += idx.tolist()
        cur_weight += list(np.ones(len(idx)) / len(idx))

    use_concat = len(replay_indices) > 0 and training
    if use_concat:
        prev_dataset = WeightedSubset(base, copy.deepcopy(replay_indices), copy.deepcopy(importance_weight))
        cur_dataset = WeightedSubset(base, copy.deepcopy(cur_indices), copy.deepcopy(cur_weight))
        dataset_len_list = [len(prev_dataset), len(cur_dataset)]
        train_dataset = ConcatDataset([prev_dataset, cur_dataset])
    else:
        train_dataset = WeightedSubset(base, cur_indices + list(replay_indices),
                                       cur_weight + list(importance_weight))

    subset_indices = cur_indices + list(replay_indices)
    uk, uc = np.unique(targets[subset_indices], return_counts=True)
    replay_sample_num = uc[np.argsort(uk)]

    if use_concat:
        n_prev = int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list)))
        batch_size_list = [n_prev, opt.batch_size - n_prev]
        print('train_batch_size', batch_size_list)
        train_sampler = BatchSchedulerSampler(dataset=train_dataset, batch_size=batch_size_list)
    else:
        train_sampler = None

    print('Dataset size: {}'.format(len(subset_indices)))
    if training:
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size,
                                  shuffle=(train_sampler is None), sampler=train_sampler,
                                  **_loader_kwargs(opt))
    else:
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False,
                                  **_loader_kwargs(opt))
        print('no separate sampler')

    return train_loader, subset_indices, replay_sample_num


def set_eval_loaders(opt, replay_indices):
    """Linear-probe loaders: weighted-balanced train set (current task classes +
    replay) and a validation set over all classes seen so far."""
    train_transform = build_train_transform(opt)
    val_transform = build_eval_transform(opt)
    target_classes = list(range(0, (opt.target_task + 1) * opt.cls_per_task))

    base_train = build_dataset(opt, train=True, transform=train_transform)
    train_targets = get_targets(base_train)
    subset_indices = []
    for tc in range(opt.target_task * opt.cls_per_task, (opt.target_task + 1) * opt.cls_per_task):
        subset_indices += np.where(train_targets == tc)[0].tolist()
    subset_indices += list(replay_indices)

    ut, uc = np.unique(train_targets[subset_indices], return_counts=True)
    print(ut)
    print(uc)
    weights = np.zeros(len(subset_indices))
    sel_targets = train_targets[subset_indices]
    for t, c in zip(ut, uc):
        weights[sel_targets == t] = 1. / c
    train_dataset = Subset(base_train, subset_indices)

    base_val = build_dataset(opt, train=False, transform=val_transform)
    val_targets = get_targets(base_val)
    val_indices = []
    for tc in target_classes:
        val_indices += np.where(val_targets == tc)[0].tolist()
    val_dataset = Subset(base_val, val_indices)

    train_sampler = WeightedRandomSampler(torch.Tensor(weights), len(weights))
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False,
                              num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,
                            num_workers=8, pin_memory=True)
    return train_loader, val_loader, uc
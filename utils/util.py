"""Shared training utilities: meters, learning-rate schedules, optimizer
construction, checkpoint IO, the per-task batch sampler and misc helpers."""
import math
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import RandomSampler


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


class TwoCropTransform:
    """Create two crops of the same image."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Top-k accuracy for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size)
                for k in topk]


# --------------------------------------------------------------------------- #
# Learning-rate schedules                                                      #
# --------------------------------------------------------------------------- #
def _proto_lr_list(optimizer, lr_enc, lr_prot):
    """Build the per-group LR list for the contrastive optimizer.

    Group layout is ``[encoder, head, prototypes]`` (3 groups) or
    ``[encoder, head, prototypes, patch_head, patch_prototypes]`` (5 groups,
    HCDR).  A single-group (legacy) optimizer just gets ``lr_enc``.
    """
    n = len(optimizer.param_groups)
    if n == 5:
        return [lr_enc, lr_enc, lr_prot, lr_enc, lr_prot]
    if n == 3:
        return [lr_enc, lr_enc, lr_prot]
    return [lr_enc] * n


def adjust_learning_rate(args, optimizer, epoch):
    lr_enc, lr_prot = args.learning_rate, args.learning_rate_prototypes
    if args.cosine:
        for lr, name in ((lr_enc, 'enc'), (lr_prot, 'prot')):
            eta_min = lr * (args.lr_decay_rate ** 3)
            val = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
            if name == 'enc':
                lr_enc = val
            else:
                lr_prot = val
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr_enc *= args.lr_decay_rate ** steps
            lr_prot *= args.lr_decay_rate ** steps
    for lr, group in zip(_proto_lr_list(optimizer, lr_enc, lr_prot), optimizer.param_groups):
        group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
        lr_enc = args.warmup_from_enc + p * (args.warmup_to_enc - args.warmup_from_enc)
        lr_prot = args.warmup_from_prot + p * (args.warmup_to_prot - args.warmup_from_prot)
        for lr, group in zip(_proto_lr_list(optimizer, lr_enc, lr_prot), optimizer.param_groups):
            group['lr'] = lr


def adjust_learning_rate_linear(args, optimizer, epoch):
    """Single-LR cosine / step schedule used by the linear-probe evaluation."""
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr *= args.lr_decay_rate ** steps
    for group in optimizer.param_groups:
        group['lr'] = lr


def warmup_learning_rate_linear(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)
        for group in optimizer.param_groups:
            group['lr'] = lr


# --------------------------------------------------------------------------- #
# Optimizer / checkpoint IO                                                    #
# --------------------------------------------------------------------------- #
def _has_prototype_params(model):
    return any(k.endswith('prototypes.weight') for k in model.state_dict().keys())


def set_optimizer(opt, model):
    """SGD with per-group learning rates.

    Encoder + head use ``opt.learning_rate``; (patch-)prototypes use
    ``opt.learning_rate_prototypes``.  Build this *before* ``torch.compile`` so
    the per-group rates survive (a compiled module prefixes its state-dict keys
    with ``_orig_mod.``).  ``opt.legacy_optimizer`` falls back to a single group.
    """
    fused = torch.cuda.is_available()
    legacy = getattr(opt, 'legacy_optimizer', False)
    if legacy or not _has_prototype_params(model):
        return optim.SGD(model.parameters(), lr=opt.learning_rate,
                         momentum=opt.momentum, weight_decay=opt.weight_decay, fused=fused)

    groups = [
        {'params': model.encoder.parameters()},
        {'params': model.head.parameters()},
        {'params': model.prototypes.parameters(), 'lr': opt.learning_rate_prototypes},
    ]
    if getattr(model, 'patch_head', None) is not None:
        groups.append({'params': model.patch_head.parameters()})
        groups.append({'params': model.patch_prototypes.parameters(),
                       'lr': opt.learning_rate_prototypes})
    return optim.SGD(groups, lr=opt.learning_rate, momentum=opt.momentum,
                     weight_decay=opt.weight_decay, fused=fused)


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...' + save_file)
    torch.save({'opt': opt, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch}, save_file)


def load_model(model, optimizer, save_file):
    print('==> Loading...' + save_file)
    loaded = torch.load(save_file, weights_only=False)
    model.load_state_dict(loaded['model'])
    optimizer.load_state_dict(loaded['optimizer'])
    return model, optimizer


# --------------------------------------------------------------------------- #
# Per-task batch sampler                                                       #
# --------------------------------------------------------------------------- #
class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """Iterate over the (previous, current) sub-datasets of a ConcatDataset,
    drawing a fixed number of samples from each per mini-batch."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size  # list: [n_prev, n_cur]
        self.number_of_datasets = len(dataset.datasets)
        self.dataset_len = sum(len(d) for d in dataset.datasets)

    def __len__(self):
        return self.dataset_len

    def __iter__(self):
        sampler_iterators = [iter(RandomSampler(d)) for d in self.dataset.datasets]
        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = sum(self.batch_size)

        final = []
        for _ in range(0, self.dataset_len, step):
            for i in range(self.number_of_datasets):
                for _ in range(self.batch_size[i]):
                    try:
                        final.append(next(sampler_iterators[i]) + push_index_val[i])
                    except StopIteration:
                        break
        return iter(final)
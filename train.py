"""Continual contrastive training (CCLIS baseline and HCDR extension).

Run with ``--method cclis`` or ``--method hcdr``.  HCDR adds a patch-level
InfoNCE term and a hierarchical-PRD distillation term on top of the shared
global objective; everything else (replay selection, importance-sampled score
computation, prototype distillation) is common to both.
"""
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import parse_train_option
from data.loaders import set_replay_samples, set_train_loader
from losses.supcon import ISSupConLoss
from losses.hcdr import HCDRLoss, PatchBuffer, update_patch_buffer
from networks.resnet import SupConResNet
from utils.util import (AverageMeter, adjust_learning_rate, warmup_learning_rate,
                        set_optimizer, save_model, load_model, set_seed)


def _uncompiled(model):
    return getattr(model, '_orig_mod', model)


def set_model(opt):
    """Build the (uncompiled) model and its criterion, moved to CUDA if available."""
    model = SupConResNet(name=opt.model, opt=opt)
    criterion = HCDRLoss(temperature=opt.temp, opt=opt) if opt.use_hcdr \
        else ISSupConLoss(temperature=opt.temp, opt=opt)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    return model, criterion


def _normalize_prototypes(model, use_hcdr):
    with torch.no_grad():
        w = nn.functional.normalize(model.prototypes.weight.data.clone(), dim=1, p=2)
        model.prototypes.weight.copy_(w)
        if use_hcdr and model.patch_prototypes is not None:
            wp = nn.functional.normalize(model.patch_prototypes.weight.data.clone(), dim=1, p=2)
            model.patch_prototypes.weight.copy_(wp)


def _ird_distill(features, model2, images, target_labels, opt):
    labels_mask = None  # only needed for IRD_type == 'prev'
    feats1 = features
    sim1 = torch.div(torch.matmul(feats1, feats1.T), opt.current_temp)
    logits_mask = torch.scatter(
        torch.ones_like(sim1), 1,
        torch.arange(sim1.size(0), device=sim1.device).view(-1, 1), 0)
    sim1 = sim1 - torch.max(sim1 * logits_mask, dim=1, keepdim=True)[0].detach()
    row = sim1.size(0)
    e1 = torch.exp(sim1[logits_mask.bool()].view(row, -1))
    logits1 = e1 / e1.sum(dim=1, keepdim=True)
    with torch.no_grad():
        feats2, _ = model2(images)
        sim2 = torch.div(torch.matmul(feats2, feats2.T), opt.past_temp)
        sim2 = sim2 - torch.max(sim2 * logits_mask, dim=1, keepdim=True)[0].detach()
        e2 = torch.exp(sim2[logits_mask.bool()].view(row, -1))
        logits2 = e2 / e2.sum(dim=1, keepdim=True)
    return (-logits2 * torch.log(logits1)).sum(1).mean()


def train(train_loader, model, model2, criterion, optimizer, epoch,
          subset_sample_num, score_mask, opt, scaler, amp_dtype):
    """One training epoch (CCLIS + optional HCDR terms)."""
    model.train()
    use_hcdr = opt.use_hcdr
    distill_type = opt.distill_type

    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, distill = AverageMeter(), AverageMeter()
    end = time.time()

    for idx, (images, labels, importance_weight, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            images = images.to('cuda', memory_format=torch.channels_last, non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        _normalize_prototypes(model, use_hcdr)
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        target_labels = list(range(opt.target_task * opt.cls_per_task,
                                   (opt.target_task + 1) * opt.cls_per_task))

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
            if use_hcdr:
                features, output, _, patch_feat, patch_proto_logits = model(images, return_spatial=True)
            else:
                features, output = model(images)
            device = features.device

            loss = criterion(output, features, labels, importance_weight, index,
                             target_labels=target_labels, sample_num=subset_sample_num,
                             score_mask=score_mask)

            if use_hcdr:
                loss = loss + opt.alpha_patch * criterion.patch_nce_forward(patch_proto_logits, labels)

            if opt.target_task > 0:
                if distill_type == 'IRD':
                    loss_distill = _ird_distill(features, model2, images, target_labels, opt)
                    loss = loss + opt.distill_power * loss_distill
                    distill.update(loss_distill.item(), bsz)
                elif distill_type == 'PRD':
                    prev_all_labels = torch.arange(target_labels[0])
                    prototypes_mask = torch.scatter(
                        torch.zeros(len(prev_all_labels), opt.n_cls).float(),
                        1, prev_all_labels.view(-1, 1), 1).to(device)

                    sim1 = torch.div(torch.matmul(prototypes_mask, output), opt.current_temp)
                    sim1 = sim1 - torch.max(sim1, dim=0, keepdim=True)[0].detach()
                    logits1 = torch.exp(sim1) / torch.exp(sim1).sum(dim=0, keepdim=True)

                    with torch.no_grad():
                        if use_hcdr:
                            _, sim2_prev, _, patch_feat_prev, _ = model2(images, return_spatial=True)
                        else:
                            _, sim2_prev = model2(images)
                        sim2 = torch.div(torch.matmul(prototypes_mask, sim2_prev), opt.past_temp)
                        sim2 = sim2 - torch.max(sim2, dim=0, keepdim=True)[0].detach()
                        logits2 = torch.exp(sim2) / torch.exp(sim2).sum(dim=0, keepdim=True)

                    loss_distill = (-logits2 * torch.log(logits1)).sum(0).mean()
                    loss = loss + opt.distill_power * loss_distill
                    distill.update(loss_distill.item(), bsz)

                    if use_hcdr:
                        loss_hprd = criterion.hprd_forward(
                            patch_feat_cur=patch_feat, patch_feat_prev=patch_feat_prev,
                            proto_weight_cur=model.prototypes.weight.data,
                            proto_weight_prev=model2.prototypes.weight.data,
                            tau_cur=opt.current_temp, tau_prev=opt.past_temp)
                        loss = loss + opt.lambda_hprd * loss_hprd
                else:
                    raise ValueError('distill type {} is not supported'.format(distill_type))

        losses.update(loss.item(), bsz)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0 or idx + 1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {bt.val:.3f} ({bt.avg:.3f})\tDT {dt.val:.3f} ({dt.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f} {distill.avg:.3f})'.format(
                      epoch, idx + 1, len(train_loader), bt=batch_time, dt=data_time,
                      loss=losses, distill=distill))
            sys.stdout.flush()

    return losses.avg, model2


def score_computing(val_loader, model, model2, criterion, subset_sample_num, score_mask, opt):
    """Importance-sampling score for every replay candidate (averaged over max_iter passes)."""
    model.eval()
    cur_task_n_cls = (opt.target_task + 1) * opt.cls_per_task
    len_val = sum(subset_sample_num)
    print('val_loader length', len_val)

    _score = torch.zeros(cur_task_n_cls, len_val)
    all_score_sum = torch.zeros(cur_task_n_cls, cur_task_n_cls)
    eye = torch.eye(cur_task_n_cls)

    for _ in range(opt.max_iter):
        index_list, score_list, label_list = [], [], []
        score_sum = torch.zeros(cur_task_n_cls, cur_task_n_cls)
        for images, labels, importance_weight, index in val_loader:
            index_list += index
            label_list += labels
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            with torch.no_grad():
                features, output = model(images)
                score_mat, batch_score_sum = criterion.score_calculate(
                    output, features, labels, importance_weight, index,
                    target_labels=list(range(opt.target_task * opt.cls_per_task,
                                              (opt.target_task + 1) * opt.cls_per_task)),
                    sample_num=subset_sample_num, score_mask=score_mask)
                score_list.append(score_mat)
                score_sum += batch_score_sum

        index_list = torch.tensor(index_list)
        label_list = torch.tensor(label_list).tolist()
        label_score_mask = torch.eq(torch.arange(cur_task_n_cls).view(-1, 1), torch.tensor(label_list))

        scores = torch.concat(score_list, dim=1).to('cpu')
        _score -= _score * label_score_mask
        _score += scores / scores.sum(dim=1, keepdim=True)
        all_score_sum += score_sum
        all_score_sum -= all_score_sum * eye

    _score /= opt.max_iter
    score = _score.cpu().sum(dim=0) / (_score.shape[0] - 1)
    return None, index_list, score, model2


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    opt = parse_train_option()
    set_seed(opt.seed)

    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)            # before compile -> per-group LRs survive
    if opt.compile and hasattr(torch, 'compile'):
        print('Compiling model for faster execution...')
        model = torch.compile(model)
    model = model.to(memory_format=torch.channels_last)

    # AMP: bf16 needs no loss scaling; fp16 falls back to GradScaler.
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
        else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16 and torch.cuda.is_available()))

    patch_buffer = PatchBuffer(top_k=opt.top_k_patches) if opt.use_hcdr else None

    replay_indices = importance_weight = score = score_mask = None
    if opt.resume_target_task is not None:
        load_file = os.path.join(opt.save_folder, 'last_{}_{}.pth'.format(
            opt.replay_policy, opt.resume_target_task))
        model, optimizer = load_model(model, optimizer, load_file)
        if opt.resume_target_task == 0:
            replay_indices, importance_weight = [], []
        else:
            replay_indices = np.load(os.path.join(opt.log_folder, 'replay_indices_{}_{}.npy'.format(
                opt.replay_policy, opt.resume_target_task))).tolist()
            importance_weight = np.load(os.path.join(opt.log_folder, 'importance_weight_{}_{}.npy'.format(
                opt.replay_policy, opt.resume_target_task))).tolist()
        score = np.load(os.path.join(opt.log_folder, 'score_{}_{}.npy'.format(
            opt.replay_policy, opt.resume_target_task))).tolist()

    original_epochs = opt.epochs
    if opt.end_task is not None:
        if opt.resume_target_task is not None:
            assert opt.end_task > opt.resume_target_task
        opt.end_task = min(opt.end_task + 1, opt.n_cls // opt.cls_per_task)
    else:
        opt.end_task = opt.n_cls // opt.cls_per_task

    start_task = 0 if opt.resume_target_task is None else opt.resume_target_task + 1
    for target_task in range(start_task, opt.end_task):
        opt.target_task = target_task
        model2 = copy.deepcopy(_uncompiled(model)).to(memory_format=torch.channels_last)
        model2.eval()

        print('Start Training current task {}'.format(target_task))
        replay_indices, importance_weight, val_targets = set_replay_samples(
            opt, prev_indices=replay_indices, prev_importance_weight=importance_weight, prev_score=score)

        if patch_buffer is not None and len(replay_indices) > 0:
            patch_buffer.prune(replay_indices)

        if target_task != 0:
            opt.replay_indices_0 = replay_indices[0]
        np.save(os.path.join(opt.log_folder, 'replay_indices_{}_{}.npy'.format(
            opt.replay_policy, target_task)), np.array(replay_indices))
        np.save(os.path.join(opt.log_folder, 'importance_weight_{}_{}.npy'.format(
            opt.replay_policy, target_task)), np.array(importance_weight))

        train_loader, subset_indices, subset_sample_num = set_train_loader(
            opt, replay_indices, importance_weight)
        np.save(os.path.join(opt.log_folder, 'subset_indices_{}_{}.npy'.format(
            opt.replay_policy, target_task)), np.array(subset_indices))

        opt.epochs = opt.start_epoch if (target_task == 0 and opt.start_epoch is not None) else original_epochs
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)
            t0 = time.time()
            _, model2 = train(train_loader, model, model2, criterion, optimizer, epoch,
                              subset_sample_num, score_mask, opt, scaler, amp_dtype)
            print('epoch {}, total time {:.2f}'.format(epoch, time.time() - t0))

        val_loader, _, _ = set_train_loader(opt, replay_indices, importance_weight, training=False)
        score_mask, index, _score, model2 = score_computing(
            val_loader, model, model2, criterion, subset_sample_num, score_mask, opt)

        observed_indices = []
        for tc in range(target_task * opt.cls_per_task, (target_task + 1) * opt.cls_per_task):
            observed_indices += np.where(val_targets == tc)[0].tolist()
        score_indices = replay_indices + observed_indices

        score_dict = dict(zip(index.tolist(), _score))
        score = torch.stack([score_dict[k] for k in score_indices])
        np.save(os.path.join(opt.log_folder, 'score_{}_{}.npy'.format(
            opt.replay_policy, target_task)), np.array(score.cpu()))

        if patch_buffer is not None:
            raw_dataset = getattr(val_loader.dataset, 'dataset', val_loader.dataset)
            update_patch_buffer(patch_buffer, model, score_indices, val_targets, raw_dataset, opt)
            print('PatchBuffer size: {}'.format(len(patch_buffer)))

        if target_task == opt.end_task - 1:
            save_file = os.path.join(opt.save_folder, 'last_{}_{}.pth'.format(
                opt.replay_policy, target_task))
            save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
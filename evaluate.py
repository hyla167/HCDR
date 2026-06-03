import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import parse_eval_option
from data.loaders import set_eval_loaders
from networks.resnet import SupConResNet, LinearClassifier
from utils.util import (AverageMeter, set_optimizer,
                        adjust_learning_rate_linear, warmup_learning_rate_linear)


def set_model(opt):
    model = SupConResNet(name=opt.model, opt=opt)
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    ckpt = torch.load(opt.ckpt, map_location='cpu', weights_only=False)
    # Strip wrappers that torch.compile / DataParallel add to the keys.
    state_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v
                  for k, v in ckpt['model'].items()}

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # strict=False: HCDR checkpoints carry patch_head / patch_prototypes keys
    # that the probe model does not define; the encoder/head/prototypes match.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print('Ignored {} checkpoint keys not used by the probe '
              '(e.g. patch heads).'.format(len(unexpected)))
    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """Train the linear classifier for one epoch (encoder frozen)."""
    model.eval()
    classifier.train()

    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()
    acc, cnt = 0.0, 0.0

    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        warmup_learning_rate_linear(opt, epoch, idx, len(train_loader), optimizer)

        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        losses.update(loss.item(), bsz)
        acc += (output.argmax(1) == labels).float().sum().item()
        cnt += bsz

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {bt.val:.3f} ({bt.avg:.3f})\t'
                  'DT {dt.val:.3f} ({dt.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1:.3f}'.format(
                      epoch, idx + 1, len(train_loader), bt=batch_time,
                      dt=data_time, loss=losses, top1=acc / cnt * 100.))
            sys.stdout.flush()

    return losses.avg, acc / cnt * 100.


def validate(val_loader, model, classifier, criterion, opt):
    """Validate, reporting Class-IL (overall) and Task-IL (within-block) acc."""
    model.eval()
    classifier.eval()

    batch_time, losses = AverageMeter(), AverageMeter()
    n_seen = (opt.target_task + 1) * opt.cls_per_task
    corr = [0.] * n_seen
    cnt = [0.] * n_seen
    correct_task = 0.0

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            output = classifier(model.encoder(images))
            loss = criterion(output, labels)
            losses.update(loss.item(), bsz)

            batch_time.update(time.time() - end)
            end = time.time()

            cls_list = np.unique(labels.cpu())
            correct_all = (output.argmax(1) == labels)

            for tc in cls_list:
                mask = labels == tc
                block = tc // opt.cls_per_task
                lo, hi = block * opt.cls_per_task, (block + 1) * opt.cls_per_task
                correct_task += (output[mask, lo:hi].argmax(1)
                                 == (tc % opt.cls_per_task)).float().sum()
            for c in cls_list:
                mask = labels == c
                corr[c] += correct_all[mask].float().sum().item()
                cnt[c] += mask.float().sum().item()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {bt.val:.3f} ({bt.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1:.3f} {task_il:.3f}'.format(
                          idx, len(val_loader), bt=batch_time, loss=losses,
                          top1=np.sum(corr) / np.sum(cnt) * 100.,
                          task_il=correct_task / np.sum(cnt) * 100.))

    class_il = np.sum(corr) / np.sum(cnt) * 100.
    task_il = correct_task / np.sum(cnt) * 100.
    print(' * Acc@1 {0:.3f} {1:.3f}'.format(class_il, task_il))
    return losses.avg, class_il, corr, cnt, task_il


def main():
    opt = parse_eval_option()

    replay_indices = (np.array([]) if opt.target_task == 0
                      else np.load(opt.logpt))
    print(len(replay_indices))

    best_acc_list, best_task_acc_list = [], []
    best_acc, best_task_acc = 0., 0.
    val_acc, task_acc = 0., 0.
    val_acc_stats = {}

    train_loader, val_loader, _ = set_eval_loaders(opt, replay_indices)
    model, classifier, criterion = set_model(opt)
    optimizer = set_optimizer(opt, classifier)
    print(optimizer.param_groups[0]['lr'])

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate_linear(opt, optimizer, epoch)

        t1 = time.time()
        _, acc = train(train_loader, model, classifier, criterion,
                       optimizer, epoch, opt)
        t2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f} {:.3f}'.format(
            epoch, t2 - t1, acc, optimizer.param_groups[0]['lr']))

        _, val_acc, val_corr, val_cnt, task_acc = validate(
            val_loader, model, classifier, criterion, opt)
        val_acc = np.sum(val_corr) / np.sum(val_cnt) * 100.
        best_acc = max(best_acc, val_acc)
        print('Task acc: {}'.format(task_acc))
        best_task_acc = max(best_task_acc, task_acc)

        val_acc_stats = {str(cls): cr / c * 100.
                         for cls, (cr, c) in enumerate(zip(val_corr, val_cnt))
                         if c > 0}

    best_acc_list.append(best_acc)
    best_task_acc_list.append(
        best_task_acc.cpu().item() if torch.is_tensor(best_task_acc) else best_task_acc)

    with open(os.path.join(opt.origin_ckpt,
                           'acc_buffer_{}.txt'.format(opt.target_task)), 'w') as f:
        out = 'best accuracy: {:.2f}\n'.format(best_acc)
        out += '{:.2f} {:.2f}'.format(val_acc, task_acc)
        print(out)
        out += '\n'
        for v in val_acc_stats.values():
            print(v)
            out += '{}\n'.format(v)
        f.write(out)

    save_file = os.path.join(
        opt.origin_ckpt, 'linear_{}.pth'.format(opt.target_task))
    print('==> Saving...' + save_file)
    torch.save({'opt': opt,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, save_file)

    print('Average accuracy for Class-IL: {}, std: {}'.format(
        np.mean(best_acc_list), np.std(best_acc_list, ddof=1)))
    print('Average accuracy for Task-IL: {}, std: {}'.format(
        np.mean(best_task_acc_list), np.std(best_task_acc_list, ddof=1)))


if __name__ == '__main__':
    main()
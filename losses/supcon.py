"""Importance-Sampled Supervised Contrastive loss (CCLIS).

This is the global, prototype-based contrastive objective shared by both the
CCLIS baseline and HCDR (HCDR's loss subclasses this one and adds the patch
terms in :mod:`losses.hcdr`).
"""
import torch
import torch.nn as nn

# Logits are accumulated in this dtype.  float32 is used throughout (matches the
# AMP training path); bump to float64 if you need bit-for-bit reproduction of
# the original CCLIS score computation.
LOGITS_DTYPE = torch.float32


class ISSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, prototypes_mode='mean',
                 base_temperature=0.07, embedding_shape=512, opt=None):
        super(ISSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.prototypes_mode = prototypes_mode
        self.n_cls = opt.n_cls
        self.mem_size = opt.mem_size
        self.cls_per_task = opt.cls_per_task
        self.embedding_shape = embedding_shape

    @staticmethod
    def _device(features):
        return torch.device('cuda') if features.is_cuda else torch.device('cpu')

    @staticmethod
    def _check_features(features):
        if features.dim() < 2:
            raise ValueError('`features` needs to be [bsz, ...], at least 2 dimensions are required')
        if features.dim() > 2:
            features = features.view(features.shape[0], -1)
        return features

    def _resolve_mask(self, features, labels, mask, device):
        """Build the positive-pair mask from either ``labels`` or ``mask``."""
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        if labels is None and mask is None:
            raise ValueError('`labels` or `mask` should be defined')
        if labels is not None:
            labels = labels.contiguous()
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            all_labels = torch.unique(labels).view(-1, 1).to(device)
            mask = torch.eq(all_labels, labels.T).float().to(device)
            return mask, all_labels
        return mask.float().to(device), None

    def score_calculate(self, output, features, labels=None, importance_weight=None,
                        index=None, target_labels=None, sample_num=[], mask=None,
                        score_mask=[], all_labels=None):
        assert target_labels is not None and len(target_labels) > 0, \
            'Target labels should be given as a list of integer'

        self.replay_sample_num = torch.tensor(sample_num)
        device = self._device(features)
        features = self._check_features(features)

        mask, all_labels = self._resolve_mask(features, labels, mask, device)
        importance_weight = importance_weight.float().to(device)

        if all_labels is not None:
            output = output[:target_labels[-1] + 1, :]

        logits = torch.div(output, self.temperature).to(LOGITS_DTYPE)  # [class_num, batch]
        with torch.no_grad():
            cur_class_num = target_labels[-1] + 1
            batch_score_sum = torch.zeros(cur_class_num, cur_class_num)
            score_mat = torch.exp(logits)
            for idx in range(cur_class_num):
                batch_score_sum[:, idx] = score_mat[:, labels == idx].sum(1)

        return score_mat, batch_score_sum

    def forward(self, output, features, labels=None, importance_weight=None,
                index=None, target_labels=None, sample_num=None, mask=None,
                score_mask=None, all_labels=None):
        assert target_labels is not None and len(target_labels) > 0, \
            'Target labels should be given as a list of integer'

        self.replay_sample_num = torch.tensor(sample_num) if sample_num is not None else torch.Tensor()
        device = self._device(features)
        features = self._check_features(features)

        mask, all_labels = self._resolve_mask(features, labels, mask, device)
        importance_weight = importance_weight.float().to(device)

        if all_labels is not None:
            prototypes_mask = torch.scatter(
                torch.zeros(len(all_labels), self.n_cls).float().to(device),
                1, all_labels.view(-1, 1).to(device), 1)
            output = torch.matmul(prototypes_mask, output)

        # logits with numerical stabilisation
        anchor_dot_contrast = torch.div(output, self.temperature).to(LOGITS_DTYPE)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        task_all_labels = all_labels // self.cls_per_task
        task_labels = labels // self.cls_per_task

        if score_mask is not None:
            label_mask = torch.tensor([score_mask[item] for item in labels.tolist()]).to(device)
            score_scale_mask = torch.eq(all_labels, label_mask).float().to(device)
        else:
            score_scale_mask = torch.ones(len(all_labels), len(labels)).to(device)

        with torch.no_grad():
            _importance_weight = importance_weight * (mask * mask.sum(dim=1, keepdim=True)).sum(dim=0)

        cur_task_mask_col = (task_all_labels != (target_labels[-1] // 2)).float()
        cur_task_mask_row = (task_labels != (target_labels[-1] // 2)).float()
        cur_task_mask = (cur_task_mask_col.view(-1, 1) * cur_task_mask_row).to(device)
        all_mask = score_scale_mask * cur_task_mask * (torch.ones_like(mask) - mask)

        _logits = logits - torch.log(_importance_weight) * all_mask
        log_prob = logits - torch.log(torch.exp(_logits).sum(1, keepdim=True))

        return -(log_prob * mask).sum() / mask.sum()
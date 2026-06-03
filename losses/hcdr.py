"""HCDR loss components.

HCDR (Hierarchical Contrastive Distillation Replay) extends the CCLIS global
objective with two terms:

* ``L_patch_NCE`` (weight ``alpha_patch``) - patch-level InfoNCE that contrasts
  projected patch tokens against per-class patch prototypes.
* ``L_H_PRD``     (weight ``lambda_hprd``) - hierarchical PRD that distils each
  patch's softmax distribution over the *global* class prototypes between the
  frozen past model and the current model.

Full objective: ``L = L_global_NCE + alpha_patch * L_patch_NCE + lambda_hprd * L_H_PRD``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .supcon import ISSupConLoss


class PatchBuffer:
    """Stores frozen global embeddings and top-K discriminative patch positions
    for every image in the replay buffer.

    Raw patch features are intentionally *not* stored - they are re-encoded
    through the current model at replay time so they never go stale.
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self._store = {}  # dataset_idx -> {'z_g': Tensor, 'patch_positions': Tensor}

    def update(self, dataset_indices, z_g: torch.Tensor, patch_positions: torch.Tensor):
        z_g_cpu = z_g.detach().cpu()
        pos_cpu = patch_positions.detach().cpu()
        for i, idx in enumerate(dataset_indices):
            self._store[int(idx)] = {'z_g': z_g_cpu[i], 'patch_positions': pos_cpu[i]}

    def get(self, dataset_indices, device):
        valid, z_g_list, pos_list = [], [], []
        for idx in dataset_indices:
            entry = self._store.get(int(idx))
            valid.append(entry is not None)
            if entry is not None:
                z_g_list.append(entry['z_g'])
                pos_list.append(entry['patch_positions'])
        valid_mask = torch.tensor(valid, dtype=torch.bool)
        if z_g_list:
            return valid_mask, torch.stack(z_g_list).to(device), torch.stack(pos_list).to(device)
        return valid_mask, None, None

    def prune(self, keep_indices):
        keep = {int(i) for i in keep_indices}
        self._store = {k: v for k, v in self._store.items() if k in keep}

    def __len__(self):
        return len(self._store)


def compute_discriminative_patch_scores(patch_feat, proto_weight, labels,
                                        top_k: int = 10, percentile: float = 90.0):
    """HCDR discriminative patch scoring.

    ``score(z) = d(z, c_y) * 1[d(z, c_y) < delta]`` where ``d`` is cosine
    distance, ``c_y`` the ground-truth class prototype and ``delta`` the
    per-image ``percentile``-th within-class distance.  The boundary criterion
    selects tokens near the class manifold edge; the membership gate excludes
    genuine outliers / background.

    Returns ``(top_k_positions [B, top_k], scores [B, P])``.
    """
    with torch.no_grad():
        B, P, D = patch_feat.shape
        proto_per_sample = proto_weight[labels]                      # [B, D]

        patch_norm = F.normalize(patch_feat, dim=-1)                 # [B, P, D]
        proto_norm = F.normalize(proto_per_sample, dim=-1)           # [B, D]
        cos_sim = torch.bmm(patch_norm, proto_norm.unsqueeze(-1)).squeeze(-1)  # [B, P]
        cos_dist = 1.0 - cos_sim

        delta = torch.quantile(cos_dist, percentile / 100.0, dim=1, keepdim=True)
        gate = (cos_dist < delta).float()
        scores = cos_dist * gate

        k = min(top_k, P)
        _, top_k_positions = torch.topk(scores, k, dim=1)
    return top_k_positions, scores


class HCDRLoss(ISSupConLoss):
    """ISSupConLoss + patch-level InfoNCE + hierarchical PRD.

    The global term is computed by ``forward`` (delegated to ISSupConLoss);
    the patch terms are exposed as separate methods and added in the training
    loop so the distillation can reuse the frozen-model forward pass.
    """

    def __init__(self, temperature=0.07, prototypes_mode='mean',
                 base_temperature=0.07, embedding_shape=512, opt=None):
        super().__init__(temperature=temperature, prototypes_mode=prototypes_mode,
                         base_temperature=base_temperature,
                         embedding_shape=embedding_shape, opt=opt)
        self.alpha_patch = opt.alpha_patch if opt is not None else 0.5
        self.lambda_hprd = opt.lambda_hprd if opt is not None else 0.6

    def patch_nce_forward(self, patch_proto_logits, labels):
        """Patch-level InfoNCE over the flattened B*P patch tokens.

        ``patch_proto_logits`` : [B, P, n_cls]; ``labels`` : [B].
        """
        B, P, n_cls = patch_proto_logits.shape
        device = labels.device

        logits = patch_proto_logits.reshape(B * P, n_cls).T            # [n_cls, B*P]
        logits = torch.div(logits, self.temperature).to(torch.float32)

        patch_labels = labels.unsqueeze(1).expand(B, P).reshape(B * P)
        class_ids = torch.arange(n_cls, device=device).view(-1, 1)
        mask = torch.eq(class_ids, patch_labels.unsqueeze(0)).float()  # [n_cls, B*P]

        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        log_prob = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True))
        return -(log_prob * mask).sum() / mask.sum()

    def hprd_forward(self, patch_feat_cur, patch_feat_prev,
                     proto_weight_cur, proto_weight_prev, tau_cur, tau_prev):
        """Distil patch-to-global-prototype distributions between past/current models."""
        logits_cur = torch.matmul(patch_feat_cur, proto_weight_cur.T) / tau_cur
        logits_cur = logits_cur - logits_cur.detach().max(dim=-1, keepdim=True).values
        log_q_cur = F.log_softmax(logits_cur, dim=-1)

        with torch.no_grad():
            logits_prev = torch.matmul(patch_feat_prev, proto_weight_prev.T) / tau_prev
            logits_prev = logits_prev - logits_prev.max(dim=-1, keepdim=True).values
            q_prev = F.softmax(logits_prev, dim=-1)

        return -(q_prev * log_q_cur).sum(dim=-1).mean()


def update_patch_buffer(patch_buffer, model, replay_indices, val_targets,
                        dataset, opt, batch_size: int = 64):
    """Refresh ``patch_buffer`` with the top-K discriminative patch positions of
    every ``replay_indices`` image, using the just-trained model."""
    if not getattr(opt, 'use_hcdr', False) or len(replay_indices) == 0:
        return

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    proto_w = model.prototypes.weight.data
    with torch.no_grad():
        for start in range(0, len(replay_indices), batch_size):
            batch_idx = replay_indices[start:start + batch_size]
            images = torch.stack([dataset[i][0] for i in batch_idx]).to(device)
            batch_labels = torch.tensor([val_targets[i] for i in batch_idx],
                                        dtype=torch.long, device=device)
            feat, _, _, patch_feat, _ = model(images, return_spatial=True)
            top_k_pos, _ = compute_discriminative_patch_scores(
                patch_feat, proto_w, batch_labels, top_k=opt.top_k_patches)
            patch_buffer.update(batch_idx, feat, top_k_pos)
    if was_training:
        model.train()
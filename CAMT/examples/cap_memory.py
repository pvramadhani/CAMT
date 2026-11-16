"""O2CAP-inspired offline+online camera-aware proxy contrast for MMT.

Standalone PyTorch implementation. This is NOT a verbatim replication of O2CAP:
MMT sampling, two teachers, four original losses, and optimizer are held fixed
for a controlled comparison against the user's MMT+CC-B32 baseline.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CameraProxyMemory(nn.Module):
    def __init__(self, features, labels, cameras, temperature=0.05,
                 momentum=0.2, hard_negatives=50, online_positives=3,
                 balance_w=0.15):
        super().__init__()
        if not (temperature > 0 and 0 <= momentum < 1):
            raise ValueError('Invalid temperature/momentum')
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.hard_negatives = int(hard_negatives)
        self.online_positives = int(online_positives)
        self.balance_w = float(balance_w)
        if not (self.hard_negatives >= 1 and self.online_positives >= 1):
            raise ValueError('Invalid number of proxies to mine')

        f = F.normalize(torch.as_tensor(features).float().detach(), dim=1)
        lab = torch.as_tensor(labels, dtype=torch.long, device=f.device)
        cam = torch.as_tensor(cameras, dtype=torch.long, device=f.device)
        if not (len(f) == len(lab) == len(cam)):
            raise ValueError('Feature/label/camera length mismatch')
        valid = lab >= 0
        keys = torch.stack((lab[valid], cam[valid]), dim=1)
        unique_keys, inv = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
        if len(unique_keys) < 2:
            raise ValueError('Need at least two camera-aware proxies')
        counts = torch.bincount(inv, minlength=len(unique_keys)).to(f.device).float()
        sums = f.new_zeros((len(unique_keys), f.shape[1]))
        sums.index_add_(0, inv, f[valid])
        centers = F.normalize(sums / counts[:, None], dim=1)
        self.register_buffer('features', centers.detach().clone())
        self.register_buffer('proxy_labels', unique_keys[:, 0].contiguous())
        self.register_buffer('proxy_cameras', unique_keys[:, 1].contiguous())
        self.register_buffer('proxy_counts', counts)
        self.num_clusters = int(lab[valid].max().item()) + 1
        self.num_proxies = len(unique_keys)
        self.num_cameras = int(torch.unique(cam[valid]).numel())
        self.multicam_clusters = sum(int((self.proxy_labels == k).sum().item() > 1)
                                     for k in range(self.num_clusters))
        # Per-query pseudo-label -> camera proxy map is always from unlabeled
        # training data (never from Market-1501 target ground-truth PIDs).
        self._key_to_proxy = {(int(k), int(c)): j
                              for j, (k, c) in enumerate(unique_keys.tolist())}

    def _proxy_indexes(self, targets, cameras):
        pairs = zip(targets.detach().cpu().tolist(), cameras.detach().cpu().tolist())
        indices = []
        for p in pairs:
            if p not in self._key_to_proxy:
                raise ValueError('Missing camera-aware proxy for pseudo-ID/camera %s' % (p,))
            indices.append(self._key_to_proxy[p])
        return torch.tensor(indices, dtype=torch.long, device=self.features.device)

    @staticmethod
    def _multi_positive_loss(row_logits, pos_indexes, hard_negatives, n_total):
        # Positives are all visible camera proxies from the same pseudo-ID.
        # Negatives are the hardest non-positives. A soft target allocates
        # equal probability mass to every positive proxy (O2CAP-style).
        selected_mask = torch.ones(n_total, dtype=torch.bool, device=row_logits.device)
        selected_mask[pos_indexes] = False
        negative_ids = torch.nonzero(selected_mask, as_tuple=False).view(-1)
        if negative_ids.numel():
            count = min(hard_negatives, negative_ids.numel())
            top_idx = row_logits[negative_ids].detach().topk(count).indices
            negatives = negative_ids[top_idx]
            selected = torch.cat((pos_indexes, negatives), 0)
        else:
            selected = pos_indexes
        scaled = row_logits[selected]
        return -F.log_softmax(scaled, dim=0)[:pos_indexes.numel()].mean()

    def forward(self, embeddings, targets, cameras):
        if embeddings.size(0) != len(targets) or len(targets) != len(cameras):
            raise ValueError('Batch dimensions mismatch')
        q = F.normalize(embeddings.float(), dim=1)
        # Detached memory: gradient flows ONLY through live student embedding.
        memory = self.features.detach().clone()
        scores = q.mm(memory.t())
        logits = scores / self.temperature
        own_proxies = self._proxy_indexes(targets, cameras)
        labels = targets.long()
        offline_losses, online_losses = [], []
        for i in range(q.shape[0]):
            same = torch.nonzero(self.proxy_labels == labels[i], as_tuple=False).view(-1)
            offline_losses.append(self._multi_positive_loss(
                logits[i], same, self.hard_negatives, self.num_proxies))

            # Online association: one nearest proxy per camera; similarity
            # balances the current sample and its own proxy (O2CAP principle).
            with torch.no_grad():
                own = own_proxies[i]
                # Explicit matrix product to preserve correct tensor dimensions.
                related = self.balance_w * scores[i].detach() + (
                    1 - self.balance_w) * memory.mv(memory[own])
                best_per_camera = []
                for cam_id in torch.unique(self.proxy_cameras):
                    idx = torch.nonzero(self.proxy_cameras == cam_id, as_tuple=False).view(-1)
                    best_per_camera.append(idx[related[idx].argmax()])
                candidates = torch.stack(best_per_camera)
                selected_n = min(self.online_positives, len(candidates))
                chosen = candidates[related[candidates].topk(selected_n).indices]
                # Preserve at least the known positive proxy for the sample.
                # This is a safety constraint added for MMT integration.
                if not (chosen == own).any():
                    chosen[-1] = own
                chosen = torch.unique(chosen)
            online_losses.append(self._multi_positive_loss(
                logits[i], chosen, self.hard_negatives, self.num_proxies))
        return torch.stack(offline_losses).mean(), torch.stack(online_losses).mean()

    @torch.no_grad()
    def update(self, teacher_features, targets, cameras):
        """After optimizer step: order-invariant momentum mean per proxy."""
        q = F.normalize(teacher_features.detach().float(), dim=1)
        proxy_targets = self._proxy_indexes(targets, cameras)
        uniques, inv = torch.unique(proxy_targets, sorted=True, return_inverse=True)
        sums = self.features.new_zeros((uniques.numel(), q.shape[1]))
        sums.index_add_(0, inv, q)
        counts = torch.bincount(inv, minlength=len(uniques)).to(q.device).clamp_min(1)
        cur = F.normalize(sums / counts[:, None], dim=1)
        old = self.features.index_select(0, uniques)
        new = F.normalize(self.momentum * old + (1 - self.momentum) * cur, dim=1)
        self.features.index_copy_(0, uniques, new)

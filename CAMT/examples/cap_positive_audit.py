#!/usr/bin/env python3
"""Read-only MMT+CAP positive proxy audit (requires MMT project). No target PID is read.

This reconstructs a POST-TRAINING snapshot, not the online memory/labels at an
historical training iteration. Teacher checkpoints MUST be from the same epoch.
It matches the user's MMT_CAP/cap_memory.py online mining rule:
  score=0.15*sample_proxy_similarity + 0.85*own_proxy_proxy_similarity;
  choose top-1 proxy per camera; top-3 camera candidates; force own proxy.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN


def quantiles(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return None
    return {str(k): round(float(np.quantile(arr, q)), 6) for k, q in
            [("p10", .1), ("median", .5), ("p90", .9)]}


def read_eps(log_path):
    content = Path(log_path).read_text(errors='replace')
    m = re.search(r'eps for cluster:\s*([0-9.]+)', content)
    if not m:
        raise RuntimeError('DBSCAN eps not in log. Cannot faithfully reproduce threshold.')
    return float(m.group(1))


def initialize_proxies(fused, labels, cameras):
    """Exactly the proxy initialization of MMT_CAP/cap_memory.py.

    fused: normalized torch tensor (N,D), labels: numpy (N,), cams: numpy (N,)
    """
    device = fused.device
    lab = torch.as_tensor(labels, dtype=torch.long, device=device)
    cam = torch.as_tensor(cameras, dtype=torch.long, device=device)
    valid = lab >= 0
    if int(valid.sum()) == 0:
        raise ValueError('DBSCAN found no inlier samples')
    keys, inv = torch.unique(torch.stack((lab[valid], cam[valid]), dim=1),
                             dim=0, sorted=True, return_inverse=True)
    sums = torch.zeros(len(keys), fused.size(1), dtype=torch.float32, device=device)
    sums.index_add_(0, inv, F.normalize(fused[valid].float(), dim=1))
    cnt = torch.bincount(inv, minlength=len(keys)).float().to(device)
    centers = F.normalize(sums / cnt[:, None], dim=1)
    return valid, inv, keys, centers, cnt


def audit_embeddings(teacher1, teacher2, labels, cameras,
                     batch_size=512, online_k=3, balance_w=.15, device='cpu'):
    """Pure tensor core: permits unit tests without loading MMT or checkpoints."""
    f1 = F.normalize(torch.as_tensor(teacher1, dtype=torch.float32, device=device), dim=1)
    f2 = F.normalize(torch.as_tensor(teacher2, dtype=torch.float32, device=device), dim=1)
    if f1.shape != f2.shape:
        raise ValueError('Teacher feature dimensions differ')
    fused = F.normalize((f1 + f2) * .5, dim=1)
    cam = np.asarray(cameras, dtype=np.int64)
    lab = np.asarray(labels, dtype=np.int64)
    if len(lab) != len(f1) or len(cam) != len(f1):
        raise ValueError('Image/label/camera mismatch')
    valid, inv, keys, mem, counts = initialize_proxies(fused, lab, cam)
    idx_in = np.flatnonzero(lab >= 0)
    assigned = lab[idx_in]
    nc = np.unique(cam[idx_in]).size
    k = keys.size(0)
    if k < 2:
        raise ValueError('Not enough proxies')
    proxy_labs = keys[:, 0]
    proxy_cams = keys[:, 1]
    unique_cams = torch.unique(proxy_cams)
    # MMT_CAP's initialization uses torch.unique(keys,return_inverse=True).
    own_all = inv
    cam_groups = [(proxy_cams == c).nonzero(as_tuple=True)[0] for c in unique_cams]

    coverage = Counter()
    for cl in torch.unique(proxy_labs).tolist():
        coverage[int((proxy_labs == cl).sum())] += 1
    support = []
    self_forced = []
    cross_counts = []
    query_cross = []
    teacher_top_agree = []
    teacher_supported = []
    margins = []
    total_queries = int(len(idx_in))
    # Use the same fixed-K + self-preservation as the *user's* CAP code.
    for start in range(0, total_queries, batch_size):
        stop = min(start + batch_size, total_queries)
        gidx = idx_in[start:stop]
        q = fused[gidx]
        own = own_all[start:stop]
        same = torch.as_tensor(lab[gidx], device=device, dtype=torch.long)
        scores = q @ mem.T
        # O2CAP-inspired score from MMT_CAP/cap_memory.py
        rel = balance_w * scores + (1. - balance_w) * (mem[own] @ mem.T)
        per_cam = torch.stack([
            group[rel[:, group].argmax(dim=1)] for group in cam_groups
        ], dim=1)
        per_cam_scores = rel.gather(1, per_cam)
        nk = min(online_k, per_cam.shape[1])
        choice = per_cam.gather(1, per_cam_scores.topk(nk, dim=1).indices)
        already_own = (choice == own[:, None]).any(dim=1)
        force = ~already_own
        if force.any():
            choice[force, -1] = own[force]
        mismatch = (proxy_labs[choice] != same[:, None])
        cross_counts.extend(mismatch.sum(dim=1).cpu().tolist())
        query_cross.extend(mismatch.any(dim=1).cpu().tolist())
        self_forced.extend(force.cpu().tolist())
        # Coverage from keys as true offline positive count, no identity labels.
        uniq_cl, inverse = torch.unique(same, sorted=True, return_inverse=True)
        cam_per_cluster = {int(cluster): int((proxy_labs == cluster).sum()) for cluster in uniq_cl.tolist()}
        support.extend([cam_per_cluster[int(c)] for c in same.tolist()])

        # Two EMA teachers evaluated against the SAME shared proxy memory.
        # This is a diagnostic, NOT independent proof of proxy correctness.
        score1 = f1[gidx] @ mem.T
        score2 = f2[gidx] @ mem.T
        vote1 = torch.stack([group[score1[:, group].argmax(dim=1)] for group in cam_groups], dim=1)
        vote2 = torch.stack([group[score2[:, group].argmax(dim=1)] for group in cam_groups], dim=1)
        teacher_top_agree.extend((vote1 == vote2).float().mean(dim=1).cpu().tolist())
        chosen_cams = proxy_cams[choice]
        chosen_vote1 = torch.zeros_like(choice, dtype=torch.bool)
        chosen_vote2 = torch.zeros_like(choice, dtype=torch.bool)
        for ci, cam_value in enumerate(unique_cams):
            mask = chosen_cams == cam_value
            chosen_vote1 |= mask & (choice == vote1[:, ci:ci+1])
            chosen_vote2 |= mask & (choice == vote2[:, ci:ci+1])
        teacher_supported.extend((chosen_vote1 & chosen_vote2).float().mean(dim=1).cpu().tolist())
        if per_cam_scores.size(1) >= 2:
            margins.extend((per_cam_scores.topk(2, dim=1).values[:, 0] -
                            per_cam_scores.topk(2, dim=1).values[:, 1]).cpu().tolist())

    nout = int((lab < 0).sum())
    summary = {
        'snapshot_type': 'POST-TRAINING reconstruction using paired last-epoch EMA teachers',
        'n_target_images': int(len(lab)), 'n_inliers': total_queries,
        'n_outliers': nout, 'n_clusters': int(torch.unique(proxy_labs).numel()),
        'n_proxies': int(k), 'n_cameras': int(nc),
        'cluster_camera_coverage_histogram': {str(k): int(v) for k, v in sorted(coverage.items())},
        'single_camera_clusters_pct': round(100 * coverage.get(1, 0) / sum(coverage.values()), 3),
        'clusters_with_fewer_cameras_than_online_K_pct': round(
            100 * sum(v for x, v in coverage.items() if x < online_k) / sum(coverage.values()), 3),
        'query_images_with_fewer_cluster_cameras_than_online_K_pct': round(
            100 * float(np.mean(np.asarray(support) < online_k)), 3),
        'online_positive_K': int(min(online_k, nc)),
        'online_selected_cross_pseudo_cluster_proxies_pct': round(
            100 * float(np.mean(cross_counts)) / min(online_k, nc), 3),
        'queries_with_any_online_cross_cluster_positive_pct': round(
            100 * float(np.mean(query_cross)), 3),
        'online_forced_self_proxy_pct_of_queries': round(
            100 * float(np.mean(self_forced)), 3),
        'cross_cluster_online_proxy_count_per_image':
            {str(j): int(v) for j, v in sorted(Counter(cross_counts).items())},
        'two_teacher_per_camera_nearest_proxy_agreement_mean_pct': round(
            100 * float(np.mean(teacher_top_agree)), 3),
        'online_selected_proxy_top1_under_both_teachers_mean_pct': round(
            100 * float(np.mean(teacher_supported)), 3),
        'top_camera_relevance_margin_quantiles': quantiles(margins),
        'interpretation_warning': 'Cross-cluster != false identity association. Teacher agreement != identity correctness. No target ground-truth identities read. Snapshot proxies are freshly initialized, not live epoch memory.',
    }
    return summary


def load_features(model, checkpoint, data_loader, train_order):
    from mmt.evaluators import extract_features
    from mmt.utils.serialization import load_checkpoint, copy_state_dict
    state = load_checkpoint(str(checkpoint))
    copy_state_dict(state['state_dict'], model)
    print(f"Checkpoint: {checkpoint}, epoch={state.get('epoch')}, stored_mAP={state.get('best_mAP')}", flush=True)
    features, _ = extract_features(model, data_loader, print_freq=50)
    missing = [fname for fname in train_order if fname not in features]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} features; first={missing[:2]}")
    return torch.stack([features[fname] for fname in train_order]), state.get('epoch')


def main():
    p = argparse.ArgumentParser(description='One-shot CAP positive association audit, read-only')
    p.add_argument('--log-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--data-dir', default='examples/data')
    p.add_argument('--dataset', default='market1501')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--jaccard-cpu', action='store_true')
    args = p.parse_args()
    from mmt import models, datasets
    from mmt.utils.data import transforms as T
    from mmt.utils.data.preprocessor import Preprocessor
    from mmt.utils.rerank import compute_jaccard_dist
    from torch import nn
    from torch.utils.data import DataLoader

    logs = Path(args.log_dir)
    out = Path(args.out_dir)
    ck1, ck2 = logs / 'model1_checkpoint.pth.tar', logs / 'model2_checkpoint.pth.tar'
    for f in (ck1, ck2, logs / 'log.txt'):
        if not f.is_file():
            raise FileNotFoundError(f)
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU needed for feature extraction; CUDA not detected')
    dataset = datasets.create(args.dataset, str(Path(args.data_dir) / args.dataset))
    # Crucial: use ONLY filename and CAMERA ID, NEVER target training PID (entry[1]).
    records = sorted(dataset.train, key=lambda x: x[0])
    names = [r[0] for r in records]
    cams = np.array([int(r[2]) for r in records], dtype=np.int64)
    trans = T.Compose([T.Resize((256, 128), interpolation=3), T.ToTensor(),
        T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    loader = DataLoader(Preprocessor(records, root=dataset.images_dir, transform=trans),
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    model = nn.DataParallel(models.create('resnet50', pretrained=False,
                                num_features=0, dropout=0, num_classes=0).cuda())
    print('[1/4] Extract paired last-epoch EMA features', flush=True)
    f1, e1 = load_features(model, ck1, loader, names)
    f2, e2 = load_features(model, ck2, loader, names)
    if e1 != e2:
        raise RuntimeError(f'Teacher epochs are different: {e1} vs {e2}; audit aborted')
    print(f'Paired EMA checkpoint epoch={e1}; extracted {len(names)} aligned images', flush=True)
    del model
    torch.cuda.empty_cache()
    f1 = F.normalize(f1.float(), dim=1)
    f2 = F.normalize(f2.float(), dim=1)
    merged = F.normalize((f1 + f2) / 2, dim=1)
    eps = read_eps(logs / 'log.txt')
    print(f'[2/4] Jaccard distance; eps={eps:.3f} (rounded in training log)', flush=True)
    try:
        d = compute_jaccard_dist(merged, use_gpu=(not args.jaccard_cpu)).numpy()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        if args.jaccard_cpu or 'out of memory' not in str(error).lower():
            raise
        print('GPU memory insufficient: retrying Jaccard on CPU (same algorithm)', flush=True)
        torch.cuda.empty_cache()
        d = compute_jaccard_dist(merged, use_gpu=False).numpy()
    if d.shape != (len(names), len(names)):
        raise RuntimeError('Distance matrix shape invalid')
    print('[3/4] DBSCAN reconstruction; this is a snapshot, not stored train-time labels', flush=True)
    label = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1).fit_predict(d)
    del d
    # Only a few proxy x query matrices on GPU, not full NxN Jaccard.
    print('[4/4] Audit CAP top-3 including forced-self and two-teacher agreement', flush=True)
    audit = audit_embeddings(f1, f2, label, cams, batch_size=512, online_k=3,
                             balance_w=.15, device='cuda')
    audit['checkpoint_epoch'] = int(e1)
    audit['dbscan_eps_from_training_log_rounded'] = eps
    audit['source'] = 'DukeMTMC-reID'
    audit['target'] = 'Market-1501 (unlabeled train only)'
    dest = out / 'cap_positive_audit.json'
    dest.write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
    lines = [
        '=== MMT + CAP: POSITIVE ASSOCIATION AUDIT ===',
        f"Paired teachers from checkpoint epoch: {e1}",
        f"Clusters / proxies / inliers: {audit['n_clusters']} / {audit['n_proxies']} / {audit['n_inliers']}",
        f"Camera coverage by cluster: {audit['cluster_camera_coverage_histogram']}",
        f"Single-camera clusters: {audit['single_camera_clusters_pct']:.2f}%",
        f"Clusters with camera count < K=3: {audit['clusters_with_fewer_cameras_than_online_K_pct']:.2f}%",
        f"Online-selected proxies outside query pseudo-cluster: {audit['online_selected_cross_pseudo_cluster_proxies_pct']:.2f}%",
        f"Queries with >=1 online cross-cluster positive: {audit['queries_with_any_online_cross_cluster_positive_pct']:.2f}%",
        f"Queries where own proxy forced into top-3: {audit['online_forced_self_proxy_pct_of_queries']:.2f}%",
        f"Two-teacher per-camera top1 proxy agreement: {audit['two_teacher_per_camera_nearest_proxy_agreement_mean_pct']:.2f}%",
        f"Selected proxy top1 according to BOTH teachers: {audit['online_selected_proxy_top1_under_both_teachers_mean_pct']:.2f}%",
        'WARNING: cross-cluster != incorrect identity. Labels/memory are post-training reconstruction.',
        f'Full JSON: {dest}',
    ]
    report = '\n'.join(lines) + '\n'
    (out / 'cap_positive_summary.txt').write_text(report)
    print(report, flush=True)


if __name__ == '__main__':
    main()

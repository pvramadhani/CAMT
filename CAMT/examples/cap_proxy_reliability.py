#!/usr/bin/env python3
"""Read-only proxy reliability diagnostic for the user's MMT+CAP implementation.

Post-training reconstructed pseudo labels (not true identities), paired EMA checkpoints
from same epoch, own camera-specific proxies rebuilt as mean. No target train IDs used.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

from cap_positive_audit import (read_eps, initialize_proxies, load_features)


def percent(value):
    return round(float(value) * 100.0, 3)


def summary_quantiles(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return {"p10": round(float(np.quantile(a, .1)), 5),
            "median": round(float(np.median(a)), 5),
            "p90": round(float(np.quantile(a, .9)), 5)}


def proxy_audit(f1, f2, labels, cameras, device="cpu"):
    """Pure numeric core; can be unit tested without datasets or GPU."""
    f1 = F.normalize(torch.as_tensor(f1, dtype=torch.float32, device=device), dim=1)
    f2 = F.normalize(torch.as_tensor(f2, dtype=torch.float32, device=device), dim=1)
    if f1.shape != f2.shape or f1.shape[0] != len(labels):
        raise ValueError("Features or label alignment incorrect")
    fused = F.normalize((f1 + f2) * .5, dim=1)
    valid, inv, keys, proxies, counts = initialize_proxies(fused, labels, cameras)
    n = len(keys)
    if n < 2:
        raise RuntimeError("Insufficient reconstructed proxies")

    # Cluster-internal dispersion: 1 - mean cosine(x_i, camera-specific center)
    # This is not sample-label correctness. Singleton proxy has trivial dispersion.
    sample_sim = (fused[valid] * proxies[inv]).sum(1)
    sums = torch.zeros(n, dtype=torch.float32, device=device)
    sums.index_add_(0, inv, 1.0 - sample_sim)
    dispersion = sums / counts

    # Teacher center disagreement evaluated with IDENTICAL pseudo labels.
    sums1 = torch.zeros_like(proxies)
    sums2 = torch.zeros_like(proxies)
    sums1.index_add_(0, inv, f1[valid])
    sums2.index_add_(0, inv, f2[valid])
    p1 = F.normalize(sums1 / counts[:, None], dim=1)
    p2 = F.normalize(sums2 / counts[:, None], dim=1)
    teacher_center_cos = (p1 * p2).sum(1).clamp(-1, 1)

    pseudoid = keys[:, 0]
    camid = keys[:, 1]
    # Compare strongest OTHER-CAMERA same pseudo-ID vs strongest OTHER-CAMERA
    # different pseudo-ID, proxy-by-proxy. This is proxy ambiguity, not error rate.
    different_camera = camid[:, None] != camid[None, :]
    positives = (pseudoid[:, None] == pseudoid[None, :]) & different_camera
    impostors = (pseudoid[:, None] != pseudoid[None, :]) & different_camera
    present = positives.any(1) & impostors.any(1)

    def margins(p):
        sims = p @ p.T
        pos = sims.masked_fill(~positives, float("-inf")).max(dim=1).values
        neg = sims.masked_fill(~impostors, float("-inf")).max(dim=1).values
        gap = pos - neg
        gap[~present] = float("nan")
        return gap, pos, neg

    margin, pos, neg = margins(proxies)
    margin1, _, _ = margins(p1)
    margin2, _, _ = margins(p2)

    asnp = lambda x: x.detach().cpu().numpy()
    disp = asnp(dispersion)
    count = asnp(counts).astype(int)
    teacher_cos = asnp(teacher_center_cos)
    gap = asnp(margin)
    gap1 = asnp(margin1)
    gap2 = asnp(margin2)
    keys_np = asnp(keys)
    good = np.isfinite(gap)
    non_single = count >= 2
    eligible = good & non_single
    # 'ambiguous' means a different pseudo-ID proxy is at least as similar as
    # the most similar same pseudo-ID other-camera proxy. No GT identity usage.
    ambiguous = good & (gap <= 0)
    teacher_sign_disagree = good & ((gap1 <= 0) != (gap2 <= 0))

    # A directional check: are highly dispersed groups *more often* ambiguous?
    if eligible.any():
        q75 = float(np.quantile(disp[eligible], .75))
        high = eligible & (disp >= q75)
        low = eligible & (disp < q75)
    else:
        q75, high, low = None, np.zeros(n, bool), np.zeros(n, bool)

    def rate(condition, denominator):
        return percent(condition[denominator].mean()) if np.any(denominator) else None

    report = {
        "audit_type": "post-training reconstructed proxy compactness + cross-camera ambiguity",
        "n_proxies": int(n),
        "n_pseudo_clusters": int(torch.unique(pseudoid).numel()),
        "n_inlier_images": int(valid.sum()),
        "n_singleton_proxies": int(np.sum(count == 1)),
        "n_multi_camera_eligible_proxies": int(good.sum()),
        "proxy_support_quantiles": summary_quantiles(count),
        "proxy_dispersion_non_singleton_quantiles": summary_quantiles(disp[non_single]),
        "cross_camera_proxy_margin_quantiles": summary_quantiles(gap[good]),
        "cross_camera_ambiguous_proxy_pct": rate(ambiguous, good),
        "teacher_proxy_center_cosine_quantiles": summary_quantiles(teacher_cos),
        "teacher_cross_camera_margin_sign_disagreement_pct": rate(teacher_sign_disagree, good),
        "high_dispersion_threshold_q75": round(q75, 6) if q75 is not None else None,
        "ambiguous_rate_high_dispersion_pct": rate(ambiguous, high),
        "ambiguous_rate_lower_dispersion_pct": rate(ambiguous, low),
        "warnings": [
            "No target training identity labels are used. Pseudo-cluster != ground-truth identity.",
            "Negative proxy margin indicates feature ambiguity, NOT a confirmed incorrect association.",
            "Teacher agreement and proxy compactness can be high for incorrectly clustered samples.",
            "Reconstructed pseudo-labels use rounded epsilon, not original epoch-specific labels.",
            "Recomputed centers are not the mutable memory state during training.",
        ],
    }
    rows = []
    for i, ((pid, camera), ct, di, ag, gm, g1, g2) in enumerate(zip(
            keys_np.tolist(), count, disp, teacher_cos, gap, gap1, gap2)):
        rows.append({
            "proxy_index": i, "pseudo_cluster": int(pid), "camera_id": int(camera),
            "n_images": int(ct), "dispersion": float(di),
            "teacher_center_cosine": float(ag),
            "crosscamera_margin": float(gm) if np.isfinite(gm) else "",
            "teacher1_margin": float(g1) if np.isfinite(g1) else "",
            "teacher2_margin": float(g2) if np.isfinite(g2) else "",
            "ambiguous_proxy": int(bool(ambiguous[i])),
        })
    return report, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data-dir", default="examples/data")
    p.add_argument("--dataset", default="market1501")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--jaccard-cpu", action="store_true")
    args = p.parse_args()
    from mmt import datasets, models
    from mmt.utils.data import transforms as T
    from mmt.utils.data.preprocessor import Preprocessor
    from mmt.utils.rerank import compute_jaccard_dist
    from torch import nn
    from torch.utils.data import DataLoader

    logs = Path(args.log_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c1, c2 = logs / "model1_checkpoint.pth.tar", logs / "model2_checkpoint.pth.tar"
    for fn in (c1, c2, logs / "log.txt"):
        if not fn.is_file():
            raise FileNotFoundError(str(fn))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for original MMT feature extraction")
    ds = datasets.create(args.dataset, str(Path(args.data_dir) / args.dataset))
    # Intentionally do not access r[1], the target training ground-truth PID.
    records = sorted(ds.train, key=lambda r: r[0])
    names = [r[0] for r in records]
    cams = np.array([int(r[2]) for r in records], dtype=np.int64)
    trans = T.Compose([T.Resize((256, 128), interpolation=3), T.ToTensor(),
        T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    loader = DataLoader(Preprocessor(records, root=ds.images_dir, transform=trans),
        batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, pin_memory=True)
    model = nn.DataParallel(models.create('resnet50', pretrained=False,
                             num_features=0, dropout=0, num_classes=0).cuda())
    print('[1/3] Extracting two same-epoch EMA teachers...', flush=True)
    f1, ep1 = load_features(model, c1, loader, names)
    f2, ep2 = load_features(model, c2, loader, names)
    if ep1 != ep2:
        raise RuntimeError('Teacher checkpoint epochs differ: %s vs %s' % (ep1, ep2))
    del model
    torch.cuda.empty_cache()
    f1 = F.normalize(f1.float(), dim=1)
    f2 = F.normalize(f2.float(), dim=1)
    fused = F.normalize((f1 + f2) / 2, dim=1)
    eps = read_eps(logs / 'log.txt')
    print('[2/3] Reconstructing DBSCAN snapshot with rounded epsilon %s...' % eps, flush=True)
    try:
        dist = compute_jaccard_dist(fused, use_gpu=(not args.jaccard_cpu)).numpy()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if args.jaccard_cpu or 'out of memory' not in str(e).lower():
            raise
        print('GPU Jaccard OOM; retrying identical algorithm on CPU', flush=True)
        torch.cuda.empty_cache()
        dist = compute_jaccard_dist(fused, use_gpu=False).numpy()
    pseudo_labels = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1).fit_predict(dist)
    del dist
    print('[3/3] Proxy compactness, cross-camera margins, EMA teacher consistency...', flush=True)
    report, rows = proxy_audit(f1, f2, pseudo_labels, cams, device='cuda')
    report.update({'checkpoint_epoch': int(ep1), 'dbscan_epsilon_rounded': eps,
                   'source': 'dukemtmc', 'target': args.dataset})
    (out / 'proxy_reliability.json').write_text(json.dumps(report, indent=2) + '\n')
    with (out / 'proxy_per_group.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = [
        '=== MMT + CAP: PROXY RELIABILITY AUDIT ===',
        'Checkpoint epoch: %s' % ep1,
        'Proxies / clusters / inlier images: %s / %s / %s' % (
            report['n_proxies'], report['n_pseudo_clusters'], report['n_inlier_images']),
        'Singleton proxies: %s' % report['n_singleton_proxies'],
        'Proxy support: %s' % report['proxy_support_quantiles'],
        'Non-singleton dispersion: %s' % report['proxy_dispersion_non_singleton_quantiles'],
        'Cross-camera margin: %s' % report['cross_camera_proxy_margin_quantiles'],
        'Proxy ambiguity (negative cross-camera margin): %s%%' % report['cross_camera_ambiguous_proxy_pct'],
        'Teacher proxy center cosine: %s' % report['teacher_proxy_center_cosine_quantiles'],
        'Teacher cross-camera margin sign disagreement: %s%%' % (
            report['teacher_cross_camera_margin_sign_disagreement_pct']),
        'Ambiguity: high dispersion / lower dispersion: %s%% / %s%%' % (
            report['ambiguous_rate_high_dispersion_pct'], report['ambiguous_rate_lower_dispersion_pct']),
        'CAUTION: pseudo-ID is not ground truth; negative margin is ambiguous, not confirmed error.',
        'JSON: ' + str(out / 'proxy_reliability.json'),
        'CSV: ' + str(out / 'proxy_per_group.csv')
    ]
    summary = '\n'.join(lines) + '\n'
    (out / 'proxy_reliability_summary.txt').write_text(summary)
    print(summary, flush=True)

if __name__ == '__main__':
    main()

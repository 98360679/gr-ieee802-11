#!/usr/bin/env python3
"""
exp3_device_similarity.py — screen devices for near-duplicates BEFORE a full collection.

Given a fresh capture (each device transmitting the SAME content, fixed gain), reports two
views of similarity and recommends which device to REPLACE to maximize distinctness:

  (1) impairment-space distance (model-free): per-device DC / I-Q gain / I-Q quadrature /
      CFO / PAPR signature, z-scored per feature, pairwise Euclidean distance.
  (2) fingerprint confusability (the decisive metric): train a quick K-class CNN and read the
      confusion matrix; pairwise confusion = how often a fingerprinter mistakes one device for
      the other. High confusion / low separability = a pair a classifier CANNOT tell apart.

Leakage-safe when >1 run/device (holds out --val-run); single-run falls back to a frame-index
split (semi-leaky — flagged). Content is shared across devices, so confusion reflects HARDWARE.

  python3 exp3_device_similarity.py --root <dir with device_1../device_N/*.bin> \
      [--val-run 3] [--epochs 15] [--thr-mult 2]
"""
import os, argparse, itertools, re, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from exp3_fp_model import (FingerprintCNN, frame_to_windows, iq_to_input,
                           NUM_CLASSES, DEVICE_NAMES, WIN)
from exp3_extract_frames import extract_frames_for_file
from exp3_finetune import per_device_eval, augment, find_runs
from exp3_impairment_fit import frame_stats


def build(root, thr_mult, floor_pct, val_run, hop_train, max_frames):
    Xtr, ytr, Xva, yva, vfid, sigs, present = [], [], [], [], [], {}, []
    fid = 0
    for d in range(1, NUM_CLASSES + 1):
        runs = find_runs(root, d)
        if not runs:
            continue
        present.append(d)
        multi = len(runs) > 1
        dstats = []
        for cap in runs:
            rm = re.search(r'run_?(\d+)', os.path.basename(cap))
            run_no = int(rm.group(1)) if rm else 1
            frames = extract_frames_for_file(cap, floor_pct=floor_pct, thr_mult=thr_mult)[0][:max_frames]
            for i, f in enumerate(frames):
                dstats.append(frame_stats(f))
                is_val = (run_no == val_run) if multi else (i % 5 == 0)
                x = iq_to_input(frame_to_windows(f, hop=WIN if is_val else hop_train))
                if is_val:
                    Xva.append(x); yva.append(np.full(len(x), d - 1))
                    vfid.append(np.full(len(x), fid)); fid += 1
                else:
                    Xtr.append(x); ytr.append(np.full(len(x), d - 1))
        sigs[d] = np.mean(dstats, 0)
        split = f"run{val_run} held out" if multi else "frame-index split (semi-leaky)"
        print(f"  device_{d}: {len(dstats)} frames over {len(runs)} run(s)  ({split})")
    tr = (np.concatenate(Xtr), np.concatenate(ytr).astype(np.int64))
    va = (np.concatenate(Xva), np.concatenate(yva).astype(np.int64), np.concatenate(vfid).astype(np.int64))
    return tr, va, sigs, present


def train_quick(tr, va, epochs, dev):
    Xtr, ytr = tr; Xva, yva, vfid = va
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=256, shuffle=True)
    m = FingerprintCNN(NUM_CLASSES).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_C, best_per, bw = 0, None, None, 0
    Xva_t = torch.from_numpy(Xva)
    for ep in range(1, epochs + 1):
        m.train()
        for xb, yb in dl:
            xb = augment(xb.to(dev)); yb = yb.to(dev)
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
        sched.step()
        w, f, per, C = per_device_eval(m, dev, Xva_t, yva, vfid, NUM_CLASSES)
        if f > best:
            best, best_C, best_per, bw = f, C, per, w
    return best, bw, best_per, best_C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--val-run', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--thr-mult', type=float, default=2.0)
    ap.add_argument('--floor-pct', type=float, default=20.0)
    ap.add_argument('--hop-train', type=int, default=384)
    ap.add_argument('--max-frames', type=int, default=400)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); np.random.seed(0)
    nm = lambda d: DEVICE_NAMES[d - 1]

    print(f"Device-similarity screen: {a.root}\n")
    tr, va, sigs, present = build(a.root, a.thr_mult, a.floor_pct, a.val_run, a.hop_train, a.max_frames)
    frame_acc, wacc, per, C = train_quick(tr, va, a.epochs, dev)
    print(f"\n  quick {a.epochs}-epoch model: overall frame acc {frame_acc:.3f}")
    for d in present:
        ok, tot = per[d - 1]; print(f"    {nm(d)}: {ok/max(1,tot):.3f} ({ok}/{tot})")

    # (2) pairwise confusability from the confusion matrix
    print("\n  === pairwise fingerprint confusability (higher = more similar) ===")
    conf = {}
    rs = C.sum(1)
    for i, j in itertools.combinations([d - 1 for d in present], 2):
        c = (C[i, j] + C[j, i]) / max(1, rs[i] + rs[j])
        conf[(i, j)] = c
    for (i, j), c in sorted(conf.items(), key=lambda x: -x[1]):
        sep = 1 - c
        bar = '#' * int(round(c * 40))
        print(f"    {nm(i+1)}<->{nm(j+1)}: confusion {c:.3f}  separability {sep:.3f}  {bar}")

    # (1) impairment-space distance (z-scored)
    print("\n  === impairment-space distance (model-free, z-scored) ===")
    ds = sorted(sigs)
    M = np.array([sigs[d] for d in ds])
    Z = (M - M.mean(0)) / (M.std(0) + 1e-9)
    imp = {}
    for a_, b_ in itertools.combinations(range(len(ds)), 2):
        imp[(ds[a_], ds[b_])] = float(np.linalg.norm(Z[a_] - Z[b_]))
    for (i, j), dnorm in sorted(imp.items(), key=lambda x: x[1]):
        print(f"    {nm(i)}<->{nm(j)}: impairment-dist {dnorm:.2f}")

    # recommendation: the most-confused pair; replace the more globally-redundant of the two
    tot_conf = {d - 1: sum(conf.get(tuple(sorted((d - 1, e - 1))), 0) for e in present if e != d) for d in present}
    (pi, pj), cmax = max(conf.items(), key=lambda x: x[1])
    drop = pi if tot_conf[pi] >= tot_conf[pj] else pj
    keep = pj if drop == pi else pi
    print("\n  === RECOMMENDATION ===")
    print(f"  Most similar pair: {nm(pi+1)} <-> {nm(pj+1)}  (confusion {cmax:.3f}, separability {1-cmax:.3f})")
    print(f"  Replace -> {nm(drop+1)}  (more globally redundant: total confusion "
          f"{tot_conf[drop]:.3f} vs {tot_conf[keep]:.3f} for {nm(keep+1)})")
    print(f"  Keeping {nm(keep+1)} preserves the distinct member of the pair.")
    if cmax < 0.05:
        print("  NOTE: even the closest pair is well-separated (confusion <0.05) — no swap needed.")


if __name__ == '__main__':
    main()

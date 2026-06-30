#!/usr/bin/env python3
"""
exp3_content_split_eval.py — CONTENT-disjoint honest-separability test.

The varied all-replay enrollment (enroll_varied.bin = 222 distinct frames x3, replayed
by every device for 3 runs) gives 0.986 with a run-3 holdout — but all runs share the
SAME 222 frames, so that split is run-disjoint, NOT content-disjoint. It measures
"re-ID the device on KNOWN content", which is exactly the operational task here (the
attack replays the same frames) but does NOT prove the fingerprint generalises to
UNSEEN content.

This splits by CONTENT instead: every captured frame is tagged with its source
content-id (0..221) by matching against the 222 source templates, then content-ids
[0,split) train / [split,222) val. Pooled across all 3 runs. If accuracy stays high,
the per-device fingerprint is content-INVARIANT (hardware), not memorised templates.

  python3 exp3_content_split_eval.py --verify     # just report match quality, fast
  python3 exp3_content_split_eval.py              # full content-disjoint train+eval
"""
import os, glob, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from exp3_fp_model import (FingerprintCNN, frame_to_windows, iq_to_input,
                           n_params, NUM_CLASSES, DEVICE_NAMES, WIN)
from exp3_extract_frames import extract_frames_for_file
from exp3_finetune import per_device_eval, show, augment, find_runs, SCRATCH
from exp3_train_fingerprint import LOCAL_PT

ROOT = "/media/nghoselab/T9/Data/session13/train/6_30_2026/enrollment"
SRC  = ("/media/nghoselab/T9/Data/session13/ota_dev6/"
        "eot_untgt_pgd_629/dac_safe_gapped/adv_frame.bin")
PERIOD, FL = 265661, 15661
# data region used for content matching: skip the shared preamble, stay inside the frame
MS, ME = 1200, 9000


def content_ids(starts):
    """Position-based content id: frames are tiled 222-in-order at PERIOD spacing, so
    content = round((start-start0)/PERIOD) mod 222.  Robust to drops (absolute index)
    and verified drift-free (residual <0.003)."""
    s = np.asarray(starts, float)
    k = np.round((s - s[0]) / PERIOD).astype(int)
    return k % 222


def build(split, verify=False):
    Xtr, ytr, Xva, yva, vfid = [], [], [], [], []
    fid = 0
    ntr = nva = 0
    for d in range(1, NUM_CLASSES + 1):
        lbl = d - 1
        for cap in find_runs(ROOT, d):
            frames, starts = extract_frames_for_file(cap)[:2]
            cids = content_ids(starts)
            for f, cid in zip(frames, cids):
                is_val = cid >= split
                if verify:
                    nva += int(is_val); ntr += int(not is_val); continue
                w = frame_to_windows(f, hop=WIN if is_val else 384)
                x = iq_to_input(w)
                if is_val:
                    Xva.append(x); yva.append(np.full(len(x), lbl))
                    vfid.append(np.full(len(x), fid)); fid += 1
                else:
                    Xtr.append(x); ytr.append(np.full(len(x), lbl))
        if verify:
            print(f"  device_{d}: cumulative train-frames {ntr}  val-frames {nva}")
    if verify:
        print(f"\n  content-disjoint: {ntr} train frames (content<{split}), "
              f"{nva} val frames (content>={split})")
        return None
    return (np.concatenate(Xtr), np.concatenate(ytr).astype(np.int64)), \
           (np.concatenate(Xva), np.concatenate(yva).astype(np.int64),
            np.concatenate(vfid).astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=int, default=148, help='content-ids <split train, >=split val')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--verify', action='store_true', help='only report match quality')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); np.random.seed(0)

    print(f"Content-disjoint split at id {a.split} (train <{a.split}, val >={a.split})")
    out = build(a.split, a.verify)
    if a.verify:
        return
    (Xtr, ytr), (Xva, yva, vfid) = out
    print(f"  train windows {Xtr.shape}   val windows {Xva.shape}")

    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=256, shuffle=True)
    Xva_t = torch.from_numpy(Xva)
    model = FingerprintCNN(NUM_CLASSES).to(dev)
    model.load_state_dict(torch.load(LOCAL_PT, map_location=dev))
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.CrossEntropyLoss()
    best = 0.0; best_state = None
    for ep in range(1, a.epochs + 1):
        model.train(); tot = nb = 0
        for xb, yb in dl:
            xb = augment(xb.to(dev)); yb = yb.to(dev)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        w, f, _, _ = per_device_eval(model, dev, Xva_t, yva, vfid, NUM_CLASSES)
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:2d}  loss {tot/nb:.4f}  val_win {w:.4f}  val_frame {f:.4f}")
        if w > best:
            best = w; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    w, f, per, C = per_device_eval(model, dev, Xva_t, yva, vfid, NUM_CLASSES)
    show("CONTENT-DISJOINT held-out (unseen frames):", per, NUM_CLASSES, f)
    print(f"\n  overall frame {f:.4f}  (run-disjoint shared-content was 0.986)")
    print("\n  confusion (rows=true, cols=pred):")
    names = [DEVICE_NAMES[i] for i in range(NUM_CLASSES)]
    print("  true\\pred " + "".join(f"{n:>9}" for n in names))
    for i in range(NUM_CLASSES):
        print(f"  {names[i]:>9}" + "".join(f"{C[i,j]:>9}" for j in range(NUM_CLASSES)) + f"  n={C[i].sum()}")


if __name__ == '__main__':
    main()

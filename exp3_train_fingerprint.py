#!/usr/bin/env python3
"""
exp3_train_fingerprint.py — train the 6-device RF-fingerprint CNN from scratch
────────────────────────────────────────────────────────────────────────────

Trains FingerprintCNN (exp3_fp_model.py) on the captured frames in
  /media/.../session12/processed/frames/frames_dev{1..6}.npz
which are the validated segmentation of the raw session12/train captures
(321 frames/device over runs 1-3, frame_len=15360, pre_roll=256, fs=5 MHz).

Split: runs 1+2 = train, run 3 = held-out validation (frame-level, no window
leakage). Reports window accuracy and frame accuracy (majority vote over a
frame's windows). Saves the trained model + config to the DRIVE.

Run:
  .venv/bin/python exp3_train_fingerprint.py --epochs 25
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from exp3_fp_model import (FingerprintCNN, frame_to_windows, iq_to_input,
                           n_params, NUM_CLASSES, DEVICE_NAMES, WIN, FS)

DRIVE_PROC = "/media/cse-nghose-25/T9/Data/session12/processed"
FRAMES_DIR = os.path.join(DRIVE_PROC, "frames")
SAVE_PT    = os.path.join(DRIVE_PROC, "fingerprint_cnn_retrained.pt")
SAVE_JSON  = os.path.join(DRIVE_PROC, "fingerprint_cnn_retrained.json")
LOCAL_PT   = os.path.join(os.path.dirname(__file__), "fingerprint_cnn_retrained.pt")


def load_frames(dev_idx):
    """Return (frames[N,15360] complex64, run[N]) for device dev_idx (1-based).

    Copies the npz locally first — np.load over the exFAT mount is unreliable
    for larger archives.
    """
    src = os.path.join(FRAMES_DIR, f"frames_dev{dev_idx}.npz")
    tmp = f"/tmp/frames_dev{dev_idx}.npz"
    if not os.path.exists(tmp) or os.path.getsize(tmp) != os.path.getsize(src):
        shutil.copy(src, tmp)
    with np.load(tmp, allow_pickle=True) as z:
        return z["frames"].astype(np.complex64), z["run"].astype(int)


def build_split(hop_train, hop_val):
    """Window every frame; runs 1,2 -> train, run 3 -> val. Returns tensors
    plus the per-frame val grouping for frame-level accuracy."""
    Xtr, ytr = [], []
    Xva, yva, va_frame_id = [], [], []
    fid = 0
    counts = {}
    for d in range(1, NUM_CLASSES + 1):
        frames, run = load_frames(d)
        counts[DEVICE_NAMES[d - 1]] = int(len(frames))
        for f, r in zip(frames, run):
            is_val = (r == 3)
            w = frame_to_windows(f, hop=hop_val if is_val else hop_train)
            x = iq_to_input(w)                     # [n,2,1024] unit-RMS
            if is_val:
                Xva.append(x); yva.append(np.full(len(x), d - 1))
                va_frame_id.append(np.full(len(x), fid)); fid += 1
            else:
                Xtr.append(x); ytr.append(np.full(len(x), d - 1))
    tr = (np.concatenate(Xtr), np.concatenate(ytr))
    va = (np.concatenate(Xva), np.concatenate(yva), np.concatenate(va_frame_id))
    return tr, va, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hop-train', type=int, default=512)
    p.add_argument('--hop-val', type=int, default=WIN)
    p.add_argument('--threads', type=int, default=0, help='0 = all cores')
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0); np.random.seed(0)

    print("Loading + windowing frames ...")
    (Xtr, ytr), (Xva, yva, va_fid), counts = build_split(args.hop_train, args.hop_val)
    print(f"  frames/device: {counts}")
    print(f"  train windows: {Xtr.shape}   val windows: {Xva.shape}")

    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=args.batch, shuffle=True, num_workers=0)
    Xva_t = torch.from_numpy(Xva); yva_t = torch.from_numpy(yva)

    model = FingerprintCNN(NUM_CLASSES)
    print(f"  model params: {n_params(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss()

    def evaluate():
        model.eval()
        with torch.no_grad():
            logits = []
            for i in range(0, len(Xva_t), 1024):
                logits.append(model(Xva_t[i:i + 1024]))
            logits = torch.cat(logits)
            win_pred = logits.argmax(1).numpy()
        win_acc = (win_pred == yva).mean()
        # frame-level: majority vote per frame id
        frame_acc_num = frame_tot = 0
        for fidv in np.unique(va_fid):
            m = va_fid == fidv
            vote = np.bincount(win_pred[m], minlength=NUM_CLASSES).argmax()
            frame_acc_num += (vote == yva[m][0]); frame_tot += 1
        return win_acc, frame_acc_num / frame_tot

    best = 0.0; best_state = None
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0; nb = 0
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        wacc, facc = evaluate()
        print(f"  epoch {ep:2d}  loss {tot/nb:.4f}  val_win_acc {wacc:.4f}  "
              f"val_frame_acc {facc:.4f}")
        if wacc > best:
            best = wacc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    wacc, facc = evaluate()
    print(f"\nBEST val_win_acc {wacc:.4f}  val_frame_acc {facc:.4f}")

    cfg = dict(arch="FingerprintCNN", num_classes=NUM_CLASSES, fs=FS, win=WIN,
               preprocessing="per-window unit-RMS, input [2,1024] I/Q",
               val_window_acc=float(wacc), val_frame_acc=float(facc),
               epochs=args.epochs, params=n_params(model),
               frames_per_device=counts)
    torch.save(best_state, SAVE_PT)
    torch.save(best_state, LOCAL_PT)
    with open(SAVE_JSON, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"\nSaved model -> {SAVE_PT}")
    print(f"      config -> {SAVE_JSON}")
    print(f" local copy -> {LOCAL_PT}")


if __name__ == '__main__':
    main()

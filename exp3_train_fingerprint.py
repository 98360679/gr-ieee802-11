#!/usr/bin/env python3
"""
exp3_train_fingerprint.py — train the RF-fingerprint CNN from scratch
────────────────────────────────────────────────────────────────────────────

Trains FingerprintCNN (exp3_fp_model.py) on the captured frames in
  /media/.../session13/processed/frames/frames_dev{1..5}.npz
which are the validated segmentation of the raw session13/train captures
(frame_len=15360, pre_roll=256, fs=5 MHz; device/class count comes from
exp3_fp_model.NUM_CLASSES).

Split: runs 1+2 = train, run 3 = held-out validation (frame-level, no window
leakage). Reports window accuracy and frame accuracy (majority vote over a
frame's windows). Saves the trained model + config to the DRIVE.

Run:
  .venv/bin/python exp3_train_fingerprint.py --epochs 25
"""

import os
import sys
import math
import json
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from exp3_fp_model import (FingerprintCNN, frame_to_windows, iq_to_input,
                           n_params, NUM_CLASSES, DEVICE_NAMES, WIN, FS)

DRIVE_PROC = os.environ.get(
    "SESSION13_PROC",
    "/media/nghoselab/T9/Data/session13/processed")
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
    # Refresh the local cache from the drive when it's present; if the external
    # drive has disconnected, fall back to a valid existing /tmp cache.
    if os.path.exists(src):
        if not os.path.exists(tmp) or os.path.getsize(tmp) != os.path.getsize(src):
            shutil.copy(src, tmp)
    elif not os.path.exists(tmp):
        raise FileNotFoundError(
            f"frames for dev{dev_idx}: drive source missing ({src}) and no "
            f"cache at {tmp}. Reconnect the T9 drive.")
    with np.load(tmp, allow_pickle=True) as z:
        return z["frames"].astype(np.complex64), z["run"].astype(int)


def build_split(hop_train, hop_val, keep=None):
    """Window every frame; runs 1,2 -> train, run 3 -> val. Returns tensors
    plus the per-frame val grouping for frame-level accuracy.

    keep: 1-based device ids to include (default all NUM_CLASSES). Kept devices
    are remapped to contiguous labels 0..len(keep)-1 in the order given, so the
    model sees a clean N-class problem even when some devices are excluded.
    """
    if keep is None:
        keep = list(range(1, NUM_CLASSES + 1))
    label_of = {d: i for i, d in enumerate(keep)}     # device id -> class label
    Xtr, ytr = [], []
    Xva, yva, va_frame_id = [], [], []
    fid = 0
    counts = {}
    for d in keep:
        frames, run = load_frames(d)
        counts[DEVICE_NAMES[d - 1]] = int(len(frames))
        lbl = label_of[d]
        for f, r in zip(frames, run):
            is_val = (r == 3)
            w = frame_to_windows(f, hop=hop_val if is_val else hop_train)
            x = iq_to_input(w)                     # [n,2,1024] unit-RMS
            if is_val:
                Xva.append(x); yva.append(np.full(len(x), lbl))
                va_frame_id.append(np.full(len(x), fid)); fid += 1
            else:
                Xtr.append(x); ytr.append(np.full(len(x), lbl))
    tr = (np.concatenate(Xtr), np.concatenate(ytr))
    va = (np.concatenate(Xva), np.concatenate(yva), np.concatenate(va_frame_id))
    return tr, va, counts


def augment(xb, snr_lo=12.0, snr_hi=35.0):
    """Channel-nuisance augmentation on a batch of [B,2,1024] unit-RMS IQ.

    Simulates the cross-run variation that makes run-3 differ from runs 1-2,
    so the model learns hardware features invariant to it instead of overfitting
    run-specific channel state:
      * random global carrier phase rotation  (a receiver sees a random phase
        per capture; the device's I/Q-imbalance fingerprint must survive it)
      * additive complex Gaussian noise at a random SNR in [snr_lo, snr_hi] dB
    Power/amplitude is already removed by unit-RMS, so we don't touch it.
    """
    B = xb.shape[0]
    dev = xb.device
    theta = torch.rand(B, 1, device=dev) * (2 * math.pi)
    c, s = torch.cos(theta), torch.sin(theta)
    I, Q = xb[:, 0], xb[:, 1]
    xb = torch.stack([I * c - Q * s, I * s + Q * c], dim=1)
    snr_db = torch.empty(B, 1, 1, device=dev).uniform_(snr_lo, snr_hi)
    npow = 10 ** (-snr_db / 10)                       # noise power (sig power ~1)
    xb = xb + torch.randn_like(xb) * torch.sqrt(npow / 2)
    return xb


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hop-train', type=int, default=512)
    p.add_argument('--hop-val', type=int, default=WIN)
    p.add_argument('--threads', type=int, default=0, help='0 = all cores')
    p.add_argument('--exclude', type=str, default='',
                   help='comma-separated 1-based device ids to drop, e.g. "2"')
    p.add_argument('--aug', action='store_true',
                   help='channel-nuisance augmentation (phase rot + noise)')
    p.add_argument('--wd', type=float, default=0.0, help='weight decay')
    p.add_argument('--label-smooth', type=float, default=0.0)
    p.add_argument('--device', default='auto', help='auto|cpu|cuda')
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0); np.random.seed(0)

    dev = ('cuda' if (args.device == 'auto' and torch.cuda.is_available())
           else args.device if args.device != 'auto' else 'cpu')
    print(f"device: {dev}   aug: {args.aug}   wd: {args.wd}   "
          f"label_smooth: {args.label_smooth}")

    drop = {int(x) for x in args.exclude.split(',') if x.strip()}
    keep = [d for d in range(1, NUM_CLASSES + 1) if d not in drop]
    n_classes = len(keep)
    if drop:
        print(f"Excluding devices {sorted(drop)} -> {n_classes}-class problem "
              f"over {[DEVICE_NAMES[d-1] for d in keep]}")

    print("Loading + windowing frames ...")
    (Xtr, ytr), (Xva, yva, va_fid), counts = build_split(
        args.hop_train, args.hop_val, keep)
    print(f"  frames/device: {counts}")
    print(f"  train windows: {Xtr.shape}   val windows: {Xva.shape}")

    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=args.batch, shuffle=True, num_workers=0)
    Xva_t = torch.from_numpy(Xva); yva_t = torch.from_numpy(yva)

    model = FingerprintCNN(n_classes).to(dev)
    print(f"  model params: {n_params(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

    def evaluate():
        model.eval()
        with torch.no_grad():
            logits = []
            for i in range(0, len(Xva_t), 1024):
                logits.append(model(Xva_t[i:i + 1024].to(dev)).cpu())
            logits = torch.cat(logits)
            win_pred = logits.argmax(1).numpy()
        win_acc = (win_pred == yva).mean()
        # frame-level: majority vote per frame id
        frame_acc_num = frame_tot = 0
        for fidv in np.unique(va_fid):
            m = va_fid == fidv
            vote = np.bincount(win_pred[m], minlength=n_classes).argmax()
            frame_acc_num += (vote == yva[m][0]); frame_tot += 1
        return win_acc, frame_acc_num / frame_tot

    best = 0.0; best_state = None
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0; nb = 0
        for xb, yb in dl:
            xb = xb.to(dev); yb = yb.to(dev)
            if args.aug:
                xb = augment(xb)
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
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    wacc, facc = evaluate()
    print(f"\nBEST val_win_acc {wacc:.4f}  val_frame_acc {facc:.4f}")

    cfg = dict(arch="FingerprintCNN", num_classes=n_classes, fs=FS, win=WIN,
               preprocessing="per-window unit-RMS, input [2,1024] I/Q",
               val_window_acc=float(wacc), val_frame_acc=float(facc),
               epochs=args.epochs, params=n_params(model),
               kept_devices=[DEVICE_NAMES[d - 1] for d in keep],
               excluded_devices=[DEVICE_NAMES[d - 1] for d in sorted(drop)],
               class_labels={DEVICE_NAMES[d - 1]: i for i, d in enumerate(keep)},
               frames_per_device=counts)

    # Tag output filenames when devices are excluded so the full 5-device model
    # (which the attack scripts load) is never silently overwritten.
    tag = f"_{n_classes}dev" if drop else ""
    def _tagged(path):
        base, ext = os.path.splitext(path)
        return base + tag + ext
    save_pt, save_json, local_pt = _tagged(SAVE_PT), _tagged(SAVE_JSON), _tagged(LOCAL_PT)

    # Always write the local copy first so a missing/unmounted drive can't
    # discard the result after a full training run.
    torch.save(best_state, local_pt)
    print(f"\n local copy -> {local_pt}")
    try:
        os.makedirs(os.path.dirname(save_pt), exist_ok=True)
        torch.save(best_state, save_pt)
        with open(save_json, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f"Saved model -> {save_pt}")
        print(f"      config -> {save_json}")
    except OSError as e:
        local_json = os.path.splitext(local_pt)[0] + ".json"
        with open(local_json, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f"Drive save skipped ({e}); config -> {local_json}")


if __name__ == '__main__':
    main()

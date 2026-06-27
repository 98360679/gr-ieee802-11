#!/usr/bin/env python3
"""
exp3_finetune.py — fine-tune the 6-class fingerprint model onto TODAY's channel
────────────────────────────────────────────────────────────────────────────────
The committed model degraded under the 2026-06-27 channel (rebaseline: frame
0.331 vs 0.934; devices 1/2/4/5 collapsed to device_3, only 3 & 6 survived).
This loads that model as the init and fine-tunes it on a fresh, single-clean-run-
per-device capture set, recovering separation under the new channel.

Single run => split by FRAME (windows from one frame stay on one side, no leakage)
but there is NO independent held-out run, so the val number reads OPTIMISTIC.
Treat it as "did fine-tuning recover separation", not a publishable accuracy.

Reuses the exact Stage-1 extraction + model windowing, so numbers are comparable
to training / rebaseline. Writes a NEW model file (does not overwrite the
canonical fingerprint_cnn_retrained.pt until you validate the fine-tune).

Run:
  python3 exp3_finetune.py            # defaults: root=train/6_27_2026, 20 epochs
"""
import os
import json
import argparse
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from exp3_fp_model import (FingerprintCNN, frame_to_windows, iq_to_input,
                           n_params, NUM_CLASSES, DEVICE_NAMES, WIN, FS)
from exp3_extract_frames import extract_frames_for_file
from exp3_train_fingerprint import LOCAL_PT, augment

SCRATCH = ("/tmp/claude-1001/-home-nghoselab/"
           "7e6fdecd-9fba-4a89-857f-b1c39ddcc651/scratchpad")
DEF_ROOT = "/media/nghoselab/T9/Data/session13/train/6_27_2026"


def find_capture(root, d):
    bins = sorted(glob.glob(os.path.join(root, f"device_{d}", "*.bin")))
    return bins[0] if bins else None


def build_dataset(root, hop_train, val_frac, cache, rebuild):
    if os.path.exists(cache) and not rebuild:
        z = np.load(cache)
        print(f"  (loaded windowed cache {cache})")
        return (z['Xtr'], z['ytr']), (z['Xva'], z['yva'], z['vfid']), \
               json.loads(str(z['counts']))
    Xtr, ytr, Xva, yva, vfid = [], [], [], [], []
    fid = 0
    counts = {}
    for d in range(1, NUM_CLASSES + 1):
        cap = find_capture(root, d)
        if cap is None:
            print(f"  device_{d}: no .bin, skipping")
            continue
        frames = extract_frames_for_file(cap)[0]      # median floor (clean cap)
        counts[DEVICE_NAMES[d - 1]] = int(len(frames))
        lbl = d - 1                                   # device_d -> class d-1
        # frame-level split: every k-th frame -> val
        k = max(2, int(round(1.0 / val_frac)))
        for i, f in enumerate(frames):
            is_val = (i % k == 0)
            w = frame_to_windows(f, hop=WIN if is_val else hop_train)
            x = iq_to_input(w)                        # [n,2,1024] unit-RMS f32
            if is_val:
                Xva.append(x); yva.append(np.full(len(x), lbl))
                vfid.append(np.full(len(x), fid)); fid += 1
            else:
                Xtr.append(x); ytr.append(np.full(len(x), lbl))
        print(f"  device_{d}: {len(frames)} frames  ({os.path.basename(cap)})")
    tr = (np.concatenate(Xtr), np.concatenate(ytr).astype(np.int64))
    va = (np.concatenate(Xva), np.concatenate(yva).astype(np.int64),
          np.concatenate(vfid).astype(np.int64))
    np.savez(cache, Xtr=tr[0], ytr=tr[1], Xva=va[0], yva=va[1], vfid=va[2],
             counts=json.dumps(counts))
    return tr, va, counts


def per_device_eval(model, dev, Xva_t, yva, vfid, n_classes):
    model.eval()
    with torch.no_grad():
        logits = []
        for i in range(0, len(Xva_t), 1024):
            logits.append(model(Xva_t[i:i+1024].to(dev)).cpu())
        win_pred = torch.cat(logits).argmax(1).numpy()
    win_acc = float((win_pred == yva).mean())
    C = np.zeros((n_classes, n_classes), int)
    per = {i: [0, 0] for i in range(n_classes)}      # label -> [ok, total]
    for fv in np.unique(vfid):
        m = vfid == fv
        true = int(yva[m][0])
        vote = int(np.bincount(win_pred[m], minlength=n_classes).argmax())
        C[true, vote] += 1
        per[true][1] += 1
        per[true][0] += (vote == true)
    frame_acc = sum(p[0] for p in per.values()) / max(1, sum(p[1] for p in per.values()))
    return win_acc, frame_acc, per, C


def show(tag, per, n_classes, frame_acc):
    print(f"\n  {tag}  overall frame {frame_acc:.3f}")
    for i in range(n_classes):
        ok, tot = per[i]
        print(f"    {DEVICE_NAMES[i]:>9}: frame {ok/max(1,tot):.3f}  ({ok}/{tot})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=DEF_ROOT)
    p.add_argument('--init', default=LOCAL_PT, help='model to fine-tune from')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=2e-4, help='low LR for fine-tune')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--hop-train', type=int, default=384)
    p.add_argument('--val-frac', type=float, default=0.2)
    p.add_argument('--no-aug', action='store_true',
                   help='disable channel-nuisance augmentation (on by default)')
    p.add_argument('--snr-lo', type=float, default=20.0)
    p.add_argument('--snr-hi', type=float, default=40.0)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--tag', default='ft20260627')
    p.add_argument('--rebuild', action='store_true', help='rebuild window cache')
    a = p.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_classes = NUM_CLASSES
    torch.manual_seed(0); np.random.seed(0)
    print(f"device {dev}   init {os.path.basename(a.init)}   "
          f"aug {not a.no_aug}   lr {a.lr}   epochs {a.epochs}")

    cache = os.path.join(SCRATCH, "ft_windows_6_27.npz")
    print("Loading + windowing today's captures ...")
    (Xtr, ytr), (Xva, yva, vfid), counts = build_dataset(
        a.root, a.hop_train, a.val_frac, cache, a.rebuild)
    print(f"  frames/device: {counts}")
    print(f"  train windows {Xtr.shape}   val windows {Xva.shape}")

    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=a.batch, shuffle=True)
    Xva_t = torch.from_numpy(Xva)

    model = FingerprintCNN(n_classes).to(dev)
    state = torch.load(a.init, map_location=dev)
    model.load_state_dict(state)
    print(f"  loaded init ({n_params(model):,} params)")

    # baseline (stale model on today's val split) — should mirror the rebaseline
    w0, f0, per0, C0 = per_device_eval(model, dev, Xva_t, yva, vfid, n_classes)
    show("STALE init on today's val:", per0, n_classes, f0)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.CrossEntropyLoss()

    best = 0.0; best_state = None
    for ep in range(1, a.epochs + 1):
        model.train(); tot = nb = 0
        for xb, yb in dl:
            xb = xb.to(dev); yb = yb.to(dev)
            if not a.no_aug:
                xb = augment(xb, a.snr_lo, a.snr_hi)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        wacc, facc, _, _ = per_device_eval(model, dev, Xva_t, yva, vfid, n_classes)
        print(f"  epoch {ep:2d}  loss {tot/nb:.4f}  val_win {wacc:.4f}  "
              f"val_frame {facc:.4f}")
        if wacc > best:
            best = wacc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    wF, fF, perF, CF = per_device_eval(model, dev, Xva_t, yva, vfid, n_classes)
    show("FINE-TUNED on today's val:", perF, n_classes, fF)
    print(f"\n  RECOVERY: overall frame {f0:.3f} -> {fF:.3f}  ({fF-f0:+.3f})")
    print("\n  FINE-TUNED frame confusion (rows=true, cols=pred):")
    names = [DEVICE_NAMES[i] for i in range(n_classes)]
    print("  true\\pred " + "".join(f"{n:>9}" for n in names))
    for i in range(n_classes):
        print(f"  {names[i]:>9}" + "".join(f"{CF[i,j]:>9}" for j in range(n_classes))
              + f"   n={CF[i].sum()}")

    # save NEW files (do not clobber the canonical model)
    out_pt = os.path.join(os.path.dirname(__file__),
                          f"fingerprint_cnn_{a.tag}.pt")
    out_json = os.path.splitext(out_pt)[0] + ".json"
    torch.save(best_state, out_pt)
    cfg = dict(arch="FingerprintCNN", num_classes=n_classes, fs=FS, win=WIN,
               preprocessing="per-window unit-RMS, input [2,1024] I/Q",
               finetuned_from=os.path.basename(a.init),
               finetune_root=a.root, val_window_acc=float(wF),
               val_frame_acc=float(fF), val_note="single-run frame split; optimistic",
               stale_val_frame_acc=float(f0), epochs=a.epochs, lr=a.lr,
               aug=(not a.no_aug), params=n_params(model),
               class_labels={DEVICE_NAMES[i]: i for i in range(n_classes)},
               frames_per_device=counts)
    json.dump(cfg, open(out_json, 'w'), indent=2)
    print(f"\n  saved -> {out_pt}\n        -> {out_json}")
    try:
        drive = "/media/nghoselab/T9/Data/session13/processed"
        torch.save(best_state, os.path.join(drive, os.path.basename(out_pt)))
        json.dump(cfg, open(os.path.join(drive, os.path.basename(out_json)), 'w'),
                  indent=2)
        print(f"  drive copy -> {drive}/{os.path.basename(out_pt)}")
    except OSError as e:
        print(f"  (drive copy skipped: {e})")


if __name__ == '__main__':
    main()

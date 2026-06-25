#!/usr/bin/env python3
"""exp3_confusion.py — confusion matrix of a trained fingerprint CNN.

Evaluates a saved model on the held-out run-3 validation split (same split
build_split() uses) and reports per-device accuracy + window/frame confusion
matrices, to see WHICH devices get confused for which.

  # 5-device model
  python3 exp3_confusion.py
  # 4-device model (device_2 excluded)
  python3 exp3_confusion.py --exclude 2 --model fingerprint_cnn_retrained_4dev.pt
"""
import os
import argparse
import numpy as np
import torch

from exp3_fp_model import FingerprintCNN, NUM_CLASSES, DEVICE_NAMES
from exp3_train_fingerprint import build_split, WIN, LOCAL_PT

p = argparse.ArgumentParser()
p.add_argument('--exclude', type=str, default='',
               help='comma-separated 1-based device ids to drop, e.g. "2"')
p.add_argument('--model', default=LOCAL_PT, help='path to the .pt to evaluate')
args = p.parse_args()

drop = {int(x) for x in args.exclude.split(',') if x.strip()}
keep = [d for d in range(1, NUM_CLASSES + 1) if d not in drop]
names = [DEVICE_NAMES[d - 1] for d in keep]          # column/row order
n_classes = len(keep)

torch.manual_seed(0); np.random.seed(0)
print(f"Model: {args.model}")
print(f"Classes ({n_classes}): {names}")
print("Building run-3 validation split ...")
(_Xtr, _ytr), (Xva, yva, va_fid), counts = build_split(
    hop_train=512, hop_val=WIN, keep=keep)
print(f"  frames/device (all runs): {counts}")
print(f"  val windows: {Xva.shape}")

model = FingerprintCNN(n_classes)
model.load_state_dict(torch.load(args.model, map_location="cpu"))
model.eval()

Xva_t = torch.from_numpy(Xva)
with torch.no_grad():
    logits = torch.cat([model(Xva_t[i:i + 1024]) for i in range(0, len(Xva_t), 1024)])
win_pred = logits.argmax(1).numpy()

win_acc = (win_pred == yva).mean()
Cw = np.zeros((n_classes, n_classes), int)
for t, pr in zip(yva, win_pred):
    Cw[t, pr] += 1

frame_true, frame_pred = [], []
for fidv in np.unique(va_fid):
    m = va_fid == fidv
    frame_true.append(yva[m][0])
    frame_pred.append(np.bincount(win_pred[m], minlength=n_classes).argmax())
frame_true = np.array(frame_true); frame_pred = np.array(frame_pred)
frame_acc = (frame_true == frame_pred).mean()
Cf = np.zeros((n_classes, n_classes), int)
for t, pr in zip(frame_true, frame_pred):
    Cf[t, pr] += 1


def show(C, title):
    print(f"\n=== {title} confusion (rows=true, cols=pred) ===")
    print("true\\pred " + "".join(f"{n:>8}" for n in names) + "   acc   n")
    for i in range(n_classes):
        row = "".join(f"{C[i,j]:>8}" for j in range(n_classes))
        n = C[i].sum(); acc = C[i, i] / n if n else 0.0
        print(f"{names[i]:>9}{row}  {acc:5.2f} {n:>5}")
    off = [(i, j, C[i, j]) for i in range(n_classes)
           for j in range(n_classes) if i != j and C[i, j] > 0]
    off.sort(key=lambda t: -t[2])
    print("  top confusions: " +
          ", ".join(f"{names[i]}->{names[j]}:{v}" for i, j, v in off[:5]))


print(f"\nval window acc {win_acc:.4f}   val frame acc {frame_acc:.4f}")
show(Cw, "WINDOW")
show(Cf, "FRAME")

#!/usr/bin/env python3
"""
exp3_plot_confusion.py — confusion-matrix heatmap figure for a fingerprint model
────────────────────────────────────────────────────────────────────────────────
Evaluates a model on clean per-device captures (frame-level majority vote) and
saves an annotated confusion-matrix PNG (+ prints text + per-device/overall acc).
Default uses a held-out frame split (every k-th frame) for an honest number;
--all uses every frame.

  python3 exp3_plot_confusion.py --model fingerprint_cnn_ft20260628.pt \
      --root /media/.../train/6_28_2026 --out confusion_ft20260628.png
"""
import os, glob, re, argparse, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp3_make_perturbation import load_fp_model, predict_frame
from exp3_extract_frames import extract_frames_for_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="fingerprint_cnn_ft20260628.pt")
    p.add_argument("--root", default="/media/nghoselab/T9/Data/session13/train/6_28_2026")
    p.add_argument("--out", default="confusion_ft20260628.png")
    p.add_argument("--val-frac", type=float, default=0.2, help="held-out fraction (every k-th frame)")
    p.add_argument("--all", action="store_true", help="use every frame (not just held-out)")
    a = p.parse_args()

    model, nc, n2i, i2n = load_fp_model(a.model)
    names = [i2n[i] for i in range(nc)]
    k = max(2, int(round(1.0 / a.val_frac)))
    C = np.zeros((nc, nc), int)
    for d in range(1, nc + 1):
        name = f"device_{d}"
        if name not in n2i:
            continue
        bins = sorted(glob.glob(os.path.join(a.root, name, "*.bin")))
        if not bins:
            print(f"  {name}: no capture"); continue
        frames = extract_frames_for_file(bins[0])[0]
        true = n2i[name]
        for i, fr in enumerate(frames):
            if not a.all and i % k != 0:        # held-out frames only
                continue
            pred, _ = predict_frame(model, fr, nc)
            C[true, pred] += 1
        n = C[true].sum()
        print(f"  {name}: {n} frames  acc {C[true,true]/max(1,n):.3f}")

    tot = C.sum(); acc = np.trace(C) / max(1, tot)
    split = "all frames" if a.all else f"held-out 1/{k} frames"
    print(f"\nOVERALL frame acc {acc:.4f}  ({tot} frames, {split})")

    # ── heatmap ──
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    Cn = C / np.clip(C.sum(1, keepdims=True), 1, None)     # row-normalized
    im = ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
    ax.set_xticklabels([n.replace("device_", "d") for n in names])
    ax.set_yticklabels([n.replace("device_", "d") for n in names])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{os.path.basename(a.model)}\nframe acc {acc:.3f} ({split})")
    for i in range(nc):
        for j in range(nc):
            if C[i, j]:
                ax.text(j, i, C[i, j], ha="center", va="center",
                        color="white" if Cn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalized")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"saved figure -> {a.out}")


if __name__ == "__main__":
    main()

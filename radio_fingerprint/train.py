#!/usr/bin/env python3
"""
Stage 3 — Train one or all of the 4 configurations and evaluate on run_3.

  config = {domain} x {model}
    domain : time | freq      (freq = fftshift(FFT(window)))
    model  : cnn1d | cvnn

Per config we report window-level AND frame-level (majority vote over the
burst a window came from) accuracy on the held-out run_3 test set, save a
confusion matrix and training curves, and dump metrics to JSON.

Run:
  python3 train.py --config time_cnn1d
  python3 train.py --all
"""
import os, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import config as C
from models import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIGS = ["time_cnn1d", "time_cvnn", "freq_cnn1d", "freq_cvnn"]


# ─────────────────────────────── data ───────────────────────────────────
def load_features(domain):
    d = np.load(C.DATASET, allow_pickle=True)
    names = list(d["label_names"])

    def feat(X):
        X = X.astype(np.complex64)
        if domain == "freq":
            Z = np.fft.fftshift(np.fft.fft(X, axis=1), axes=1) / np.sqrt(X.shape[1])
            return Z.astype(np.complex64)
        return X
    splits = {}
    for s in ("train", "val", "test"):
        splits[s] = (feat(d[f"X_{s}"]), d[f"y_{s}"].astype(np.int64), d[f"g_{s}"].astype(np.int64))
    return splits, names


def to_model_input(Xc, kind):
    """Xc: (B,L) complex tensor on device -> model input tensor."""
    if kind == "cnn1d":
        return torch.stack([Xc.real, Xc.imag], dim=1)      # (B,2,L)
    return Xc.unsqueeze(1)                                  # (B,1,L) complex


def batches(X, y, bs, shuffle, gen=None):
    n = len(X)
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield X[j], y[j]


# ─────────────────────────────── train ──────────────────────────────────
def train_one(cfg, epochs=30, bs=256, lr=2e-3, wd=1e-4, width=None):
    domain, kind = cfg.split("_")
    print(f"\n=== {cfg}  (domain={domain}, model={kind}, dev={DEVICE}) ===")
    splits, names = load_features(domain)
    n_classes = len(names)
    L = splits["train"][0].shape[1]

    # move whole splits to GPU as complex tensors (small enough)
    def gpu(arr):
        return torch.from_numpy(arr).to(DEVICE)
    Xtr = gpu(splits["train"][0]); ytr = gpu(splits["train"][1])
    Xva = gpu(splits["val"][0]);   yva = gpu(splits["val"][1])
    Xte = gpu(splits["test"][0]);  yte = gpu(splits["test"][1]); gte = splits["test"][2]

    model = build_model(kind, n_classes, L).to(DEVICE)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    gen = torch.Generator().manual_seed(C.SEED)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_va, best_state = 0.0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for xb, yb in batches(Xtr, ytr, bs, True, gen):
            opt.zero_grad()
            out = model(to_model_input(xb, kind))
            loss = lossf(out, yb)
            loss.backward(); opt.step()
            tl += loss.item() * len(yb); tc += (out.argmax(1) == yb).sum().item(); tn += len(yb)
        sched.step()

        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in batches(Xva, yva, bs, False):
                out = model(to_model_input(xb, kind))
                vl += lossf(out, yb).item() * len(yb)
                vc += (out.argmax(1) == yb).sum().item(); vn += len(yb)
        tr_acc, va_acc = tc / tn, vc / vn
        hist["train_loss"].append(tl / tn); hist["val_loss"].append(vl / vn)
        hist["train_acc"].append(tr_acc);  hist["val_acc"].append(va_acc)
        if va_acc >= best_va:
            best_va = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:3d}  tr_loss {tl/tn:.3f} acc {tr_acc:.3f} | "
                  f"val_loss {vl/vn:.3f} acc {va_acc:.3f}")

    model.load_state_dict(best_state)

    # ── test: window-level predictions ──
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, yb in batches(Xte, yte, bs, False):
            out = model(to_model_input(xb, kind))
            preds.append(out.argmax(1).cpu().numpy())
    win_pred = np.concatenate(preds)
    win_true = yte.cpu().numpy()
    win_acc = float((win_pred == win_true).mean())

    # ── frame-level: majority vote of windows sharing a burst id ──
    fr_true, fr_pred = [], []
    for g in np.unique(gte):
        m = gte == g
        fr_true.append(int(np.bincount(win_true[m]).argmax()))
        fr_pred.append(int(np.bincount(win_pred[m]).argmax()))
    fr_true, fr_pred = np.array(fr_true), np.array(fr_pred)
    fr_acc = float((fr_pred == fr_true).mean())

    print(f"  TEST  window-acc {win_acc:.4f}   frame-acc {fr_acc:.4f}   "
          f"(best val {best_va:.4f}, {nparam/1e3:.0f}k params, {time.time()-t0:.0f}s)")

    # ── save artifacts ──
    torch.save(best_state, os.path.join(C.OUT_DIR, f"{cfg}_best.pt"))
    cm_win = confusion_matrix(win_true, win_pred, labels=range(n_classes))
    cm_fr = confusion_matrix(fr_true, fr_pred, labels=range(n_classes))
    metrics = {
        "config": cfg, "domain": domain, "model": kind,
        "window_acc": win_acc, "frame_acc": fr_acc, "best_val_acc": best_va,
        "n_params": int(nparam), "epochs": epochs,
        "n_test_windows": int(len(win_true)), "n_test_frames": int(len(fr_true)),
        "history": hist,
        "report": classification_report(win_true, win_pred, target_names=names,
                                        labels=range(n_classes), output_dict=True, zero_division=0),
    }
    with open(os.path.join(C.OUT_DIR, f"{cfg}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion(cm_win, cm_fr, names, cfg, win_acc, fr_acc)
    plot_training(hist, cfg)
    return metrics


# ─────────────────────────────── plots ──────────────────────────────────
def _draw_cm(ax, cm, names, title):
    cmn = cm.astype(float) / cm.sum(1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cmn[i,j]*100:.0f}%\n{cm[i,j]}",
                    ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    return im


def plot_confusion(cm_win, cm_fr, names, cfg, win_acc, fr_acc):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    _draw_cm(ax[0], cm_win, names, f"Window-level  (acc {win_acc*100:.1f}%)")
    _draw_cm(ax[1], cm_fr, names, f"Frame-level vote  (acc {fr_acc*100:.1f}%)")
    fig.suptitle(f"Confusion matrix — {cfg}", fontsize=13, y=1.00)
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, f"{cfg}_confusion.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")


def plot_training(hist, cfg):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["train_loss"], label="train"); ax[0].plot(ep, hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, hist["train_acc"], label="train"); ax[1].plot(ep, hist["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(f"Training curves — {cfg}")
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, f"{cfg}_training.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)


def comparison_plot(all_metrics):
    cfgs = [m["config"] for m in all_metrics]
    wins = [m["window_acc"] * 100 for m in all_metrics]
    frs = [m["frame_acc"] * 100 for m in all_metrics]
    x = np.arange(len(cfgs)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, wins, w, label="window-level")
    b2 = ax.bar(x + w/2, frs, w, label="frame-level")
    ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=20)
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title(f"{C.DAY} device fingerprinting — 4 configurations")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f"{b.get_height():.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, "comparison.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)

    targets = CONFIGS if a.all else [a.config]
    results = []
    for cfg in targets:
        results.append(train_one(cfg, epochs=a.epochs, bs=a.batch))
    if a.all:
        comparison_plot(results)
        with open(os.path.join(C.OUT_DIR, "summary.json"), "w") as f:
            json.dump([{k: m[k] for k in ("config", "window_acc", "frame_acc",
                       "best_val_acc", "n_params")} for m in results], f, indent=2)
        print("\n=== SUMMARY ===")
        for m in results:
            print(f"  {m['config']:14s}  window {m['window_acc']*100:5.1f}%  "
                  f"frame {m['frame_acc']*100:5.1f}%")


if __name__ == "__main__":
    main()

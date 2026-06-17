#!/usr/bin/env python3
"""
exp3_analyze_d3d5.py — dig into the device-3 / device-5 confusion
──────────────────────────────────────────────────────────────────
Reports, on the held-out run_3 test set:
  (1) per-device window & frame-vote accuracy + precision/recall
  (2) where each device's misclassified windows actually go
  (3) a physical-layer carrier-frequency-offset (CFO) estimate per device,
      from the short-training-field repetition in each frame's preamble
  (4) the trained model's embedding-space geometry: per-device centroids and
      pairwise distances (which devices look most alike to the model)
"""
import os, glob, argparse
import numpy as np
import torch
from exp3_train_fingerprint import CNN1D

DATA   = '/media/nghoselab/T9/Data/session12/processed/dataset_win1024.npz'
CKPT   = '/media/nghoselab/T9/Data/session12/processed/fingerprint_cnn.pt'
FRAMES = '/media/nghoselab/T9/Data/session12/processed/frames'
FS     = 5e6
PRE_ROLL = 256


# ── (3) CFO estimate from the short-training-field repetition ──────────
def estimate_cfo(frame, onset=PRE_ROLL, span=160, lag=16):
    """Coarse CFO via autocorrelation at the STF repetition lag (16 samples).

    The 802.11 L-STF is 10 repetitions of a 16-sample sequence at the start of
    the frame. sum(x[n]*conj(x[n+16])) over the STF has phase 2*pi*CFO*16/fs.
    We also return the normalized autocorr magnitude (0..1) as a confidence:
    near 1 => clean periodic STF present, low => estimate is unreliable.
    """
    seg = frame[onset:onset + span]
    a, b = seg[:-lag], seg[lag:]
    ac = np.sum(a * np.conj(b))
    norm = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2)) + 1e-12
    conf = np.abs(ac) / norm
    cfo = np.angle(ac) / (2 * np.pi * lag / FS)
    return cfo, conf


def per_device_cfo():
    print("\n(3) Physical-layer CFO per device (preamble STF lag-16 autocorr)")
    print("    using only frames with clean STF (autocorr confidence > 0.6)")
    print("    dev   n_clean   CFO_median(Hz)  CFO_IQR(Hz)")
    stats = {}
    for fp in sorted(glob.glob(os.path.join(FRAMES, 'frames_dev*.npz'))):
        z = np.load(fp)
        dev = int(z['device']); F = z['frames']
        cfos, confs = [], []
        for i in range(len(F)):
            c, cf = estimate_cfo(F[i])
            cfos.append(c); confs.append(cf)
        cfos = np.array(cfos); confs = np.array(confs)
        keep = confs > 0.6
        cc = cfos[keep]
        stats[dev] = cc
        if len(cc):
            iqr = np.percentile(cc, 75) - np.percentile(cc, 25)
            print(f"    d{dev}   {keep.sum():4d}/{len(F)}   {np.median(cc):12.1f}   {iqr:10.1f}")
        else:
            print(f"    d{dev}   0/{len(F)}   (no clean STF found)")
    if 3 in stats and 5 in stats and len(stats[3]) and len(stats[5]):
        m3, m5 = np.median(stats[3]), np.median(stats[5])
        spread = 0.5 * ((np.percentile(stats[3],75)-np.percentile(stats[3],25)) +
                        (np.percentile(stats[5],75)-np.percentile(stats[5],25)))
        print(f"    d3 vs d5 CFO median gap = {abs(m3-m5):.0f} Hz  vs avg IQR "
              f"{spread:.0f} Hz  -> separation {abs(m3-m5)/(spread+1e-9):.2f}")
    return stats


# ── (4) model embedding geometry ───────────────────────────────────────
@torch.no_grad()
def embeddings(model, X, device, bs=256):
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(device)
        z = model.pool(model.features(xb)).flatten(1)   # penultimate 128-d
        outs.append(z.cpu())
    return torch.cat(outs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    z = np.load(DATA)
    Xte = torch.from_numpy(z['X_test']); yte = z['y_test']; mte = z['m_test']
    names = [str(s) for s in z['label_names']]
    nC = len(names)

    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = CNN1D(n_classes=nC).to(device)
    model.load_state_dict(ck['model'])

    # window predictions  (eval mode: use BN running stats, disable dropout)
    model.eval()
    with torch.no_grad():
        logits = []
        for i in range(0, len(Xte), 256):
            logits.append(model(Xte[i:i+256].to(device)).cpu())
        logits = torch.cat(logits)
    prob = torch.softmax(logits, 1).numpy()
    pred = prob.argmax(1)

    # ── (1) per-device window precision/recall + frame-vote ──
    print("(1) Per-device accuracy on held-out run_3 test set")
    print("    WINDOW level:")
    print("    dev   support  recall   precision   f1")
    for c in range(nC):
        tp = np.sum((pred == c) & (yte == c))
        fn = np.sum((pred != c) & (yte == c))
        fp = np.sum((pred == c) & (yte != c))
        rec = tp / (tp + fn + 1e-9)
        prc = tp / (tp + fp + 1e-9)
        f1  = 2 * rec * prc / (rec + prc + 1e-9)
        print(f"    d{c+1}   {tp+fn:6d}   {rec:.3f}    {prc:.3f}     {f1:.3f}")
    print(f"    overall window acc = {(pred==yte).mean():.4f}")

    # frame-vote
    from collections import defaultdict
    acc = defaultdict(lambda: np.zeros(nC)); truth = {}
    for k, pr, lab in zip([tuple(r) for r in mte[:, :3].tolist()], prob, yte):
        acc[k] += pr; truth[k] = lab
    fpred = {k: acc[k].argmax() for k in acc}
    print("\n    FRAME-VOTE level:")
    print("    dev   frames   correct   acc")
    for c in range(nC):
        ks = [k for k in truth if truth[k] == c]
        corr = sum(int(fpred[k] == c) for k in ks)
        print(f"    d{c+1}   {len(ks):6d}   {corr:6d}    {corr/len(ks):.3f}")
    fr_corr = sum(int(fpred[k] == truth[k]) for k in truth)
    print(f"    overall frame-vote acc = {fr_corr/len(truth):.4f}")

    # ── (2) where do d3 and d5 errors go? ──
    print("\n(2) Misclassification routing (window level)")
    for c in (2, 4):   # device 3 (idx2), device 5 (idx4)
        mask = yte == c
        dest = np.bincount(pred[mask], minlength=nC)
        order = np.argsort(dest)[::-1]
        s = "  ".join(f"d{o+1}:{dest[o]}" for o in order if dest[o] > 0)
        print(f"    true d{c+1} ({mask.sum()} win) -> {s}")

    # ── (3) physical CFO ──
    per_device_cfo()

    # ── (4) embedding geometry ──
    print("\n(4) Model embedding geometry (test windows, 128-d penultimate)")
    emb = embeddings(model, Xte, device)
    cent = np.stack([emb[yte == c].mean(0) for c in range(nC)])
    # normalize by avg within-class spread for an interpretable distance
    D = np.zeros((nC, nC))
    for a in range(nC):
        for b in range(nC):
            D[a, b] = np.linalg.norm(cent[a] - cent[b])
    print("    pairwise centroid distance (rows/cols = d1..d6):")
    print("         " + " ".join(f"d{b+1:>5}" for b in range(nC)))
    for a in range(nC):
        print(f"    d{a+1}  " + " ".join(f"{D[a,b]:6.2f}" for b in range(nC)))
    # nearest neighbor per device
    print("\n    nearest other device in embedding space:")
    for a in range(nC):
        d = D[a].copy(); d[a] = np.inf
        b = d.argmin()
        print(f"    d{a+1} closest to d{b+1}  (dist {D[a,b]:.2f})")


if __name__ == '__main__':
    main()

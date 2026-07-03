#!/usr/bin/env python3
"""
exp3_cvnn.py — complex-valued CNN (CVNN) fingerprinter for experiment_3.

PyTorch port of the experiment_1/2 complex primitives (ComplexConv1D = 4 real convs
rr/ri/ir/ii; ComplexBatchNorm; ComplexAvgPool; CReLU; ComplexDropout), built at exp3's
scale: complex windows of length WIN=1024, 6-class output. Complex is carried as an
(re, im) pair of real tensors, so torch autograd works end-to-end -> the PGD/FGSM/EOT
attack tooling can differentiate the loss w.r.t. the input IQ.

Trains on the fresh varied 3-run enrollment (leakage-safe run-3 holdout).

  python3 exp3_cvnn.py --root .../train/7_02_2026 --val-run 3 --epochs 30
"""
import os, re, glob, json, math, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from exp3_fp_model import frame_to_windows, NUM_CLASSES, DEVICE_NAMES, WIN, FS
from exp3_extract_frames import extract_frames_for_file


def drift_augment(w, cfo_max=3000.0, snr_lo=8.0, snr_hi=30.0, tmax=6):
    """Mimic inter-run drift on complex windows w [B, L] so the model learns drift-INVARIANT
    fingerprint features: random global phase, small CFO ramp, circular time-shift, AWGN."""
    B, L = w.shape
    dev = w.device
    n = torch.arange(L, device=dev, dtype=torch.float32)
    df = (torch.rand(B, 1, device=dev) * 2 - 1) * cfo_max                    # Hz
    phi = (torch.rand(B, 1, device=dev) * 2 - 1) * math.pi                   # full phase
    w = w * torch.exp(1j * (2 * math.pi * df / FS * n + phi))
    sh = torch.randint(-tmax, tmax + 1, (B, 1), device=dev)                  # circular shift
    idx = (torch.arange(L, device=dev).unsqueeze(0) - sh) % L
    w = torch.gather(w, 1, idx)
    snr = torch.rand(B, 1, device=dev) * (snr_hi - snr_lo) + snr_lo
    npow = (w.abs() ** 2).mean(1, keepdim=True) / (10 ** (snr / 10))
    w = w + torch.sqrt(npow / 2) * (torch.randn_like(w.real) + 1j * torch.randn_like(w.imag))
    return w

SCRATCH = ("/tmp/claude-1001/-home-nghoselab/7e6fdecd-9fba-4a89-857f-b1c39ddcc651/scratchpad")


# ── complex primitives (re, im) both [B, C, L] ──────────────────────────────
class ComplexConv1d(nn.Module):
    def __init__(self, ci, co, k):
        super().__init__()
        p = k // 2
        self.rr = nn.Conv1d(ci, co, k, padding=p)
        self.ri = nn.Conv1d(ci, co, k, padding=p)
        self.ir = nn.Conv1d(ci, co, k, padding=p)
        self.ii = nn.Conv1d(ci, co, k, padding=p)

    def forward(self, re, im):
        return self.rr(re) - self.ii(im), self.ri(re) + self.ir(im)


class ComplexBN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.bn_r = nn.BatchNorm1d(c)
        self.bn_i = nn.BatchNorm1d(c)

    def forward(self, re, im):
        return self.bn_r(re), self.bn_i(im)


def cpool(re, im):
    return F.avg_pool1d(re, 2), F.avg_pool1d(im, 2)


def crelu(re, im):
    return F.relu(re), F.relu(im)


class CVNN(nn.Module):
    def __init__(self, nc=NUM_CLASSES):
        super().__init__()
        chans = [(1, 32, 7), (32, 64, 5), (64, 128, 5), (128, 128, 3)]
        self.convs = nn.ModuleList([ComplexConv1d(ci, co, k) for ci, co, k in chans])
        self.bns = nn.ModuleList([ComplexBN(co) for _, co, _ in chans])
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, nc))

    def forward(self, re, im):
        for conv, bn in zip(self.convs, self.bns):
            re, im = conv(re, im)
            re, im = bn(re, im)
            re, im = crelu(re, im)
            re, im = cpool(re, im)
        mag = torch.sqrt(re * re + im * im + 1e-12)      # complex -> real magnitude
        z = F.adaptive_avg_pool1d(mag, 1).flatten(1)
        return self.head(z)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ── complex windowing (unit-RMS) ────────────────────────────────────────────
def iq_to_complex(windows, cfo_correct=False):
    """[n, WIN] complex -> unit-RMS complex64 [n, WIN]. If cfo_correct, first remove each
    window's coarse carrier offset (delay-and-multiply estimate) — like a WiFi RX syncing
    before decode — so a per-run LO retune offset no longer shifts the input."""
    w = np.asarray(windows, np.complex64)
    if cfo_correct:
        r = np.sum(w[:, 1:] * np.conj(w[:, :-1]), axis=1)     # per-window mean freq
        fest = np.angle(r).astype(np.float32)                  # rad/sample
        n = np.arange(w.shape[1], dtype=np.float32)
        w = w * np.exp(-1j * fest[:, None] * n[None, :])
    rms = np.sqrt(np.mean(np.abs(w) ** 2, axis=1, keepdims=True)) + 1e-9
    return (w / rms).astype(np.complex64)


def find_runs(root, d):
    return sorted(glob.glob(os.path.join(root, f"device_{d}", "*.bin")))


def build(root, val_run, hop_train, thr_mult, cache, rebuild, skip_run=None, cfo_correct=False):
    if os.path.exists(cache) and not rebuild:
        z = np.load(cache)
        print(f"  (cache {cache})")
        return z['Xtr'], z['ytr'], z['Xva'], z['yva'], z['vfid'], json.loads(str(z['counts']))
    Xtr, ytr, Xva, yva, vfid, counts = [], [], [], [], [], {}
    fid = 0
    for d in range(1, NUM_CLASSES + 1):
        runs = find_runs(root, d)
        if not runs:
            continue
        multi = len(runs) > 1
        nfr = 0
        for cap in runs:
            rm = re.search(r'run_?(\d+)', os.path.basename(cap))
            run_no = int(rm.group(1)) if rm else 1
            if skip_run is not None and run_no == skip_run:
                continue                                    # drop a corrupted run entirely
            frames = extract_frames_for_file(cap, floor_pct=20.0, thr_mult=thr_mult)[0]
            nfr += len(frames)
            for i, f in enumerate(frames):
                is_val = (run_no == val_run) if multi else (i % 5 == 0)
                x = iq_to_complex(frame_to_windows(f, hop=WIN if is_val else hop_train), cfo_correct)
                if is_val:
                    Xva.append(x); yva.append(np.full(len(x), d - 1))
                    vfid.append(np.full(len(x), fid)); fid += 1
                else:
                    Xtr.append(x); ytr.append(np.full(len(x), d - 1))
        counts[DEVICE_NAMES[d - 1]] = int(nfr)
        print(f"  device_{d}: {nfr} frames over {len(runs)} run(s)  "
              f"(val: {'run%d' % val_run if multi else 'frame split'})")
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr).astype(np.int64)
    Xva = np.concatenate(Xva); yva = np.concatenate(yva).astype(np.int64)
    vfid = np.concatenate(vfid).astype(np.int64)
    np.savez(cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, vfid=vfid, counts=json.dumps(counts))
    return Xtr, ytr, Xva, yva, vfid, counts


def to_reim(xb, dev):
    xb = xb.to(dev)
    return xb.real.unsqueeze(1), xb.imag.unsqueeze(1)      # [B,1,WIN] each


def per_device_eval(model, dev, Xva, yva, vfid, nc):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xva), 1024):
            re, im = to_reim(torch.from_numpy(Xva[i:i + 1024]), dev)
            preds.append(model(re, im).argmax(1).cpu().numpy())
    wp = np.concatenate(preds)
    win_acc = float((wp == yva).mean())
    C = np.zeros((nc, nc), int); per = {i: [0, 0] for i in range(nc)}
    for fv in np.unique(vfid):
        m = vfid == fv; t = int(yva[m][0])
        v = int(np.bincount(wp[m], minlength=nc).argmax())
        C[t, v] += 1; per[t][1] += 1; per[t][0] += (v == t)
    frame_acc = sum(p[0] for p in per.values()) / max(1, sum(p[1] for p in per.values()))
    return win_acc, frame_acc, per, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default="/media/nghoselab/T9/Data/session13/train/7_02_2026")
    ap.add_argument('--val-run', type=int, default=3)
    ap.add_argument('--skip-run', type=int, default=None, help='drop a corrupted run entirely')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--hop-train', type=int, default=384)
    ap.add_argument('--thr-mult', type=float, default=2.0)
    ap.add_argument('--tag', default='cvnn20260702')
    ap.add_argument('--init', default=None, help='CVNN checkpoint to FINE-TUNE from (else train from scratch)')
    ap.add_argument('--no-aug', action='store_true', help='disable drift augmentation')
    ap.add_argument('--cfo-correct', action='store_true', help='CFO-sync each window before the model (frequency-invariant)')
    ap.add_argument('--cfo-max', type=float, default=3000.0)
    ap.add_argument('--snr-lo', type=float, default=8.0)
    ap.add_argument('--snr-hi', type=float, default=30.0)
    ap.add_argument('--rebuild', action='store_true')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); np.random.seed(0)

    cache = os.path.join(SCRATCH, f"cvnn_{os.path.basename(a.root.rstrip('/'))}_v{a.val_run}s{a.skip_run}c{int(a.cfo_correct)}.npz")
    print(f"CVNN train  root {a.root}  val_run {a.val_run}  skip_run {a.skip_run}  cfo_correct {a.cfo_correct}  device {dev}")
    Xtr, ytr, Xva, yva, vfid, counts = build(a.root, a.val_run, a.hop_train, a.thr_mult, cache, a.rebuild, a.skip_run, a.cfo_correct)
    print(f"  frames/device: {counts}\n  train {Xtr.shape}  val {Xva.shape}")

    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=256, shuffle=True)
    m = CVNN(NUM_CLASSES).to(dev)
    if a.init:
        m.load_state_dict(torch.load(a.init, map_location=dev))
        print(f"  FINE-TUNING from {a.init}  (lr {a.lr})")
    print(f"  CVNN params: {n_params(m):,}")
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_state = 0.0, None
    for ep in range(1, a.epochs + 1):
        m.train(); tot = nb = 0
        for xb, yb in dl:
            xb = xb.to(dev)
            if not a.no_aug:
                xb = drift_augment(xb, a.cfo_max, a.snr_lo, a.snr_hi)
            re, im = xb.real.unsqueeze(1), xb.imag.unsqueeze(1); yb = yb.to(dev)
            opt.zero_grad(); loss = lossf(m(re, im), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        w, f, _, _ = per_device_eval(m, dev, Xva, yva, vfid, NUM_CLASSES)
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:2d}  loss {tot/nb:.4f}  val_win {w:.4f}  val_frame {f:.4f}")
        if w > best:
            best = w; best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    w, f, per, C = per_device_eval(m, dev, Xva, yva, vfid, NUM_CLASSES)
    print(f"\n  BEST CVNN  val_win {w:.4f}  val_frame {f:.4f}")
    for i in range(NUM_CLASSES):
        ok, tot = per[i]; print(f"    {DEVICE_NAMES[i]}: {ok/max(1,tot):.3f} ({ok}/{tot})")
    print("\n  confusion (rows=true):")
    names = [DEVICE_NAMES[i] for i in range(NUM_CLASSES)]
    print("  true\\pred " + "".join(f"{n:>9}" for n in names))
    for i in range(NUM_CLASSES):
        print(f"  {names[i]:>9}" + "".join(f"{C[i,j]:>9}" for j in range(NUM_CLASSES)) + f"  n={C[i].sum()}")
    out = os.path.join(os.path.dirname(__file__), f"fingerprint_{a.tag}.pt")
    torch.save(best_state, out)
    print(f"\n  saved -> {out}")


if __name__ == '__main__':
    main()

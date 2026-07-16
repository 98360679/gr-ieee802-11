#!/usr/bin/env python3
"""
ff_arch.py — architecture comparison for Fading Fingerprints.

Trains a real-valued 1-D CNN (exp3_fp_model.FingerprintCNN) on the SAME windows / split /
augmentation as the CVNN (ff_train.py), in either the time or frequency domain, so we can
compare the real CNN (both domains) against the complex CVNN on the same-model task.
Complex windows [B,WIN] are fed as 2 real channels [B,2,WIN].

  python3 ff_arch.py --exp 1 --protocol within-day --day 1 --domain time
  python3 ff_arch.py --exp 1 --protocol within-day --day 1 --domain freq
"""
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from exp3_fp_model import FingerprintCNN, WIN
from exp3_cvnn import drift_augment
from ff_dataset import build_dataset, DEFAULT_ROOT


def prep(xb, domain='time'):
    """[B,WIN] complex -> [B,2,WIN] real (I/Q). domain='freq' takes the centered,
    energy-normalized spectrum first (CFO -> spectral shift, I/Q imbalance -> mirror
    asymmetry, PA regrowth -> spectral skirts)."""
    if domain == 'freq':
        xb = torch.fft.fftshift(torch.fft.fft(xb, dim=-1, norm='ortho'), dim=-1)
    return torch.stack([xb.real, xb.imag], dim=1).float()      # [B,WIN] cplx -> [B,2,WIN]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def per_device_eval(model, dev, Xva, yva, vfid, nc, domain='time'):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(Xva), 1024):
            xb = torch.from_numpy(Xva[i:i + 1024]).to(dev)
            preds.append(model(prep(xb, domain)).argmax(1).cpu().numpy())
    wp = np.concatenate(preds)
    win_acc = float((wp == yva).mean())
    C = np.zeros((nc, nc), int); per = {i: [0, 0] for i in range(nc)}
    for fv in np.unique(vfid):
        m = vfid == fv; t = int(yva[m][0])
        v = int(np.bincount(wp[m], minlength=nc).argmax())
        C[t, v] += 1; per[t][1] += 1; per[t][0] += (v == t)
    frame_acc = sum(p[0] for p in per.values()) / max(1, sum(p[1] for p in per.values()))
    return win_acc, frame_acc, per, C


def train_one(Xtr, ytr, Xva, yva, vfid, nc, dev, epochs, lr, no_aug, domain='time'):
    model = FingerprintCNN(nc).to(dev)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl:
            xb = xb.to(dev)
            if not no_aug:
                xb = drift_augment(xb)
            x2 = prep(xb, domain); yb = yb.to(dev)
            opt.zero_grad(); loss = lossf(model(x2), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
        w, _, _, _ = per_device_eval(model, dev, Xva, yva, vfid, nc, domain)
        if w > best:
            best = w; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--exp', type=int, default=1, choices=[1, 2])
    ap.add_argument('--protocol', default='within-day', choices=['within-day', 'cross-day'])
    ap.add_argument('--day', type=int, default=1)
    ap.add_argument('--val-run', type=int, default=3)
    ap.add_argument('--train-day', type=int, default=1)
    ap.add_argument('--val-day', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--no-aug', action='store_true')
    ap.add_argument('--domain', default='time', choices=['time', 'freq'])
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); np.random.seed(0)

    Xtr, ytr, Xva, yva, vfid, id2label, names, counts = build_dataset(
        a.root, a.exp, a.protocol, train_day=a.train_day, val_day=a.val_day,
        day=a.day, val_run=a.val_run)
    nc = len(names)
    print(f"arch-compare  exp{a.exp} {a.protocol}  DOMAIN={a.domain}  devices {names}  train {Xtr.shape}  val {Xva.shape}")
    model = train_one(Xtr, ytr, Xva, yva, vfid, nc, dev, a.epochs, a.lr, a.no_aug, a.domain)
    w, f, per, C = per_device_eval(model, dev, Xva, yva, vfid, nc, a.domain)
    print(f"\n===== CNN {a.domain} ({n_params(model):,} params) =====")
    print(f"  val_win {w:.4f}   val_frame {f:.4f}")
    for i in range(nc):
        ok, t = per[i]; print(f"    {names[i]}: {ok/max(1,t):.3f} ({ok}/{t})")
    print("  confusion (rows=true):  " + "  ".join(names))
    for i in range(nc):
        print(f"    {names[i]:>9} " + " ".join(f"{C[i,j]:>4}" for j in range(nc)))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
exp3_sweep.py — fooling rate vs PSR, swept over victim device
─────────────────────────────────────────────────────────────

For each victim device (the device being transmitted/fingerprinted) and each
PSR, craft an untargeted per-frame PGD perturbation against the retrained model
and measure the per-frame fooling rate on held-out (run-3) frames. Produces an
accuracy-vs-PSR table (the per-frame analogue of their attack_metrics.json) and
saves results (+ optional plot) to the drive.

PGD is vectorized across a device's frames (all windows in one forward pass per
step), so the full sweep is fast on CPU.

Run:
  .venv/bin/python exp3_sweep.py                       # all devices, default PSRs
  .venv/bin/python exp3_sweep.py --psrs -40,-30,-25,-20,-15 --n-frames 20 --devices 1,2,3
"""

import os
import json
import warnings
import argparse
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F

from exp3_fp_model import (FingerprintCNN, torch_iq_to_input, iq_to_input,
                           WIN, ACTIVE, PRE_ROLL, NUM_CLASSES, DEVICE_NAMES, FS)
from exp3_train_fingerprint import load_frames, LOCAL_PT, SAVE_PT

_EPS = 1e-12
DRIVE_PROC = "/media/cse-nghose-25/T9/Data/session12/processed"
W_PER_FRAME = (ACTIVE - WIN) // WIN + 1            # 14 inference windows/frame


def frame_windows_complex(frame):
    """(FRAME_LEN,) stored frame -> [W_PER_FRAME, WIN] complex tensor (active region)."""
    s = torch.from_numpy(frame.astype(np.complex64))
    starts = range(PRE_ROLL, PRE_ROLL + ACTIVE - WIN + 1, WIN)
    return torch.stack([s[a:a + WIN] for a in starts])


def clean_predict(model, frames):
    """Per-frame majority-vote predictions (no perturbation)."""
    preds = []
    for f in frames:
        x = iq_to_input(frame_windows_complex(f).numpy())
        with torch.no_grad():
            v = model(torch.from_numpy(x)).argmax(1).numpy()
        preds.append(int(np.bincount(v, minlength=NUM_CLASSES).argmax()))
    return np.array(preds)


def batched_pgd_fooling(model, frames, true_label, psr_db, steps, step_frac):
    """Vectorized untargeted per-frame PGD; returns (fooling_rate, adv_frame_preds)."""
    F_ = len(frames)
    sig = torch.stack([frame_windows_complex(f) for f in frames])     # [F,W,1024] cplx
    sig = sig.reshape(F_ * W_PER_FRAME, WIN)
    sig_norm = torch.sqrt((sig.real ** 2 + sig.imag ** 2).sum(1) + _EPS)
    budget = (10.0 ** (psr_db / 20.0)) * sig_norm                    # per-window L2 cap
    y = torch.full((F_ * W_PER_FRAME,), true_label, dtype=torch.long)

    d = torch.zeros(F_ * W_PER_FRAME, WIN, 2)
    d.normal_(0, 1e-3); d.requires_grad_(True)

    def inp(delta):
        dc = torch.complex(delta[..., 0], delta[..., 1])
        return torch_iq_to_input(sig + dc)

    for _ in range(steps):
        if d.grad is not None:
            d.grad.zero_()
        loss = F.cross_entropy(model(inp(d)), y)
        loss.backward()
        with torch.no_grad():
            g = d.grad
            gn = g.flatten(1).norm(dim=1).clamp_min(_EPS)
            d += (step_frac * budget / gn).view(-1, 1, 1) * g
            dn = d.flatten(1).norm(dim=1)
            d *= torch.clamp(budget / dn.clamp_min(_EPS), max=1.0).view(-1, 1, 1)

    with torch.no_grad():
        pred = model(inp(d)).argmax(1).numpy().reshape(F_, W_PER_FRAME)
    frame_pred = np.array([np.bincount(pr, minlength=NUM_CLASSES).argmax() for pr in pred])
    fooling = float((frame_pred != true_label).mean())
    return fooling, frame_pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--psrs', default='-40,-35,-30,-25,-20,-15',
                   help='comma list of PSR dB')
    p.add_argument('--devices', default='1,2,3,4,5,6')
    p.add_argument('--n-frames', type=int, default=20, help='held-out frames per device')
    p.add_argument('--steps', type=int, default=80)
    p.add_argument('--step-frac', type=float, default=0.1)
    p.add_argument('--model', default=LOCAL_PT if os.path.exists(LOCAL_PT) else SAVE_PT)
    p.add_argument('--out', default=os.path.join(DRIVE_PROC, 'sweep_metrics_retrained.json'))
    p.add_argument('--plot', default=os.path.join(DRIVE_PROC, 'sweep_fooling_vs_psr.png'))
    a = p.parse_args()

    psrs = [float(x) for x in a.psrs.split(',')]
    devices = [int(x) for x in a.devices.split(',')]

    model = FingerprintCNN(NUM_CLASSES)
    model.load_state_dict(torch.load(a.model, map_location='cpu'))
    model.eval()

    res = {'psrs': psrs, 'devices': devices, 'n_frames': a.n_frames,
           'clean_acc': {}, 'fooling': {}, 'mis_target': {}}

    hdr = "device      clean   " + "  ".join(f"{p:>6.0f}" for p in psrs)
    print(hdr); print('-' * len(hdr))
    for d in devices:
        frames, run = load_frames(d)
        idx = np.where(run == 3)[0]
        # use only frames the clean model gets right, so fooling is meaningful
        fr_all = [frames[i] for i in idx]
        cp = clean_predict(model, fr_all)
        correct = [fr_all[i] for i in range(len(fr_all)) if cp[i] == d - 1]
        clean_acc = float((cp == d - 1).mean())
        fr = correct[:a.n_frames]
        res['clean_acc'][d] = clean_acc

        row_fool, row_mis = [], []
        for psr in psrs:
            fool, fpred = batched_pgd_fooling(model, fr, d - 1, psr, a.steps, a.step_frac)
            row_fool.append(fool)
            err = fpred[fpred != d - 1]
            row_mis.append(int(np.bincount(err, minlength=NUM_CLASSES).argmax())
                           + 1 if len(err) else None)
        res['fooling'][d] = row_fool
        res['mis_target'][d] = row_mis
        cells = "  ".join(f"{f*100:>5.0f}%" for f in row_fool)
        print(f"{DEVICE_NAMES[d-1]:>9}  {clean_acc*100:>5.0f}%  {cells}", flush=True)
        # persist after EACH device so a long run can never lose everything
        with open(a.out, 'w') as jf:
            json.dump(res, jf, indent=2)

    print("\n(values = per-frame fooling rate %; accuracy = 100 - fooling)")
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"saved metrics -> {a.out}")

    # optional plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        for d in devices:
            plt.plot(psrs, [100 - 100 * x for x in res['fooling'][d]],
                     marker='o', label=DEVICE_NAMES[d - 1])
        plt.xlabel('PSR (dB)'); plt.ylabel('fingerprint accuracy (%)')
        plt.title('Per-frame PGD: accuracy vs PSR (retrained model)')
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(a.plot, dpi=120)
        print(f"saved plot    -> {a.plot}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == '__main__':
    main()

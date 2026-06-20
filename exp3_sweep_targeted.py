#!/usr/bin/env python3
"""
exp3_sweep_targeted.py — targeted per-frame PGD: victim x target success matrix
────────────────────────────────────────────────────────────────────────────────

Untargeted PGD only asks "make it wrong". Targeted PGD asks "make device V look
like device T". For each (victim V, target T != V) at a fixed PSR, craft a
targeted per-frame perturbation against the retrained model and measure the
targeted success rate = fraction of V's held-out frames the model calls T.

Vectorized across frames; results saved incrementally to the drive (per victim
row) so a long run can't lose progress. Render with exp3_plot_sweep.py.

Run:
  .venv/bin/python exp3_sweep_targeted.py --psr -20 --n-frames 15 --steps 80
"""

import os
import json
import warnings
import argparse
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F

from exp3_fp_model import FingerprintCNN, torch_iq_to_input, NUM_CLASSES, DEVICE_NAMES
from exp3_sweep import frame_windows_complex, clean_predict, W_PER_FRAME, DRIVE_PROC
from exp3_train_fingerprint import load_frames, LOCAL_PT, SAVE_PT

_EPS = 1e-12


def batched_targeted(model, frames, target_label, psr_db, steps, step_frac):
    """Targeted PGD: push V's frames toward `target_label`. Returns success rate."""
    F_ = len(frames)
    sig = torch.stack([frame_windows_complex(f) for f in frames]).reshape(
        F_ * W_PER_FRAME, -1)
    sig_norm = torch.sqrt((sig.real ** 2 + sig.imag ** 2).sum(1) + _EPS)
    budget = (10.0 ** (psr_db / 20.0)) * sig_norm
    y = torch.full((F_ * W_PER_FRAME,), target_label, dtype=torch.long)

    d = torch.zeros(F_ * W_PER_FRAME, sig.shape[1], 2)
    d.normal_(0, 1e-3); d.requires_grad_(True)

    def inp(delta):
        return torch_iq_to_input(sig + torch.complex(delta[..., 0], delta[..., 1]))

    for _ in range(steps):
        if d.grad is not None:
            d.grad.zero_()
        loss = F.cross_entropy(model(inp(d)), y)   # MINIMIZE -> classify as target
        loss.backward()
        with torch.no_grad():
            g = d.grad
            gn = g.flatten(1).norm(dim=1).clamp_min(_EPS)
            d -= (step_frac * budget / gn).view(-1, 1, 1) * g   # descent toward target
            dn = d.flatten(1).norm(dim=1)
            d *= torch.clamp(budget / dn.clamp_min(_EPS), max=1.0).view(-1, 1, 1)

    with torch.no_grad():
        pred = model(inp(d)).argmax(1).numpy().reshape(F_, W_PER_FRAME)
    frame_pred = np.array([np.bincount(p, minlength=NUM_CLASSES).argmax() for p in pred])
    return float((frame_pred == target_label).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--psr', type=float, default=-20.0)
    p.add_argument('--victims', default='1,2,3,4,5,6')
    p.add_argument('--targets', default='1,2,3,4,5,6')
    p.add_argument('--n-frames', type=int, default=15)
    p.add_argument('--steps', type=int, default=80)
    p.add_argument('--step-frac', type=float, default=0.1)
    p.add_argument('--model', default=LOCAL_PT if os.path.exists(LOCAL_PT) else SAVE_PT)
    p.add_argument('--out', default=os.path.join(DRIVE_PROC, 'sweep_targeted_retrained.json'))
    a = p.parse_args()

    victims = [int(x) for x in a.victims.split(',')]
    targets = [int(x) for x in a.targets.split(',')]

    model = FingerprintCNN(NUM_CLASSES)
    model.load_state_dict(torch.load(a.model, map_location='cpu'))
    model.eval()

    res = {'psr': a.psr, 'victims': victims, 'targets': targets,
           'n_frames': a.n_frames, 'success': {}}

    hdr = "victim \\ target  " + "  ".join(f"{DEVICE_NAMES[t-1][-3:]:>5}" for t in targets)
    print(f"Targeted PGD @ PSR {a.psr} dB  (cell = % of victim frames classified as target)")
    print(hdr); print('-' * len(hdr))
    for v in victims:
        frames, run = load_frames(v)
        idx = np.where(run == 3)[0]
        fr_all = [frames[i] for i in idx]
        cp = clean_predict(model, fr_all)
        correct = [fr_all[i] for i in range(len(fr_all)) if cp[i] == v - 1][:a.n_frames]
        row = {}
        cells = []
        for t in targets:
            if t == v:
                row[t] = None; cells.append("   - "); continue
            s = batched_targeted(model, correct, t - 1, a.psr, a.steps, a.step_frac)
            row[t] = s; cells.append(f"{s*100:>4.0f}%")
        res['success'][v] = row
        print(f"{DEVICE_NAMES[v-1]:>14}  " + "  ".join(cells), flush=True)
        with open(a.out, 'w') as jf:
            json.dump(res, jf, indent=2)

    print(f"\nsaved -> {a.out}")


if __name__ == '__main__':
    main()

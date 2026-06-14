#!/usr/bin/env python3
"""
exp3_sweep_targeted_psr.py — targeted attack success vs PSR, for chosen pairs
─────────────────────────────────────────────────────────────────────────────

The matrix script (exp3_sweep_targeted.py) fixes PSR and sweeps pairs. This one
fixes a few victim->target pairs and sweeps PSR, so we can see how low the
perturbation power can go before a *targeted* impersonation stops working.

Default pairs tell the story:
  1:2  easy victim -> easy target
  4:6  the robust victim (device_4) -> its easiest target
  1:4  targeting the robust device as the forged identity
  5:3  the mutually-confusable pair (expected to succeed at very low PSR)

Reuses the validated targeted PGD; saves JSON + a plot to the drive.

Run:
  .venv/bin/python exp3_sweep_targeted_psr.py
  .venv/bin/python exp3_sweep_targeted_psr.py --pairs 1:2,4:6 --psrs -45,-40,-35,-30,-25,-20,-15
"""

import os
import json
import warnings
import argparse
warnings.filterwarnings('ignore')
import numpy as np
import torch

from exp3_fp_model import FingerprintCNN, NUM_CLASSES, DEVICE_NAMES
from exp3_sweep import clean_predict, DRIVE_PROC
from exp3_sweep_targeted import batched_targeted
from exp3_train_fingerprint import load_frames, LOCAL_PT, SAVE_PT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', default='1:2,4:6,1:4,5:3', help='victim:target,...')
    p.add_argument('--psrs', default='-45,-40,-35,-30,-25,-20,-15')
    p.add_argument('--n-frames', type=int, default=15)
    p.add_argument('--steps', type=int, default=80)
    p.add_argument('--step-frac', type=float, default=0.1)
    p.add_argument('--model', default=LOCAL_PT if os.path.exists(LOCAL_PT) else SAVE_PT)
    p.add_argument('--out', default=os.path.join(DRIVE_PROC, 'sweep_targeted_psr_retrained.json'))
    p.add_argument('--plot', default=os.path.join(DRIVE_PROC, 'sweep_targeted_psr.png'))
    a = p.parse_args()

    pairs = [tuple(int(x) for x in pr.split(':')) for pr in a.pairs.split(',')]
    psrs = [float(x) for x in a.psrs.split(',')]

    model = FingerprintCNN(NUM_CLASSES)
    model.load_state_dict(torch.load(a.model, map_location='cpu'))
    model.eval()

    # cache each victim's clean-correct held-out frames once
    frame_cache = {}
    def victim_frames(v):
        if v not in frame_cache:
            frames, run = load_frames(v)
            idx = np.where(run == 3)[0]
            fr = [frames[i] for i in idx]
            cp = clean_predict(model, fr)
            frame_cache[v] = [fr[i] for i in range(len(fr)) if cp[i] == v - 1][:a.n_frames]
        return frame_cache[v]

    res = {'psrs': psrs, 'pairs': [f"{v}:{t}" for v, t in pairs], 'success': {}}

    hdr = "victim->target  " + "  ".join(f"{p:>6.0f}" for p in psrs)
    print(f"Targeted success vs PSR  (cell = % frames forced to target)")
    print(hdr); print('-' * len(hdr))
    for (v, t) in pairs:
        fr = victim_frames(v)
        row = []
        for psr in psrs:
            row.append(batched_targeted(model, fr, t - 1, psr, a.steps, a.step_frac))
        key = f"{v}:{t}"
        res['success'][key] = row
        label = f"{DEVICE_NAMES[v-1][-1]}->{DEVICE_NAMES[t-1][-1]}"
        cells = "  ".join(f"{s*100:>5.0f}%" for s in row)
        print(f"  dev {label:>6}   {cells}", flush=True)
        with open(a.out, 'w') as jf:
            json.dump(res, jf, indent=2)

    print(f"\nsaved metrics -> {a.out}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.5, 5))
        for (v, t) in pairs:
            s = [100 * x for x in res['success'][f"{v}:{t}"]]
            plt.plot(psrs, s, marker='o',
                     label=f"{DEVICE_NAMES[v-1]} → {DEVICE_NAMES[t-1]}")
        plt.xlabel("PSR (dB)"); plt.ylabel("targeted success (%)")
        plt.title("Targeted PGD: impersonation success vs PSR")
        plt.ylim(-3, 103); plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(a.plot, dpi=120)
        print(f"saved plot    -> {a.plot}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == '__main__':
    main()

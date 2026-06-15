#!/usr/bin/env python3
"""
exp3_targeted_victim_heatmap.py — targeted success for ONE victim: target x PSR
────────────────────────────────────────────────────────────────────────────────

Fix a victim device and show, as a heatmap, how strongly it can be forced to
impersonate each other device (rows) as a function of PSR (cols). Complements the
all-victim matrix (exp3_sweep_targeted.py) when you've committed to one victim
for the OTA run.

Run:
  .venv/bin/python exp3_targeted_victim_heatmap.py --victim 2
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
    p.add_argument('--victim', type=int, default=2)
    p.add_argument('--targets', default=None, help='default: all devices except the victim')
    p.add_argument('--psrs', default='-45,-40,-35,-30,-25,-20,-15')
    p.add_argument('--n-frames', type=int, default=15)
    p.add_argument('--steps', type=int, default=80)
    p.add_argument('--step-frac', type=float, default=0.1)
    p.add_argument('--model', default=LOCAL_PT if os.path.exists(LOCAL_PT) else SAVE_PT)
    p.add_argument('--out', default=None)
    p.add_argument('--plot', default=None)
    a = p.parse_args()

    v = a.victim
    targets = [int(x) for x in a.targets.split(',')] if a.targets else \
        [d for d in range(1, NUM_CLASSES + 1) if d != v]
    psrs = [float(x) for x in a.psrs.split(',')]
    out = a.out or os.path.join(DRIVE_PROC, f'sweep_targeted_dev{v}_victim.json')
    plot = a.plot or os.path.join(DRIVE_PROC, f'sweep_targeted_dev{v}_heatmap.png')

    model = FingerprintCNN(NUM_CLASSES)
    model.load_state_dict(torch.load(a.model, map_location='cpu'))
    model.eval()

    frames, run = load_frames(v)
    idx = np.where(run == 3)[0]
    fr_all = [frames[i] for i in idx]
    cp = clean_predict(model, fr_all)
    fr = [fr_all[i] for i in range(len(fr_all)) if cp[i] == v - 1][:a.n_frames]
    print(f"victim = {DEVICE_NAMES[v-1]}  (clean-correct frames used: {len(fr)})")

    res = {'victim': v, 'targets': targets, 'psrs': psrs, 'n_frames': len(fr), 'success': {}}
    hdr = "target \\ PSR   " + "  ".join(f"{q:>6.0f}" for q in psrs)
    print(hdr); print('-' * len(hdr))
    M = np.full((len(targets), len(psrs)), np.nan)
    for i, t in enumerate(targets):
        row = [batched_targeted(model, fr, t - 1, q, a.steps, a.step_frac) for q in psrs]
        res['success'][t] = row
        M[i] = [100 * x for x in row]
        print(f"{DEVICE_NAMES[t-1]:>12}  " + "  ".join(f"{x*100:>5.0f}%" for x in row), flush=True)
        with open(out, 'w') as jf:
            json.dump(res, jf, indent=2)
    print(f"\nsaved metrics -> {out}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.5, 4.8))
        im = plt.imshow(M, vmin=0, vmax=100, cmap='magma', aspect='auto')
        plt.colorbar(im, label="targeted success (%)")
        plt.xticks(range(len(psrs)), [f"{q:.0f}" for q in psrs])
        plt.yticks(range(len(targets)), [DEVICE_NAMES[t - 1] for t in targets])
        plt.xlabel("PSR (dB)"); plt.ylabel("forced TARGET identity")
        plt.title(f"Targeted PGD success — victim {DEVICE_NAMES[v-1]}")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    plt.text(j, i, f"{M[i,j]:.0f}", ha='center', va='center',
                             color='white' if M[i, j] < 60 else 'black', fontsize=8)
        plt.tight_layout(); plt.savefig(plot, dpi=120)
        print(f"saved plot    -> {plot}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == '__main__':
    main()

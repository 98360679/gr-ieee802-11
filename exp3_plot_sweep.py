#!/usr/bin/env python3
"""
exp3_plot_sweep.py — render figures from the saved sweep JSONs (no re-run)
──────────────────────────────────────────────────────────────────────────

  * accuracy vs PSR (line per device)        from sweep_metrics_retrained.json
  * targeted success heatmap (victim x target) from sweep_targeted_retrained.json
    (only if that file exists)

Run:
  .venv/bin/python exp3_plot_sweep.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from exp3_fp_model import DEVICE_NAMES

DRIVE_PROC = "/media/cse-nghose-25/T9/Data/session12/processed"
UNTARG = os.path.join(DRIVE_PROC, "sweep_metrics_retrained.json")
TARG   = os.path.join(DRIVE_PROC, "sweep_targeted_retrained.json")
OUT_LINE = os.path.join(DRIVE_PROC, "sweep_accuracy_vs_psr.png")
OUT_HEAT = os.path.join(DRIVE_PROC, "sweep_targeted_heatmap.png")


def plot_untargeted():
    if not os.path.exists(UNTARG):
        print(f"(no {UNTARG})"); return
    r = json.load(open(UNTARG))
    psrs = r["psrs"]
    plt.figure(figsize=(7.5, 5))
    for d in r["devices"]:
        acc = [100 - 100 * x for x in r["fooling"][str(d)]] \
            if str(d) in r["fooling"] else [100 - 100 * x for x in r["fooling"][d]]
        plt.plot(psrs, acc, marker='o', label=DEVICE_NAMES[int(d) - 1])
    plt.xlabel("PSR (dB)"); plt.ylabel("fingerprint accuracy (%)")
    plt.title("Per-frame PGD (untargeted): accuracy vs PSR")
    plt.ylim(-3, 103); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(OUT_LINE, dpi=120)
    print(f"saved {OUT_LINE}")


def plot_targeted():
    if not os.path.exists(TARG):
        print(f"(no targeted JSON yet: {TARG})"); return
    r = json.load(open(TARG))
    victims = r["victims"]; targets = r["targets"]; psr = r["psr"]
    M = np.full((len(victims), len(targets)), np.nan)
    for i, v in enumerate(victims):
        row = r["success"][str(v)] if str(v) in r["success"] else r["success"][v]
        for j, t in enumerate(targets):
            key = str(t) if str(t) in row else t
            val = row.get(key) if isinstance(row, dict) else row[j]
            if val is not None:
                M[i, j] = 100 * val
    plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(M, vmin=0, vmax=100, cmap='magma', aspect='auto')
    plt.colorbar(im, label="targeted success (%)")
    plt.xticks(range(len(targets)), [DEVICE_NAMES[t - 1] for t in targets], rotation=45, ha='right')
    plt.yticks(range(len(victims)), [DEVICE_NAMES[v - 1] for v in victims])
    plt.xlabel("forced TARGET device"); plt.ylabel("true VICTIM device")
    plt.title(f"Targeted PGD success @ PSR {psr} dB")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                plt.text(j, i, f"{M[i,j]:.0f}", ha='center', va='center',
                         color='white' if M[i, j] < 60 else 'black', fontsize=8)
    plt.tight_layout(); plt.savefig(OUT_HEAT, dpi=120)
    print(f"saved {OUT_HEAT}")


if __name__ == '__main__':
    plot_untargeted()
    plot_targeted()

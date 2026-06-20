#!/usr/bin/env python3
"""
exp3_ota_compare.py — clean vs attacked OTA comparison + realized PSR
─────────────────────────────────────────────────────────────────────

One-shot analysis for the rig. Give it the captures you took with
exp3_rx_eval.py (--source usrp --capture ...) and it prints:

  * the clean replay identity D0 (what the fingerprinter calls the clean signal)
  * clean vs attacked fingerprint accuracy / fooling rate
  * the realized PSR (if a perturbation-only capture is provided)

"Fooling" is measured RELATIVE to the clean baseline D0, so the result is valid
even if the replayed signal doesn't classify as the original device (the replay
fingerprint is dominated by the TX USRP — see the OTA plan).

Typical rig flow:
  # clean (adversary epsilon=0)
  exp3_rx_eval.py --source usrp --addr <rx> --duration 20 --capture /tmp/rx_clean.bin
  # perturbation only (legit muted, adversary on)   [for realized PSR]
  exp3_rx_eval.py --source usrp --addr <rx> --duration 20 --capture /tmp/rx_pert.bin
  # attacked (legit + adversary)
  exp3_rx_eval.py --source usrp --addr <rx> --duration 20 --capture /tmp/rx_attacked.bin

  exp3_ota_compare.py --clean /tmp/rx_clean.bin --attacked /tmp/rx_attacked.bin \
        --pert-only /tmp/rx_pert.bin --true-device 1
"""

import os
import argparse
import numpy as np
import torch

from exp3_fp_model import FingerprintCNN, NUM_CLASSES, DEVICE_NAMES
from exp3_rx_eval import src_file, classify_stream, active_power
from exp3_train_fingerprint import LOCAL_PT, SAVE_PT


def preds_of(results):
    """results from classify_stream -> array of per-frame device indices."""
    return np.array([r[0] for r in results], dtype=int)


def dist_line(devs):
    h = np.bincount(devs, minlength=NUM_CLASSES)
    return "  ".join(f"{DEVICE_NAMES[d]}:{h[d]}" for d in range(NUM_CLASSES) if h[d])


def main():
    p = argparse.ArgumentParser(description="OTA clean-vs-attacked comparison")
    p.add_argument('--clean', required=True, help='clean capture (adversary off)')
    p.add_argument('--attacked', required=True, help='attacked capture (legit + perturbation)')
    p.add_argument('--pert-only', help='perturbation-only capture (legit muted) for realized PSR')
    p.add_argument('--true-device', type=int, default=None, help='device 1..6 transmitted')
    p.add_argument('--model', default=LOCAL_PT if os.path.exists(LOCAL_PT) else SAVE_PT)
    p.add_argument('--align-search', type=int, default=64)
    a = p.parse_args()

    model = FingerprintCNN(NUM_CLASSES)
    model.load_state_dict(torch.load(a.model, map_location='cpu'))
    model.eval()

    clean = src_file(a.clean)
    attacked = src_file(a.attacked)
    rc = classify_stream(clean, model, a.align_search)
    ra = classify_stream(attacked, model, a.align_search)
    if not rc:
        raise SystemExit("No frames detected in the CLEAN capture (check gain/freq/threshold).")
    if not ra:
        raise SystemExit("No frames detected in the ATTACKED capture.")
    dc, da = preds_of(rc), preds_of(ra)

    D0 = int(np.bincount(dc, minlength=NUM_CLASSES).argmax())
    clean_consistency = float((dc == D0).mean())
    fooling = float((da != D0).mean())                 # shift away from the clean baseline

    print("=" * 60)
    print("OTA clean vs attacked")
    print("=" * 60)
    print(f"  CLEAN   : {len(dc)} frames  [{dist_line(dc)}]")
    print(f"            baseline identity D0 = {DEVICE_NAMES[D0]} "
          f"(consistency {clean_consistency*100:.0f}%)")
    print(f"  ATTACKED: {len(da)} frames  [{dist_line(da)}]")
    print(f"            fooling vs baseline (shift off {DEVICE_NAMES[D0]}): {fooling*100:.0f}%")
    err = da[da != D0]
    if len(err):
        tgt = int(np.bincount(err, minlength=NUM_CLASSES).argmax())
        print(f"            most common attacked mis-ID: {DEVICE_NAMES[tgt]} "
              f"({np.bincount(err, minlength=NUM_CLASSES)[tgt]}/{len(err)})")

    if a.true_device is not None:
        tl = a.true_device - 1
        print(f"\n  vs TRUE device = {DEVICE_NAMES[tl]}:")
        print(f"    clean accuracy    : {(dc==tl).mean()*100:.1f}%")
        print(f"    attacked accuracy : {(da==tl).mean()*100:.1f}%")
        print(f"    accuracy drop     : {((dc==tl).mean()-(da==tl).mean())*100:.1f} pts")
        if D0 != tl:
            print(f"    NOTE: clean replay identity is {DEVICE_NAMES[D0]}, not "
                  f"{DEVICE_NAMES[tl]} — replay fingerprint ≠ original device. "
                  f"Use the 'fooling vs baseline' number as the attack metric.")

    # realized PSR
    Psig, _ = active_power(clean)
    print(f"\n  signal power (clean active region): {10*np.log10(Psig.mean()+1e-30):.2f} dB"
          f"  over {len(Psig)} bursts")
    if a.pert_only:
        Pp, _ = active_power(src_file(a.pert_only))
        if len(Pp):
            psr = 10 * np.log10(Pp.mean() / (Psig.mean() + 1e-30))
            print(f"  perturbation power               : "
                  f"{10*np.log10(Pp.mean()+1e-30):.2f} dB over {len(Pp)} bursts")
            print(f"  >>> REALIZED PSR = {psr:.2f} dB "
                  f"(target ~ -20 dB; raise epsilon/adv_gain if too low)")
        else:
            print("  perturbation-only capture: no bursts detected.")
    else:
        print("  (provide --pert-only to compute realized PSR)")
    print("=" * 60)


if __name__ == '__main__':
    main()

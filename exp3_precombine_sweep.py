#!/usr/bin/env python3
"""
exp3_precombine_sweep.py — pre-combine frame + delta into ONE single-channel stream
────────────────────────────────────────────────────────────────────────────────────
For single-channel TX (e.g. a B205mini that can't do the 2-channel MIMO adversary):
add the perturbation INTO the frame digitally, so one radio transmits (frame + delta).
delta is then sample-aligned to the frame by construction (no MIMO sync, no sub-sample
/ phase fragility). Takes a bundle's dac_safe_gapped dir (adv_frame.bin + adv_perturbation
_psr_*.bin, already aligned) and writes adv_combined_psr_<tag>.bin per PSR + a
adv_combined_psr_off.bin (frame only = the delta-off control, at the SAME scaling).

One common DAC-safe factor across the whole sweep keeps the frame (device fingerprint)
level constant across all PSRs.

Transmit each adv_combined_psr_*.bin on ch0 with the perturbation channel OFF.
"""
import os, re, glob, argparse, numpy as np
C64 = np.complex64


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True, help='dac_safe_gapped dir (adv_frame + adv_perturbation_psr_*)')
    p.add_argument('--out', required=True)
    p.add_argument('--headroom', type=float, default=0.95)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    frame = np.fromfile(os.path.join(a.bundle, 'adv_frame.bin'), dtype=C64)
    perts = sorted(glob.glob(os.path.join(a.bundle, 'adv_perturbation_psr_*.bin')))
    combos = {'off': frame.copy()}                       # delta-off control
    for fp in perts:
        tag = re.search(r'psr_([mp]?\d+)\.bin', os.path.basename(fp)).group(1)
        d = np.fromfile(fp, dtype=C64)
        n = min(len(frame), len(d))
        combos[tag] = frame[:n] + d[:n]                  # frame + delta (aligned by construction)

    gmax = max(float(np.max(np.abs(c))) for c in combos.values())
    k = a.headroom / gmax
    print(f"global combined peak {gmax:.3f} -> common DAC-safe scale {k:.4f}\n")
    print(f"{'PSR tag':>8} {'combined peak':>14}  file")
    for tag, c in combos.items():
        out = (c * k).astype(C64)
        fn = f"adv_combined_psr_{tag}.bin"
        out.tofile(os.path.join(a.out, fn))
        print(f"{tag:>8} {float(np.max(np.abs(out))):>14.3f}  {fn}")
    print(f"\n-> {a.out}/   (single-channel: transmit adv_combined_psr_<tag>.bin on ch0, "
          f"perturbation channel OFF; off = delta-off control)")


if __name__ == '__main__':
    main()

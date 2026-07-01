#!/usr/bin/env python3
"""
exp3_impairment_fit.py — characterize each device's hardware-impairment signature.

Step 1 of the hardware-mimicry attack (the only path with a physical route to the RF
fingerprint — a digital delta cannot, see exp3_results_summary.md "Mechanism"). Extracts
robust, scale-invariant impairment statistics per device from enrollment captures. These
are (a) what the fingerprint CNN keys on and (b) exactly what an adversary transmit chain
would have to reproduce to impersonate a target (e.g. device_4).

Estimators (per frame, active region, unit-RMS normalized so capture gain cancels):
  DC offset      |mean(x)|                      — LO leakage / bias
  IQ gain imbal  std(I)/std(Q)                  — modulator gain mismatch (1.0 = balanced)
  IQ quad skew   <I,Q>/sqrt(<I^2><Q^2>)         — quadrature error (0 = balanced)
  CFO (Hz)       FS/2pi * angle(sum x[n] conj(x[n-1]))  — carrier freq offset
  PAPR (dB)      max|x|^2 / mean|x|^2           — PA-compression / backoff proxy

  python3 exp3_impairment_fit.py                 # table over all 6 devices, run-3
  python3 exp3_impairment_fit.py --target 4      # also print device_4's mimicry target row
"""
import os, argparse, numpy as np
from exp3_extract_frames import extract_frames_for_file
from exp3_fp_model import PRE_ROLL, ACTIVE, FS, NUM_CLASSES

ROOT = "/media/nghoselab/T9/Data/session13/train/6_30_2026/enrollment"


def frame_stats(f):
    x = np.asarray(f, np.complex64)[PRE_ROLL:PRE_ROLL + ACTIVE]
    x = x - 0.0                                   # keep DC (do NOT de-mean)
    rms = np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12
    x = x / rms                                   # unit-RMS: cancels capture gain
    I, Q = x.real, x.imag
    dc = np.abs(np.mean(x))
    gimb = np.std(I) / (np.std(Q) + 1e-12)
    quad = np.mean(I * Q) / (np.sqrt(np.mean(I ** 2) * np.mean(Q ** 2)) + 1e-12)
    r = np.sum(x[1:] * np.conj(x[:-1]))
    cfo = FS / (2 * np.pi) * np.angle(r)
    papr = 10 * np.log10(np.max(np.abs(x) ** 2) / (np.mean(np.abs(x) ** 2) + 1e-12))
    return dc, gimb, quad, cfo, papr


def device_signature(d, nframes):
    cap = f"{ROOT}/device_{d}/adv_run_3.bin"
    if not os.path.exists(cap):
        return None
    frames = extract_frames_for_file(cap)[0][:nframes]
    S = np.array([frame_stats(f) for f in frames])   # [N,5]
    return S.mean(0), S.std(0), len(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nframes', type=int, default=200)
    ap.add_argument('--target', type=int, default=4)
    a = ap.parse_args()
    cols = ["DC", "IQgain", "IQquad", "CFO_Hz", "PAPR_dB"]
    print(f"Per-device impairment signature (6/30 run-3, {a.nframes} frames, unit-RMS)\n")
    print(f"  {'device':>8}  " + "".join(f"{c:>12}" for c in cols) + f"  {'n':>5}")
    sigs = {}
    for d in range(1, NUM_CLASSES + 1):
        r = device_signature(d, a.nframes)
        if r is None:
            continue
        mu, sd, n = r
        sigs[d] = mu
        mark = "  <- TARGET" if d == a.target else ""
        print(f"  device_{d}  " + "".join(f"{mu[i]:>12.4f}" for i in range(5)) + f"  {n:>5}{mark}")
    # within-device spread (for the target) vs cross-device spread — is the signature tight & distinct?
    if a.target in sigs:
        allmu = np.array([sigs[d] for d in sigs])
        tgt = sigs[a.target]
        print(f"\n  device_{a.target} mimicry target vs field (|target - field mean| / field std):")
        fmean, fstd = allmu.mean(0), allmu.std(0) + 1e-12
        for i, c in enumerate(cols):
            z = abs(tgt[i] - fmean[i]) / fstd[i]
            print(f"    {c:>10}: target {tgt[i]:>10.4f}   field {fmean[i]:>10.4f} +/- {fstd[i]:.4f}   z={z:.2f}")
        print("\n  (large |z| = a distinctive, must-reproduce impairment; small = shared, easy)")


if __name__ == '__main__':
    main()

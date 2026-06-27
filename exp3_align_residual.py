#!/usr/bin/env python3
"""
exp3_align_residual.py — did the perturbation land coherently on the recaptured bursts?
────────────────────────────────────────────────────────────────────────────────────────
For an attacked recapture, detect bursts, sync each to its transmitted frame
template via a preamble matched-filter (refined lag + CFO + complex channel gain
estimated from the FRAME — delta rides the same channel), then jointly LS-fit the
data region onto [frame_k, delta_k]:   y ≈ a·frame_k + b·delta_k.

Because delta_k is the psr_0 perturbation (PSR 0 dB by construction, ||d||≈||f||),
the realized receiver PSR is  20·log10(|b|·||d|| / (|a|·||f||)).  Design PSR for the
psr_0 file is 0 dB: if delta landed coherently, b/a≈1 (~0 dB); if it never made it
over the air, b collapses to the noise floor (very negative dB).
"""
import sys
import numpy as np

C64 = np.complex64
GAP_DIR = "/media/nghoselab/T9/Data/session13/ota_dev6/dac_safe_gapped"
PERIOD = 265661
FL = 15661
PRE = 500            # shared preamble length
DS, DE = 100, 14436  # delta / data-region span within the active frame
CHUNK = 40_000_000   # samples of the recapture to scan (~8 s)


def load_templates(psr_tag):
    f = np.fromfile(f"{GAP_DIR}/adv_frame.bin", dtype=C64)[:91*PERIOD].reshape(91, PERIOD)[:, :FL]
    d = np.fromfile(f"{GAP_DIR}/adv_perturbation_psr_{psr_tag}.bin",
                    dtype=C64)[:91*PERIOD].reshape(91, PERIOD)[:, :FL]
    return f, d


def detect_bursts(x, win=512, thr_k=0.25, min_len=12000, max_n=24):
    """coarse energy burst onsets (same spirit as the frame extractor)."""
    p = np.abs(x)**2
    # smooth power
    c = np.cumsum(p)
    mp = (c[win:] - c[:-win]) / win
    floor = np.percentile(mp, 20)
    peak = np.percentile(mp, 99.5)
    thr = floor + thr_k * (peak - floor)
    on = mp > thr
    onsets = []
    i = 0
    n = len(on)
    while i < n and len(onsets) < max_n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if j - i >= min_len:
                onsets.append(i)
            i = j + win
        else:
            i += 1
    return onsets


def best_cfo_lag(seg, pre, lag_win=300):
    """over a small lag window, FFT(seg*conj(pre)) gives per-lag best CFO + score."""
    best = (-1.0, 0, 0.0)  # score, lag, cfo_norm
    L = len(pre)
    for lag in range(0, lag_win):
        s = seg[lag:lag+L]
        if len(s) < L:
            break
        sp = np.fft.fft(s * np.conj(pre))
        k = np.argmax(np.abs(sp))
        sc = np.abs(sp[k])
        if sc > best[0]:
            cfo = (k if k < L//2 else k - L) / L   # cycles/sample
            best = (sc, lag, cfo)
    return best


def analyze(att_path, psr_tag, label):
    F, D = load_templates(psr_tag)
    pre = F[0, :PRE]                      # shared preamble (clean-frame copy)
    x = np.fromfile(att_path, dtype=C64, count=CHUNK)
    onsets = detect_bursts(x)
    print(f"\n=== {label}  ({att_path.split('/')[-1]}, design PSR per file) ===")
    print(f"scanned {len(x)/5e6:.1f}s, {len(onsets)} candidate bursts; "
          f"per-burst realized RX delta/frame ratio:")
    print(f"{'#':>2} {'lag':>4} {'CFOhz':>7} {'frame_k':>7} {'matchR':>6} "
          f"{'RX_PSR_dB':>9} {'noisefloor_dB':>13}")
    rows = []
    for bi, o in enumerate(onsets):
        seg = x[o-50: o-50 + FL + 350]    # a little slack before/after
        if len(seg) < FL + 300:
            continue
        sc, lag, cfo = best_cfo_lag(seg, pre)
        n = np.arange(FL)
        derot = seg[lag:lag+FL] * np.exp(-1j*2*np.pi*cfo*n)
        # identify frame by data-region correlation
        yd = derot[DS:DE]
        yd0 = yd - yd.mean()
        corr = np.array([np.abs(np.vdot(F[k, DS:DE] - F[k, DS:DE].mean(), yd0)) /
                         (np.linalg.norm(F[k, DS:DE]-F[k, DS:DE].mean())*np.linalg.norm(yd0)+1e-12)
                         for k in range(91)])
        k = int(np.argmax(corr))
        f = F[k, DS:DE]; d = D[k, DS:DE]
        A = np.stack([f, d], axis=1)            # (N,2)
        G = A.conj().T @ A
        c = np.linalg.solve(G, A.conj().T @ yd)
        a, b = c
        resid = yd - A @ c
        rx_psr = 20*np.log10(abs(b)*np.linalg.norm(d) /
                             (abs(a)*np.linalg.norm(f) + 1e-12) + 1e-12)
        # noise floor: project residual energy onto a delta-shaped unit => equivalent dB
        noise_db = 20*np.log10(np.linalg.norm(resid)/np.sqrt(len(resid)) *
                               np.linalg.norm(d) / (abs(a)*np.linalg.norm(f)/np.sqrt(len(f)) + 1e-12) + 1e-12)
        cfo_hz = cfo*5e6
        print(f"{bi:>2} {lag:>4} {cfo_hz:>7.0f} {k:>7} {corr[k]:>6.3f} "
              f"{rx_psr:>9.1f} {noise_db:>13.1f}")
        rows.append((corr[k], rx_psr, noise_db))
    if rows:
        r = np.array(rows)
        good = r[r[:,0] > 0.5]
        if len(good):
            print(f"\n  bursts with clean frame match (R>0.5): {len(good)}/{len(rows)}")
            print(f"  median frame match R     : {np.median(good[:,0]):.3f}")
            print(f"  median realized RX PSR   : {np.median(good[:,1]):+.1f} dB   "
                  f"(design for this file's delta)")
            print(f"  median residual-noise dB : {np.median(good[:,2]):+.1f} dB")
    return rows


if __name__ == "__main__":
    AD = "/media/nghoselab/T9/Data/session13/attacked/device_6"
    analyze(f"{AD}/adv_psr_0.bin",  "0",   "PSR 0 dB  (delta == frame power; LOUDEST)")
    analyze(f"{AD}/adv_psr_30.bin", "m30", "PSR -30 dB (control; delta ~1000x weaker)")

#!/usr/bin/env python3
"""Magnitude-domain (CFO-invariant) match of recaptured bursts vs the 91 TX frame
templates. If even |.|-correlation is ~0, the recapture doesn't contain the
replayed frames in a template-matchable form (replay chain broken upstream).
Also reports data-region vs clean-preamble power ratio (delta would inflate it)."""
import sys
import numpy as np

C64 = np.complex64
GAP_DIR = "/media/nghoselab/T9/Data/session13/ota_dev6/dac_safe_gapped"
PERIOD = 265661; FL = 15661; PRE0 = 100; DATA0, DATA1 = 500, 14436

F = np.fromfile(f"{GAP_DIR}/adv_frame.bin", dtype=C64)[:91*PERIOD].reshape(91, PERIOD)[:, :FL]
Fmag = np.abs(F)
Fmag_n = (Fmag - Fmag.mean(1, keepdims=True))
Fmag_n /= (np.linalg.norm(Fmag_n, axis=1, keepdims=True) + 1e-12)


def detect(x, win=512, thr_k=0.20, min_len=11000, max_n=12):
    p = np.abs(x)**2
    c = np.cumsum(p); mp = (c[win:] - c[:-win]) / win
    floor = np.percentile(mp, 20); peak = np.percentile(mp, 99.5)
    thr = floor + thr_k*(peak-floor); on = mp > thr
    out = []; i = 0; n = len(on)
    while i < n and len(out) < max_n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if j-i >= min_len:
                out.append(i)
            i = j+win
        else:
            i += 1
    return out


def probe(path, tag):
    x = np.fromfile(path, dtype=C64, count=40_000_000)
    ons = detect(x)
    print(f"\n=== {tag}  ({path.split('/')[-1]})  {len(ons)} bursts ===")
    print(f"{'#':>2} {'best_k':>6} {'magR':>6} {'lag':>5} {'data/pre_pow':>12}")
    for bi, o in enumerate(ons):
        seg = np.abs(x[o-50: o-50+FL+400])
        if len(seg) < FL+300:
            continue
        # magnitude xcorr over lag window vs each template (FFT correlation)
        bestR, bestk, bestlag = -1, -1, 0
        for lag in range(0, 300, 3):
            s = seg[lag:lag+FL]
            sn = s - s.mean(); nn = np.linalg.norm(sn)+1e-12
            r = Fmag_n @ (sn/nn)
            k = int(np.argmax(r))
            if r[k] > bestR:
                bestR, bestk, bestlag = float(r[k]), k, lag
        # power: clean preamble [0:100] vs data region [500:14436], using best lag
        s = x[o-50+bestlag: o-50+bestlag+FL]
        pre_pow = np.mean(np.abs(s[0:PRE0])**2)
        dat_pow = np.mean(np.abs(s[DATA0:DATA1])**2)
        print(f"{bi:>2} {bestk:>6} {bestR:>6.3f} {bestlag:>5} {dat_pow/ (pre_pow+1e-12):>12.3f}")


if __name__ == "__main__":
    AD = "/media/nghoselab/T9/Data/session13/attacked/device_6"
    probe(f"{AD}/adv_psr_30.bin", "PSR -30 dB (delta ~1000x weaker)")
    probe(f"{AD}/adv_psr_0.bin",  "PSR 0 dB   (delta == frame power)")

#!/usr/bin/env python3
"""
exp3_delay_pad.py — measure the device_6-vs-adversary arrival delay and pre-pad the
adversary's δ so it lands on device_6's payload every frame (fixes the 2-channel
dose problem: intermittent overlap -> weak BER, so δ never got a full dose).

Idea (per advisor): in a 2-channel attack the two radios' signals arrive at the RX
at DIFFERENT times (Δ = chain+propagation difference). Capture each signal ALONE,
cross-correlate to find Δ, then shift the adversary's ch1 by −Δ at TX so after the
Δ delay it superimposes on device_6's payload.

CALIBRATION CAPTURES (each signal alone, adversary transmitting the SAME adv_frame):
  cap_dev6.bin : device_6 replaying adv_frame,   adversary muted
  cap_adv.bin  : adversary replaying adv_frame,   device_6 muted
Both contain identical frame content, so cross-correlation gives the relative delay.

Usage:
  # 1) measure only
  python3 exp3_delay_pad.py --cap-dev6 cap_dev6.bin --cap-adv cap_adv.bin
  # 2) measure + build an aligned 2-channel bundle (shift the ch1 δ files)
  python3 exp3_delay_pad.py --cap-dev6 cap_dev6.bin --cap-adv cap_adv.bin \
      --pert-dir .../targeted/dac_safe_eot --out .../targeted/dac_safe_eot_aligned
  # 3) override the delay (e.g. from a scope) and just pad
  python3 exp3_delay_pad.py --delay 1234 --pert-dir ... --out ...

ASSUMPTION: Δ is repeatable — the two USRPs share a trigger / 10 MHz+PPS reference.
If they free-run, Δ drifts between runs and a single pad won't hold.
"""
import os, glob, shutil, argparse
import numpy as np

C64 = np.complex64
FS = 5_000_000


def _envelope(x, win=64):
    """coarse power envelope (CFO-invariant), decimated by `win`."""
    p = (x.real.astype(np.float64) ** 2 + x.imag.astype(np.float64) ** 2)
    m = (len(p) // win) * win
    return p[:m].reshape(-1, win).mean(1)


def measure_delay(cap_dev6, cap_adv, nsamp=8_000_000, search_ms=8.0, win=64):
    """Return (delay_samples, confidence). delay>0 => adversary arrives LATER than
    device_6 (so its δ must be advanced by `delay` at TX)."""
    c6 = np.fromfile(cap_dev6, dtype=C64, count=nsamp)
    ca = np.fromfile(cap_adv, dtype=C64, count=nsamp)
    e6 = _envelope(c6, win)
    ea = _envelope(ca, win)
    n = min(len(e6), len(ea))
    e6 = e6[:n] - e6[:n].mean()
    ea = ea[:n] - ea[:n].mean()
    # FFT circular cross-correlation of the (zero-mean) envelopes
    L = 1
    while L < 2 * n:
        L <<= 1
    corr = np.fft.irfft(np.fft.rfft(ea, L) * np.conj(np.fft.rfft(e6, L)), L)
    # lags: positive = ea delayed vs e6. search a symmetric window.
    smax = int(search_ms * 1e-3 * FS / win)
    lags = np.concatenate([np.arange(0, smax), np.arange(L - smax, L)])
    vals = corr[lags]
    k = int(np.argmax(vals))
    peak_lag = lags[k]
    delay_env = peak_lag if peak_lag < smax else peak_lag - L        # signed, in env bins
    conf = float(vals[k] / (np.median(np.abs(corr)) + 1e-30))
    return int(delay_env * win), conf


def shift(x, d):
    """delay x by d samples (d>0 = later). Zero-pad, no wraparound, same length."""
    out = np.zeros(len(x), dtype=C64)
    if d >= 0:
        out[d:] = x[:len(x) - d] if d < len(x) else 0
    else:
        out[:len(x) + d] = x[-d:]
    return out


def build_aligned(pert_dir, out, delay):
    """copy adv_frame.bin unchanged; advance every adv_perturbation_psr_*.bin by `delay`
    (shift EARLIER by delay, so after the Δ arrival delay it lands on the payload)."""
    os.makedirs(out, exist_ok=True)
    fr = os.path.join(pert_dir, 'adv_frame.bin')
    if os.path.exists(fr):
        shutil.copy(fr, os.path.join(out, 'adv_frame.bin'))
    perts = sorted(glob.glob(os.path.join(pert_dir, 'adv_perturbation_psr_*.bin')))
    for p in perts:
        x = np.fromfile(p, dtype=C64)
        shift(x, -delay).astype(C64).tofile(os.path.join(out, os.path.basename(p)))
    print(f"  aligned bundle -> {out}/  (adv_frame + {len(perts)} shifted pert files, "
          f"δ advanced by {delay} samp)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap-dev6', help='device_6-alone calibration capture')
    ap.add_argument('--cap-adv', help='adversary-alone calibration capture (replaying adv_frame)')
    ap.add_argument('--delay', type=int, default=None,
                    help='override measured delay (samples); adversary-later = positive')
    ap.add_argument('--search-ms', type=float, default=8.0, help='max |delay| to search (ms)')
    ap.add_argument('--period', type=int, default=15661,
                    help='frame period (samples) — delay is only defined mod this; reduced to principal value')
    ap.add_argument('--pert-dir', help='dac_safe_eot dir (adv_frame + adv_perturbation_psr_*)')
    ap.add_argument('--out', help='output dir for the aligned bundle')
    a = ap.parse_args()

    if a.delay is not None:
        delay = a.delay
        print(f"using override delay Δ = {delay} samp ({delay/FS*1e6:+.1f} µs)")
    else:
        if not (a.cap_dev6 and a.cap_adv):
            raise SystemExit("give --cap-dev6 and --cap-adv (or --delay)")
        raw, conf = measure_delay(a.cap_dev6, a.cap_adv, search_ms=a.search_ms)
        # frames loop, so delay is only defined mod the frame period: reduce to the
        # smallest-magnitude equivalent shift in (-period/2, +period/2].
        delay = ((raw % a.period) + a.period) % a.period
        if delay > a.period // 2:
            delay -= a.period
        print(f"measured delay (raw {raw}) -> principal Δ = {delay} samp ({delay/FS*1e6:+.1f} µs, "
              f"mod frame period {a.period})  [adversary {'LATER' if delay>0 else 'EARLIER'} than device_6]  "
              f"xcorr peak/median = {conf:.1f} {'(strong)' if conf>5 else '(WEAK — check captures)'}")

    if a.pert_dir and a.out:
        build_aligned(a.pert_dir, a.out, delay)
        print("transmit the aligned bundle as before (ch0=adv_frame, ch1=adv_perturbation_psr_X);\n"
              "then re-run the BER gate — payload BER should now climb toward the single-channel dose (~0.13).")
    elif a.pert_dir or a.out:
        print("(give BOTH --pert-dir and --out to build the aligned bundle)")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
exp3_delay_pad.py — measure the device_6-vs-adversary arrival delay and pre-pad the
adversary's delta so it lands on device_6's payload every frame (fixes the 2-channel
dose problem: intermittent overlap -> weak BER, so delta never got a full dose).

Idea: in a 2-channel attack the two radios' signals arrive at the RX at DIFFERENT times
(Delta = chain+propagation difference). Measure Delta, then shift the adversary's ch1 by
-Delta at TX so after the Delta delay it superimposes on device_6's payload.

TWO WAYS TO MEASURE Delta
-------------------------
(A) single simultaneous capture  [RECOMMENDED — no cross-capture referencing]
    Transmit ch0=adv_frame + ch1=MARKER at the same time (the normal 2-channel path),
    capture ONCE. One time reference => no manual-start offset can pollute Delta, and the
    result transfers to the attack exactly because the marker rode ch1 like delta will.
      python3 exp3_delay_pad.py --make-marker marker.bin --ref adv_frame.bin
      # (transmit ch0=adv_frame, ch1=marker simultaneously, capture -> cap_single.bin)
      python3 exp3_delay_pad.py --cap-single cap_single.bin --frame-ref adv_frame.bin \
          --marker-ref marker.bin --pert-dir .../dac_safe --out .../dac_safe_aligned

(B) two separate captures  [legacy — needs a common timed start or Delta is meaningless]
    cap_dev6.bin : device_6 replaying adv_frame,  adversary muted
    cap_adv.bin  : adversary replaying adv_frame,  device_6 muted
      python3 exp3_delay_pad.py --cap-dev6 cap_dev6.bin --cap-adv cap_adv.bin

(C) override:  --delay N  (adversary-later = positive) then --pert-dir/--out to build.

THE SHIFT IS PER-FRAME (mod the frame period): delta is periodic with the frame, so the
whole-stream slide used previously wrapped delta across frame boundaries and de-confined it
from the payload. build_aligned() now rolls each frame block circularly, keeping delta
payload-confined for any Delta.
"""
import os, glob, shutil, argparse
import numpy as np

C64 = np.complex64
FS = 5_000_000
PERIOD = 15661            # frame period (samples): adv_frame.bin is exactly 200 x 15661


def _envelope(x, win=64):
    """coarse power envelope (CFO-invariant), decimated by `win`."""
    p = (x.real.astype(np.float64) ** 2 + x.imag.astype(np.float64) ** 2)
    m = (len(p) // win) * win
    return p[:m].reshape(-1, win).mean(1)


def _principal_lag(rx_env, ref_env, win, period, search_ms):
    """Signed sample-lag of ref within rx by envelope FFT cross-correlation, reduced to the
    principal value in (-period/2, +period/2].  lag>0 => ref arrives LATER than rx origin."""
    n = min(len(rx_env), len(ref_env))
    a = rx_env[:n] - rx_env[:n].mean()
    b = ref_env[:n] - ref_env[:n].mean()
    L = 1
    while L < 2 * n:
        L <<= 1
    corr = np.fft.irfft(np.fft.rfft(a, L) * np.conj(np.fft.rfft(b, L)), L)
    smax = int(search_ms * 1e-3 * FS / win)
    lags = np.concatenate([np.arange(0, smax), np.arange(L - smax, L)])
    vals = corr[lags]
    k = int(np.argmax(vals))
    peak = lags[k]
    lag_bins = peak if peak < smax else peak - L
    conf = float(vals[k] / (np.median(np.abs(corr)) + 1e-30))
    raw = int(lag_bins * win)
    pr = ((raw % period) + period) % period
    if pr > period // 2:
        pr -= period
    return pr, conf


# ----- (A) single simultaneous capture -------------------------------------------------
def make_marker(out, n_samples, period=PERIOD, offset=14800, mlen=256, amp=0.7):
    """Write a ch1 marker file: a short chirp burst (rectangular envelope) placed in the
    frame's silent GAP at `offset`, zero elsewhere -> detected with no frame contamination."""
    x = np.zeros(n_samples, dtype=C64)
    t = np.arange(mlen)
    burst = (amp * np.exp(1j * 2 * np.pi * (0.15 * t + 0.5 * 6e-4 * t ** 2))).astype(C64)
    nf = n_samples // period
    for k in range(nf):
        s = k * period + offset
        x[s:s + mlen] = burst
    x.tofile(out)
    print(f"  marker -> {out}  ({nf} bursts of {mlen} samp @ offset {offset} in the gap, "
          f"amp {amp}, {n_samples} samp total)")
    return offset


def measure_delay_single(cap, frame_ref, marker_ref,
                         period=PERIOD, nsamp=8_000_000, search_ms=8.0, win=16):
    """From ONE simultaneous capture, return (Delta, conf_frame, conf_marker). Delta = the
    ch1(adversary)-vs-ch0(device_6) arrival delay to compensate: shift ch1 by -Delta.

    frame-ref and marker-ref each carry their content at its designed frame position, so the
    lag of each against the capture is that content's channel delay (phi_f = d0 for ch0,
    phi_m = d0+Delta for ch1). The designed positions cancel in the difference:
        Delta = phi_m - phi_f    (reduced to the principal value mod the frame period)."""
    rx = _envelope(np.fromfile(cap, dtype=C64, count=nsamp), win)
    fr = _envelope(np.fromfile(frame_ref, dtype=C64), win)
    mk = _envelope(np.fromfile(marker_ref, dtype=C64), win)
    phi_f, cf = _principal_lag(rx, fr, win, period, search_ms)   # frame (ch0) arrival phase
    phi_m, cm = _principal_lag(rx, mk, win, period, search_ms)   # marker (ch1) arrival phase
    d = phi_m - phi_f
    d = ((d % period) + period) % period
    if d > period // 2:
        d -= period
    return int(d), cf, cm


# ----- (B) two-capture (legacy) --------------------------------------------------------
def measure_delay(cap_dev6, cap_adv, nsamp=8_000_000, search_ms=8.0, win=64, period=PERIOD):
    c6 = _envelope(np.fromfile(cap_dev6, dtype=C64, count=nsamp), win)
    ca = _envelope(np.fromfile(cap_adv, dtype=C64, count=nsamp), win)
    d, conf = _principal_lag(ca, c6, win, period, search_ms)
    return d, conf


# ----- shift + build -------------------------------------------------------------------
def build_aligned(pert_dir, out, delay, period=PERIOD):
    """copy adv_frame.bin unchanged; advance every adv_perturbation_psr_*.bin by `delay`
    PER FRAME (circular roll within each `period` block), so delta stays payload-confined
    instead of wrapping across frame boundaries."""
    os.makedirs(out, exist_ok=True)
    fr = os.path.join(pert_dir, 'adv_frame.bin')
    if os.path.exists(fr):
        shutil.copy(fr, os.path.join(out, 'adv_frame.bin'))
    perts = sorted(glob.glob(os.path.join(pert_dir, 'adv_perturbation_psr_*.bin')))
    for p in perts:
        x = np.fromfile(p, dtype=C64)
        n = (len(x) // period) * period
        if n:
            blk = x[:n].reshape(-1, period)
            x[:n] = np.roll(blk, -delay, axis=1).reshape(-1)   # per-frame, delta stays in-frame
        x.astype(C64).tofile(os.path.join(out, os.path.basename(p)))
    print(f"  aligned bundle -> {out}/  (adv_frame + {len(perts)} pert files, "
          f"delta advanced {delay} samp PER FRAME, payload-confined)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--make-marker', help='write a ch1 marker file (needs --ref for length)')
    ap.add_argument('--ref', help='adv_frame.bin (used to size the marker)')
    ap.add_argument('--marker-offset', type=int, default=14800, help='marker position in the frame gap')
    ap.add_argument('--marker-len', type=int, default=256)
    ap.add_argument('--marker-amp', type=float, default=0.7)

    ap.add_argument('--cap-single', help='ONE simultaneous capture (ch0=frame + ch1=marker)')
    ap.add_argument('--frame-ref', help='adv_frame.bin reference')
    ap.add_argument('--marker-ref', help='marker.bin reference (same one transmitted)')

    ap.add_argument('--cap-dev6', help='(legacy) device_6-alone capture')
    ap.add_argument('--cap-adv', help='(legacy) adversary-alone capture (replaying adv_frame)')

    ap.add_argument('--delay', type=int, default=None, help='override Delta (samples); adversary-later positive')
    ap.add_argument('--search-ms', type=float, default=8.0)
    ap.add_argument('--period', type=int, default=PERIOD)
    ap.add_argument('--pert-dir', help='dac_safe dir (adv_frame + adv_perturbation_psr_*)')
    ap.add_argument('--out', help='output dir for the aligned bundle')
    a = ap.parse_args()

    # marker generation is a standalone action
    if a.make_marker:
        if not a.ref:
            raise SystemExit("--make-marker needs --ref adv_frame.bin (to size it)")
        nsamp = os.path.getsize(a.ref) // 8
        off = make_marker(a.make_marker, nsamp, a.period, a.marker_offset, a.marker_len, a.marker_amp)
        print(f"transmit ch0={os.path.basename(a.ref)} + ch1={os.path.basename(a.make_marker)} "
              f"SIMULTANEOUSLY, capture once, then re-run with --cap-single.")
        return

    if a.delay is not None:
        delay = a.delay
        print(f"using override delay Delta = {delay} samp ({delay/FS*1e6:+.1f} us)")
    elif a.cap_single:
        if not (a.frame_ref and a.marker_ref):
            raise SystemExit("--cap-single needs --frame-ref and --marker-ref")
        delay, cf, cm = measure_delay_single(a.cap_single, a.frame_ref, a.marker_ref,
                                             a.period, search_ms=a.search_ms)
        print(f"single-capture Delta = {delay} samp ({delay/FS*1e6:+.1f} us, mod frame {a.period})  "
              f"[adversary {'LATER' if delay>0 else 'EARLIER'} than device_6]")
        print(f"  frame xcorr peak/median = {cf:.1f} {'(strong)' if cf>5 else '(WEAK)'}; "
              f"marker peak/median = {cm:.1f} {'(strong)' if cm>5 else '(WEAK — is ch1/marker radiating?)'}")
    elif a.cap_dev6 and a.cap_adv:
        delay, conf = measure_delay(a.cap_dev6, a.cap_adv, search_ms=a.search_ms, period=a.period)
        print(f"two-capture Delta = {delay} samp ({delay/FS*1e6:+.1f} us, mod frame {a.period})  "
              f"[adversary {'LATER' if delay>0 else 'EARLIER'} than device_6]  "
              f"xcorr peak/median = {conf:.1f} {'(strong)' if conf>5 else '(WEAK)'}")
        print("  NOTE: two-capture Delta is only valid if both were started from a COMMON timed "
              "reference; a manual start makes it meaningless. Prefer --cap-single.")
    else:
        raise SystemExit("give --cap-single (+refs), or --cap-dev6/--cap-adv, or --delay, "
                         "or --make-marker")

    if a.pert_dir and a.out:
        build_aligned(a.pert_dir, a.out, delay, a.period)
        print("transmit the aligned bundle (ch0=adv_frame, ch1=adv_perturbation_psr_X); then "
              "re-run the BER gate — payload BER should climb toward the single-channel dose (~0.13).")
    elif a.pert_dir or a.out:
        print("(give BOTH --pert-dir and --out to build the aligned bundle)")


if __name__ == '__main__':
    main()

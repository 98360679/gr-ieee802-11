#!/usr/bin/env python3
"""
exp3_loopback_sim.py — predict how MIMO ch0/ch1 misalignment kills the attack
────────────────────────────────────────────────────────────────────────────────
Software stand-in for the loopback test. The targeted attack works digitally
(device_6 -> device_4, 92% @ PSR -20) when frame and delta are PERFECTLY aligned.
Over the air it did nothing — the leading hypothesis is that the 2-channel MIMO
adversary TX doesn't deliver delta (ch1) sample/phase-aligned to the frame (ch0).

This crafts the targeted per-frame delta, then re-combines  frame + impaired(delta)
under controlled impairments and measures fooling (off device_6) and target-hit
(-> device_4):
  * TIMING skew   : delta shifted k samples vs the frame (ch0/ch1 time misalign)
  * PHASE offset  : constant phase rotation of delta vs the frame
  * INCOHERENT    : independent RANDOM phase per frame (MIMO LOs not locked)
  * NOISE floor   : additive noise (sanity: attack should survive mild noise)

Read the timing/phase columns as the sync TOLERANCE the hardware must hit. If a
1-2 sample skew or an unlocked phase already collapses target-hit to ~0, that
explains the OTA null and says the fix is TX MIMO time+phase locking.
"""
import numpy as np
import torch

from exp3_make_perturbation import load_fp_model, craft, predict_frame
from exp3_train_fingerprint import load_frames
from exp3_fp_model import FRAME_LEN

MODEL = "fingerprint_cnn_ft20260627.pt"
SRC, TGT = 6, 4
PSR = -20.0
N_FRAMES = 25
SEED = 0


def rate(model, frames, deltas, true, tgt, nc, impair):
    """impair(delta, i) -> impaired delta for frame i. Returns (off%, hit%)."""
    off = hit = 0
    for i, (f, d) in enumerate(zip(frames, deltas)):
        dd = impair(d, i)
        pred, _ = predict_frame(model, f + dd, nc)
        off += (pred != true); hit += (pred == tgt)
    n = len(frames)
    return 100.0 * off / n, 100.0 * hit / n


def main():
    rng = np.random.default_rng(SEED)
    model, nc, n2i, i2n = load_fp_model(MODEL)
    true, tgt = n2i[f'device_{SRC}'], n2i[f'device_{TGT}']
    print(f"{MODEL}: device_{SRC}->device_{TGT}  PSR {PSR} dB  "
          f"{N_FRAMES} frames\n")

    frames_all, run = load_frames(SRC)
    idx = np.where(run == 3)[0]
    # keep frames the model calls device_6 cleanly, craft targeted delta for each
    frames, deltas = [], []
    for i in idx:
        if len(frames) >= N_FRAMES:
            break
        pred, _ = predict_frame(model, frames_all[i], nc)
        if pred == true:
            d, _ = craft(model, frames_all[i], true, PSR, 100, 0.1, tgt)
            frames.append(frames_all[i]); deltas.append(d)
    print(f"crafted targeted delta for {len(frames)} clean device_{SRC} frames\n")

    # baseline: perfect alignment
    o, h = rate(model, frames, deltas, true, tgt, nc, lambda d, i: d)
    print(f"PERFECT alignment:           off {o:5.0f}%   ->device_{TGT} {h:5.0f}%")

    print("\nTIMING skew (delta shifted vs frame):")
    print(f"{'samples':>8} {'off%':>6} {'->dev'+str(TGT)+'%':>8}")
    for k in [0, 1, 2, 4, 8, 16, 32, 64]:
        o, h = rate(model, frames, deltas, true, tgt, nc,
                    lambda d, i, k=k: np.roll(d, k))
        print(f"{k:>8} {o:>6.0f} {h:>8.0f}")

    print("\nPHASE offset (constant rotation of delta):")
    print(f"{'deg':>8} {'off%':>6} {'->dev'+str(TGT)+'%':>8}")
    for deg in [0, 15, 30, 45, 90, 135, 180]:
        ph = np.exp(1j * np.deg2rad(deg)).astype(np.complex64)
        o, h = rate(model, frames, deltas, true, tgt, nc,
                    lambda d, i, ph=ph: d * ph)
        print(f"{deg:>8} {o:>6.0f} {h:>8.0f}")

    # incoherent: independent random phase per frame (LOs not locked)
    rand_ph = np.exp(1j * rng.uniform(0, 2*np.pi, len(frames))).astype(np.complex64)
    o, h = rate(model, frames, deltas, true, tgt, nc,
                lambda d, i: d * rand_ph[i])
    print(f"\nINCOHERENT (random phase/frame): off {o:5.0f}%   "
          f"->device_{TGT} {h:5.0f}%")

    print("\nADDITIVE NOISE (per-window SNR; attack should survive mild noise):")
    print(f"{'SNR dB':>8} {'off%':>6} {'->dev'+str(TGT)+'%':>8}")
    for snr in [40, 30, 20, 10]:
        def add_noise(d, i, snr=snr):
            f = frames[i]
            p = np.mean(np.abs(f)**2)
            npow = p * 10 ** (-snr / 10)
            nz = (rng.standard_normal(FRAME_LEN) +
                  1j*rng.standard_normal(FRAME_LEN)) * np.sqrt(npow/2)
            return d + nz.astype(np.complex64)
        o, h = rate(model, frames, deltas, true, tgt, nc, add_noise)
        print(f"{snr:>8} {o:>6.0f} {h:>8.0f}")

    print("\nInterpretation: the smallest skew/phase that drops ->device_%d to ~0 "
          "is the\nMIMO sync tolerance the loopback hardware must beat." % TGT)


if __name__ == '__main__':
    main()

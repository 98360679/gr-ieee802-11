#!/usr/bin/env python3
"""
exp3_delta_survival.py — Exp 2 mechanism test: did the δ SURVIVE device_6's PA, or was it erased?

Given two OTA recaptures through the SAME transmitter, captured back-to-back (static channel,
same gain):
    R_off = frame only (δ-off)
    R_on  = frame + δ
this isolates the δ that physically came out of the PA and asks two things:
  M2  residual = R_on - a*R_off (per matched frame, a = best complex scale): its ENERGY vs the
      transmitted δ tells us if the δ survived; classifying the residual tells us whose
      fingerprint the *surviving* δ wears.
  M1  matched filter: correlate the residual against the known transmitted δ (optional --delta);
      a peak (vs a noise-floor control) = the δ came through with its crafted shape.

Interpretation:
  δ SURVIVES (energy ~ transmitted, residual reads device_6) AND fooling=0  -> fingerprint
    barrier PROVEN: δ present but outweighed by device_6's imprint.
  δ does NOT survive (residual ~ noise floor)                               -> 0% was δ
    fragility, not the barrier.

  python3 exp3_delta_survival.py --r-on R_on.bin --r-off R_off.bin --model <m.pt> \
      [--delta adv_perturbation.bin] [--period 265661] [--psr -5]
"""
import argparse, numpy as np
from exp3_make_perturbation import load_fp_model
from exp3_extract_frames import extract_frames_for_file
from exp3_fp_model import frame_to_windows, iq_to_input, PRE_ROLL, ACTIVE, WIN, NUM_CLASSES

C64 = np.complex64


def frames_by_content(path, period, thr_mult, floor_pct):
    fr, st = extract_frames_for_file(path, floor_pct=floor_pct, thr_mult=thr_mult)[:2]
    st = np.asarray(st, float)
    cid = (np.round((st - st[0]) / period).astype(int)) % 222 if len(st) else np.array([])
    return {int(c): np.asarray(f, C64) for c, f in zip(cid, fr)}   # one frame per content id


def align_scale(on, off):
    """coarse integer-lag align off->on (xcorr on active region) then best complex scale a."""
    a0, a1 = PRE_ROLL, PRE_ROLL + ACTIVE
    x, y = on[a0:a1], off[a0:a1]
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    xc = np.correlate(x, y, mode='same')
    lag = int(np.argmax(np.abs(xc)) - n // 2)
    yb = np.roll(off, lag)[a0:a1][:n]
    a = np.vdot(yb, x) / (np.vdot(yb, yb) + 1e-12)      # least-squares complex scale
    resid = x - a * yb
    return resid, x, a, lag


def classify(model, nc, frame_active):
    """windows over an active-region snippet -> majority device."""
    padded = np.zeros(PRE_ROLL + ACTIVE + (WIN), C64)
    padded[PRE_ROLL:PRE_ROLL + len(frame_active)] = frame_active[:ACTIVE]
    w = frame_to_windows(padded)
    import torch
    dev = next(model.parameters()).device
    with torch.no_grad():
        p = model(torch.from_numpy(iq_to_input(w)).to(dev)).argmax(1).cpu().numpy()
    return int(np.bincount(p, minlength=nc).argmax()), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r-on', required=True); ap.add_argument('--r-off', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--delta', default=None, help='known transmitted δ (optional, for M1)')
    ap.add_argument('--period', type=int, default=265661)
    ap.add_argument('--psr', type=float, default=None, help='transmitted PSR (dB) for expected δ energy')
    ap.add_argument('--thr-mult', type=float, default=2.0); ap.add_argument('--floor-pct', type=float, default=20.0)
    ap.add_argument('--max', type=int, default=200)
    a = ap.parse_args()
    m, nc, n2i, i2n = load_fp_model(a.model)

    ON = frames_by_content(a.r_on, a.period, a.thr_mult, a.floor_pct)
    OFF = frames_by_content(a.r_off, a.period, a.thr_mult, a.floor_pct)
    shared = sorted(set(ON) & set(OFF))[:a.max]
    print(f"R_on {len(ON)} frames, R_off {len(OFF)} frames, {len(shared)} content-matched\n")

    efrac, rcls, ocls = [], [], []
    for c in shared:
        resid, x, a_sc, lag = align_scale(ON[c], OFF[c])
        efrac.append(np.sum(np.abs(resid) ** 2) / (np.sum(np.abs(x) ** 2) + 1e-12))
        rcls.append(classify(m, nc, resid)[0])
        ocls.append(classify(m, nc, ON[c][PRE_ROLL:PRE_ROLL + ACTIVE])[0])
    efrac = np.array(efrac)
    exp_frac = 10 ** (a.psr / 10) if a.psr is not None else None

    print(f"M2 residual energy fraction ||R_on-a·R_off||²/||R_on||²:")
    print(f"   median {np.median(efrac):.4f}  mean {efrac.mean():.4f}"
          + (f"   (expected from PSR {a.psr}dB ≈ {exp_frac:.4f})" if exp_frac else ""))
    def dist(v):
        from collections import Counter
        return {i2n[k]: n for k, n in sorted(Counter(v).items(), key=lambda x:-x[1])}
    print(f"\n   R_on frames read as:     {dist(ocls)}")
    print(f"   RESIDUAL (surviving δ) reads as: {dist(rcls)}")
    print("\nVerdict:")
    survived = (exp_frac is None and np.median(efrac) > 0.02) or (exp_frac and np.median(efrac) > 0.3 * exp_frac)
    if survived:
        print("  δ SURVIVED the PA (residual energy is δ-sized). If fooling≈0 -> barrier PROVEN:")
        print("  the δ is present but outweighed by device_6's imprint.")
    else:
        print("  δ did NOT survive (residual ~ noise floor) -> the 0% is δ fragility, not the barrier.")

    if a.delta:
        d = np.fromfile(a.delta, C64)
        print(f"\n(M1 matched-filter vs {a.delta}: loaded {len(d)} samples — correlate residual vs δ per content offline)")


if __name__ == '__main__':
    main()

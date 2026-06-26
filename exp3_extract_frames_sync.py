#!/usr/bin/env python3
"""
exp3_extract_frames_sync.py — preamble (L-STF autocorrelation) frame extractor
──────────────────────────────────────────────────────────────────────────────
The power-gap detector in exp3_extract_frames assumes frames separated by silence
(the clean ~300 ms-cadence captures). The adversary REPLAY (build_adv_replay)
packs the selected frames back-to-back (~15661 samples apart, no gaps), so a
power detector collapses the whole block to one burst (1 frame extracted).

This extractor finds each frame via the 802.11 L-STF short-training-symbol
AUTOCORRELATION — the same metric the gr-ieee802-11 sync uses. It correlates the
signal with a `lag`-delayed copy of itself; the L-STF's repeating short symbols
make the normalized metric plateau near 1.0 at every frame onset, regardless of
inter-frame gaps, and it's CFO-robust (the offset cancels in the ratio).

Output: onset-aligned FRAME_LEN frames, drop-in for exp3_fp_model windowing — so
exp3_attack_eval / exp3_rebaseline can score dense replay captures.

  python3 exp3_extract_frames_sync.py --file capture.bin            # report count
  python3 exp3_extract_frames_sync.py --file capture.bin --thr 0.6
"""
import os
import argparse
import numpy as np

from exp3_fp_model import FRAME_LEN, PRE_ROLL

STS_LAG = 16        # L-STF short-symbol period (samples, gr-ieee802-11 numerology)
AC_WIN = 48         # autocorrelation window (~3 short symbols)
MIN_GAP = 12000     # min samples between onsets (< frame period 15661, > sub-frame)
FRAME_PERIOD = 15661  # build_adv_replay burst_len (packed replay spacing)


def _moving_sum(x, w):
    c = np.cumsum(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
    c = np.concatenate([[0], c])
    return c[w:] - c[:-w]


def sync_metric(r, lag=STS_LAG, win=AC_WIN):
    """BOUNDED L-STF autocorrelation metric M(n)=|P|/sqrt(E1·E2) in [0,1], and the
    windowed mean power (for an absolute gap gate). Geometric-mean normalization
    keeps M<=1 even where the energy is low (the naive |P|/E blows up there)."""
    a = r[:-lag] * np.conj(r[lag:])                  # r(n)·conj(r(n+lag))
    p = np.abs(_moving_sum(a, win))                   # |windowed autocorrelation|
    e1 = _moving_sum(np.abs(r[:-lag]) ** 2, win).real
    e2 = _moving_sum(np.abs(r[lag:]) ** 2, win).real
    m = p / (np.sqrt(e1 * e2) + 1e-12)
    pw = e2 / win                                     # windowed mean power
    return m, pw


def _onsets_in_segment(seg, thr, pow_gate):
    """Rising edges of the (metric>thr AND mean-power>pow_gate) plateau."""
    m, pw = sync_metric(seg)
    L = min(len(m), len(pw))
    det = (m[:L] > thr) & (pw[:L] > pow_gate)
    rising = det & ~np.concatenate([[False], det[:-1]])
    return np.nonzero(rising)[0]


def extract_frames_sync(path, thr=0.75, gate_frac=0.3, chunk=20_000_000):
    """Detect frame onsets by L-STF autocorrelation; return FRAME_LEN frames.

    Chunked so a multi-GB capture never loads whole. Onsets are collected
    globally, de-duplicated with MIN_GAP, then frames are copied out of a memmap.
    """
    n = os.path.getsize(path) // 8
    mm = np.memmap(path, dtype=np.complex64, mode='r', shape=(n,))
    overlap = FRAME_LEN + AC_WIN + STS_LAG

    # Adaptive absolute power gate: a fraction of the median windowed power over a
    # sampled slab. For a continuous packed replay the median ~ the signal level,
    # so gate_frac*median keeps the frames and rejects only deep dips/true silence.
    probe = np.asarray(mm[: min(n, 5_000_000)])
    _, ppw = sync_metric(probe)
    pow_gate = gate_frac * float(np.median(ppw[ppw > 0]))

    cand = []
    pos = 0
    while pos < n:
        base = max(0, pos - overlap)
        seg = np.asarray(mm[base:min(pos + chunk, n)])
        for ol in _onsets_in_segment(seg, thr, pow_gate):
            g = base + int(ol)
            if g >= pos or pos == 0:          # avoid re-counting the overlap region
                cand.append(g)
        pos += chunk

    cand = sorted(set(cand))
    onsets, last = [], -MIN_GAP
    for g in cand:
        if g - last >= MIN_GAP:
            onsets.append(g)
            last = g

    frames = []
    kept = []
    for o in onsets:
        s = o - PRE_ROLL
        if s < 0 or s + FRAME_LEN > n:
            continue
        frames.append(np.asarray(mm[s:s + FRAME_LEN]).astype(np.complex64))
        kept.append(o)
    del mm
    return frames, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='complex64 capture')
    ap.add_argument('--thr', type=float, default=0.75, help='autocorr metric threshold')
    ap.add_argument('--gate-frac', type=float, default=0.3,
                    help='power gate as a fraction of the median windowed power')
    a = ap.parse_args()
    frames, onsets = extract_frames_sync(a.file, thr=a.thr, gate_frac=a.gate_frac)
    print(f"file: {a.file}")
    print(f"frames detected (L-STF autocorr, thr={a.thr}): {len(frames)}")
    if len(onsets) >= 2:
        d = np.diff(onsets)
        print(f"onset spacing: median {int(np.median(d))}  min {int(d.min())}  "
              f"max {int(d.max())} samples  (frame burst_len ~15661)")


if __name__ == '__main__':
    main()

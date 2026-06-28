#!/usr/bin/env python3
"""
exp3_extract_frames.py — Stage 1 of the fingerprinting pipeline
────────────────────────────────────────────────────────────────
Detect WiFi frame bursts in each capture and save onset-aligned FULL frames
(preamble + payload) so the later perturbation study can corrupt the payload.

Input layout:
  <root>/Device_<d>/clean_run_<r>.bin       complex64 @ 5 MHz, ~32 s each

For every (device, run):
  • memmap the file, compute a coarse power envelope in chunks (low RAM),
  • threshold to find bursts (cadence ~300 ms, duration ~2.92 ms),
  • for each burst take FRAME_LEN samples starting PRE_ROLL before onset,
  • stack frames -> frames_dev<d>.npz  with arrays:
        frames : (N, FRAME_LEN) complex64
        run    : (N,) int8        which run each frame came from
        start  : (N,) int64       absolute start sample in the source file
        peak   : (N,) float32     burst peak power (QC)

Run:
  python3 exp3_extract_frames.py                    # all devices, default paths
  python3 exp3_extract_frames.py --devices 1 2      # subset
"""
import os, sys, time, argparse
import numpy as np

FS          = 5e6
FRAME_LEN   = 15360      # samples kept per frame (covers ~2.92 ms burst + margin)
PRE_ROLL    = 256        # samples kept before detected onset (capture ramp-up)
ENV_WIN     = 200        # envelope smoothing window (40 us) — finer than a frame
CHUNK       = 8_000_000  # complex samples per chunk when scanning (≈64 MB)
EXPECT_SPACING = int(FS * 0.300)   # 1.5e6 samples between frames (TX cadence)
MIN_FULL_DUR = 12000     # keep only bursts this long (≈ real preamble+payload data
                         # frame ≈14600 samp); shorter bursts are ambient ACKs/noise


def detect_bursts(path, floor_pct=None, thr_mult=6.0):
    """Return list of (onset_sample, peak_power) for bursts in a complex64 file.

    floor_pct: if given, estimate the noise floor as that percentile of the
    envelope instead of the median. The median assumes bursts are a <50%-duty
    minority (median window = noise); that fails when the RX AGC pumps the silent
    gaps up near frame power (median lands on the signal). A low percentile
    (e.g. 20) recovers the true silence level in that case. Default (None) keeps
    the median behavior used for the training-data extraction.
    """
    nbytes = os.path.getsize(path)
    nsamp  = nbytes // 8                      # complex64 = 8 bytes
    mm = np.memmap(path, dtype=np.complex64, mode='r', shape=(nsamp,))

    # ── Pass 1: build a coarse, downsampled power envelope (chunked) ──
    env_parts = []
    for off in range(0, nsamp, CHUNK):
        seg = mm[off:off + CHUNK]
        m = (len(seg) // ENV_WIN) * ENV_WIN
        if m == 0:
            continue
        p = (seg[:m].real.astype(np.float32) ** 2 +
             seg[:m].imag.astype(np.float32) ** 2)
        env_parts.append(p.reshape(-1, ENV_WIN).mean(1))
    env = np.concatenate(env_parts)

    # Threshold relative to the NOISE FLOOR (median): bursts are a low-duty,
    # high-power minority, so the median window is pure noise. The old
    # env.mean()*3 was inflated by burst energy and landed near each burst's
    # peak, fragmenting frames below MIN_FULL_DUR (-> zero frames detected).
    floor  = np.median(env) if floor_pct is None else np.percentile(env, floor_pct)
    thresh = floor * thr_mult        # lower thr_mult (e.g. 2-3) for low-SNR captures
                                     # whose frames fragment under the default *6
    active = env > thresh

    # ── Pass 2: group contiguous active windows into bursts ──
    # Keep only full-length bursts (real preamble+payload data frames); short
    # bursts are ambient WiFi management traffic (ACKs/beacons) or noise.
    bursts = []
    i, na = 0, len(active)
    while i < na:
        if active[i]:
            j = i
            while j < na and active[j]:
                j += 1
            onset = i * ENV_WIN
            dur   = (j - i) * ENV_WIN
            peak  = float(env[i:j].max())
            if dur >= MIN_FULL_DUR:
                bursts.append((onset, peak, dur))
            i = j
        else:
            i += 1
    del mm
    return bursts, nsamp, floor, thresh


def extract_frames_for_file(path, floor_pct=None, thr_mult=6.0):
    """Detect bursts and copy FRAME_LEN-sample windows out of the file."""
    bursts, nsamp, floor, thresh = detect_bursts(path, floor_pct=floor_pct, thr_mult=thr_mult)
    mm = np.memmap(path, dtype=np.complex64, mode='r', shape=(nsamp,))
    frames, starts, peaks, durs = [], [], [], []
    for onset, peak, dur in bursts:
        s = onset - PRE_ROLL
        if s < 0:
            continue
        e = s + FRAME_LEN
        if e > nsamp:
            continue
        frames.append(np.array(mm[s:e], dtype=np.complex64))  # copy out of mmap
        starts.append(s)
        peaks.append(peak)
        durs.append(dur)
    del mm
    return frames, starts, peaks, durs, floor, thresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/media/nghoselab/T9/Data/session12/train')
    ap.add_argument('--out',  default='/media/nghoselab/T9/Data/session12/processed/frames')
    ap.add_argument('--devices', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--runs',    type=int, nargs='+', default=[1, 2, 3])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"FRAME_LEN={FRAME_LEN} ({FRAME_LEN/FS*1e3:.2f} ms)  PRE_ROLL={PRE_ROLL}")
    print(f"out -> {args.out}\n")

    grand = 0
    for d in args.devices:
        dframes, drun, dstart, dpeak, ddur = [], [], [], [], []
        for r in args.runs:
            path = os.path.join(args.root, f"device_{d}", f"clean_run_{r}.bin")
            if not os.path.exists(path):
                print(f"  [dev{d} run{r}] MISSING {path}")
                continue
            t0 = time.time()
            frames, starts, peaks, durs, floor, thresh = extract_frames_for_file(path)
            dframes.extend(frames)
            drun.extend([r] * len(frames))
            dstart.extend(starts)
            dpeak.extend(peaks)
            ddur.extend(durs)
            print(f"  [dev{d} run{r}] {len(frames):4d} full frames  "
                  f"floor={floor:.2e} thr={thresh:.2e}  ({time.time()-t0:.1f}s)")
        if not dframes:
            print(f"  device {d}: NO frames extracted, skipping")
            continue
        F = np.stack(dframes).astype(np.complex64)
        out = os.path.join(args.out, f"frames_dev{d}.npz")
        np.savez(out,
                 frames=F,
                 run=np.array(drun, dtype=np.int8),
                 start=np.array(dstart, dtype=np.int64),
                 peak=np.array(dpeak, dtype=np.float32),
                 dur=np.array(ddur, dtype=np.int32),
                 fs=FS, frame_len=FRAME_LEN, pre_roll=PRE_ROLL, device=d)
        grand += len(dframes)
        print(f"  device {d}: saved {F.shape} -> {out}\n")

    print(f"DONE. {grand} frames total across devices {args.devices}.")


if __name__ == '__main__':
    main()

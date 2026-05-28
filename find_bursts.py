#!/usr/bin/env python3
"""Find bursts in raw_iq.bin and test whether they form a periodic train
matching the TX frame cadence (300 ms = 1.5e6 samples at 5 MHz).

Decisive question:
  - Regular ~1.5M-sample-spaced burst train present -> YOUR frames ARE
    arriving. Problem is decode/preamble (burst turn-on), not link/power.
  - Only sparse/irregular bursts (~16) -> your frames are NOT getting out
    in burst mode. Problem is TX-side burst keying on the N210.
"""
import numpy as np
import sys

FS = 5e6
INTERVAL_S = 0.300
EXPECT_SPACING = int(FS * INTERVAL_S)   # 1,500,000 samples

path = sys.argv[1] if len(sys.argv) > 1 else './capture/raw_iq.bin'
x = np.fromfile(path, dtype=np.complex64)
P = np.abs(x) ** 2

# Smooth power into a coarse envelope so we detect bursts, not single samples.
win = 200  # 40 us at 5 MHz — finer than a frame, coarser than a symbol
n = len(P) // win
env = P[:n * win].reshape(n, win).mean(1)

floor = np.median(env)
thresh = max(floor * 6, env.mean() * 3)   # clearly-above-floor bursts
active = env > thresh

# Group contiguous active windows into bursts.
bursts = []
i = 0
while i < len(active):
    if active[i]:
        j = i
        while j < len(active) and active[j]:
            j += 1
        start = i * win
        dur = (j - i) * win
        pk = env[i:j].max()
        bursts.append((start, dur, pk))
        i = j
    else:
        i += 1

print(f"file={path}")
print(f"samples={len(x)}  floor(med pow)={floor:.6f}  thresh={thresh:.6f}")
print(f"bursts found: {len(bursts)}")
if not bursts:
    print("NO bursts above floor -> frames are not arriving with power.")
    sys.exit(0)

starts = np.array([b[0] for b in bursts])
durs   = np.array([b[1] for b in bursts])
peaks  = np.array([b[2] for b in bursts])
print(f"burst duration (samples): min={durs.min()} median={int(np.median(durs))} max={durs.max()}")
print(f"   (median ~ {np.median(durs)/FS*1e3:.2f} ms)")
print(f"burst peak power: min={peaks.min():.4f} median={np.median(peaks):.4f} max={peaks.max():.4f}")

if len(starts) >= 2:
    gaps = np.diff(starts)
    print(f"\ninter-burst gaps (samples): {len(gaps)} gaps")
    print(f"   min={gaps.min()} median={int(np.median(gaps))} max={gaps.max()}")
    print(f"   median gap ~ {np.median(gaps)/FS*1e3:.1f} ms   (YOUR cadence = 300 ms)")
    # How many gaps are within +/-10% of the expected 1.5M-sample spacing?
    near = np.abs(gaps - EXPECT_SPACING) < 0.10 * EXPECT_SPACING
    print(f"   gaps within +/-10%% of 300 ms cadence: {near.sum()} / {len(gaps)}")
    if near.sum() >= 0.5 * len(gaps) and near.sum() >= 5:
        print("\n=> PERIODIC TRAIN AT YOUR CADENCE. Your frames ARE arriving.")
        print("   Problem is decode/preamble (burst turn-on), NOT link budget.")
    else:
        print("\n=> No periodic 300 ms train. These look like ambient/irregular.")
        print("   Your frames are likely NOT getting out in burst mode.")

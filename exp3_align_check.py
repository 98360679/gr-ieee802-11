#!/usr/bin/env python3
"""Characterize the transmitted gapped TX template (ch0 frame + ch1 delta)."""
import numpy as np

C64 = np.complex64
GAP_DIR = "/media/nghoselab/T9/Data/session13/ota_dev6/dac_safe_gapped"
FRAME_LEN = 15661
PERIOD = 265661   # 15661 + 250000 gap

frame = np.fromfile(f"{GAP_DIR}/adv_frame.bin", dtype=C64)
pert0 = np.fromfile(f"{GAP_DIR}/adv_perturbation_psr_0.bin", dtype=C64)
print(f"len frame {len(frame)}  pert {len(pert0)}  period {PERIOD} "
      f"-> {len(frame)/PERIOD:.3f} frames")

nf = len(frame) // PERIOD
F = frame[:nf*PERIOD].reshape(nf, PERIOD)
P = pert0[:nf*PERIOD].reshape(nf, PERIOD)

# active (non-gap) region
Fa = F[:, :FRAME_LEN]
Pa = P[:, :FRAME_LEN]

# are the 91 ch0 frames identical? (frozen replay vs distinct frames)
d01 = np.max(np.abs(Fa[0] - Fa[1])) if nf > 1 else 0.0
print(f"\nframes: {nf}   |frame0-frame1|max = {d01:.4g}  "
      f"({'IDENTICAL (frozen replay)' if d01 < 1e-6 else 'DISTINCT frames'})")

# where does delta live (within the 15661 active samples)?
penv = np.abs(Pa[0])
nz = np.where(penv > 1e-4 * penv.max())[0]
print(f"delta nonzero span (frame0): samples [{nz.min()}..{nz.max()}] "
      f"of {FRAME_LEN}  (len {nz.max()-nz.min()+1})")

# realized PSR of psr_0 file: delta power vs frame power in the data region
ds, de = nz.min(), nz.max()+1
pf = np.mean(np.abs(Fa[0, ds:de])**2)
pp = np.mean(np.abs(Pa[0, ds:de])**2)
print(f"data-region frame power {pf:.4g}  delta power {pp:.4g}  "
      f"PSR = {10*np.log10(pp/pf):+.2f} dB")

# gap really zero?
print(f"gap region |frame| max {np.max(np.abs(F[0, FRAME_LEN:])):.4g}  "
      f"|delta| max {np.max(np.abs(P[0, FRAME_LEN:])):.4g}")

# peak amplitudes (DAC headroom check)
print(f"peak |frame| {np.max(np.abs(frame)):.3f}  peak |delta_psr0| "
      f"{np.max(np.abs(pert0)):.3f}")

np.save("/tmp/claude-1001/-home-nghoselab/7e6fdecd-9fba-4a89-857f-b1c39ddcc651/scratchpad/frame_tmpl.npy", Fa[0])
np.save("/tmp/claude-1001/-home-nghoselab/7e6fdecd-9fba-4a89-857f-b1c39ddcc651/scratchpad/delta_tmpl.npy", Pa[0])
print("\nsaved frame_tmpl.npy / delta_tmpl.npy (active 15661-sample templates, frame0)")

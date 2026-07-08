#!/usr/bin/env python3
"""
exp3_psr_sweep.py — build a DAC-safe PSR sweep of adversary-TX perturbation files
──────────────────────────────────────────────────────────────────────────────────
Takes the build_adv_replay outputs for one run — adv_frame.bin (ch0, the legit
device frames) and adv_perturbation.bin (ch1, baked at a known reference PSR) —
and produces, for each target PSR, a scaled ch1 perturbation.

All outputs (the frame + every pert) are then divided by ONE common factor so the
largest peak across them is <= --headroom (default 0.95). That keeps every sample
inside the USRP fc32 DAC range [-1,1] (nothing clips), preserves the PSR ratios,
and keeps the legit ch0 level constant across the whole sweep.

Transmit with wifi_adversary_tx (2-ch MIMO): frame_file = <out>/adv_frame.bin
(fixed for all runs), pert_file = <out>/adv_perturbation_psr_<lvl>.bin (swap per
run), epsilon = 1.0, USRP gain fixed across the sweep.

Run:
  python3 exp3_psr_sweep.py \
      --frame .../ota_dev6/adv_frame.bin \
      --pert  .../ota_dev6/adv_perturbation.bin \
      --out   .../ota_dev6/dac_safe
"""
import os
import argparse
import numpy as np

C64 = np.complex64
FS = 5_000_000          # sample rate (Hz)
FRAME_PERIOD = 15661    # build_adv_replay burst_len (one frame, packed)


def _tag(psr):
    p = int(round(psr))
    return f"m{abs(p)}" if p < 0 else (f"p{p}" if p > 0 else "0")


def insert_gaps(x, frame_period, gap):
    """Insert `gap` zero samples after every `frame_period`-sample frame, so a
    packed (continuous) replay becomes gapped — the power-based frame extractor
    can then segment it (frames separated by silence, like the clean captures).
    Applied identically to ch0 and ch1 so they stay sample-aligned."""
    if gap <= 0:
        return x
    nf = len(x) // frame_period
    out = np.zeros(nf * (frame_period + gap), dtype=C64)
    for i in range(nf):
        src = x[i * frame_period:(i + 1) * frame_period]
        dst = i * (frame_period + gap)
        out[dst:dst + frame_period] = src
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frame', required=True, help='adv_frame.bin (ch0 legit)')
    p.add_argument('--pert', required=True,
                   help='adv_perturbation.bin (ch1) baked at --base-psr')
    p.add_argument('--base-psr', type=float, default=-20.0,
                   help='PSR dB already baked into --pert (default -20)')
    p.add_argument('--psr', type=float, nargs='+',
                   default=[-30, -25, -20, -15, -10, -5, 0],
                   help='target PSRs in dB')
    p.add_argument('--headroom', type=float, default=0.95,
                   help='max |amp| after the common DAC-safe scale (default 0.95)')
    p.add_argument('--gap-ms', type=float, default=0.0,
                   help='zero gap (ms) inserted after each frame so the recapture '
                        'is gapped and the power extractor can segment it (0=packed)')
    p.add_argument('--frame-period', type=int, default=FRAME_PERIOD,
                   help='frame burst_len in samples (default 15661)')
    p.add_argument('--out', required=True, help='output directory')
    a = p.parse_args()
    gap = int(round(a.gap_ms * 1e-3 * FS))

    os.makedirs(a.out, exist_ok=True)
    frame = np.fromfile(a.frame, dtype=C64)
    base = np.fromfile(a.pert, dtype=C64)
    if len(frame) != len(base):
        raise SystemExit(f"length mismatch: frame {len(frame)} vs pert {len(base)} "
                         "(frame/pert must be the aligned build_adv_replay pair)")

    # MEASURE the realized base PSR instead of trusting --base-psr: the pert is
    # crafted on the (low-amplitude) RX frame but injected into the (higher-amplitude)
    # TX frame_run, so its true level vs the frame is NOT --base-psr. Measure
    # ||pert|| / ||frame|| over the payload region (where pert is nonzero) and scale
    # from that so realized PSR == label.
    pmask = np.abs(base) > 1e-6
    if pmask.sum() == 0:
        raise SystemExit("perturbation is all-zero — nothing to scale")
    meas_base = 20 * np.log10(
        np.sqrt(np.sum(np.abs(base[pmask]) ** 2)) /
        (np.sqrt(np.sum(np.abs(frame[pmask]) ** 2)) + 1e-30))
    print(f"measured base PSR of --pert vs frame (payload region): {meas_base:+.1f} dB "
          f"(--base-psr hint was {a.base_psr:+.1f}; using MEASURED)\n")

    # ch1 scaled to each target PSR, relative to the MEASURED base PSR
    scaled = {psr: (base * 10 ** ((psr - meas_base) / 20)).astype(C64)
              for psr in a.psr}

    # one common factor so the loudest peak across frame + all perts == headroom
    gmax = max(float(np.max(np.abs(frame))),
               max(float(np.max(np.abs(s))) for s in scaled.values()))
    k = a.headroom / gmax
    print(f"global peak (ch0/ch1) {gmax:.3f} -> common DAC-safe scale {k:.4f} "
          f"(target peak {a.headroom})\n")

    fr_out = insert_gaps((frame * k).astype(C64), a.frame_period, gap)
    fr_out.tofile(os.path.join(a.out, 'adv_frame.bin'))
    if gap:
        nf = len(frame) // a.frame_period
        print(f"gap: {gap} samples ({a.gap_ms:.0f} ms) after each of {nf} frames "
              f"-> period {a.frame_period + gap} samp ({(a.frame_period+gap)/FS*1e3:.1f} ms)\n")
    print(f"{'PSR':>5} {'ch1 peak':>9}  file")
    for psr in a.psr:
        out = insert_gaps((scaled[psr] * k).astype(C64), a.frame_period, gap)
        fn = f"adv_perturbation_psr_{_tag(psr)}.bin"
        out.tofile(os.path.join(a.out, fn))
        print(f"{psr:>5.0f} {float(np.max(np.abs(out))):>9.3f}  {fn}")
    print(f"\nframe peak {float(np.max(np.abs(frame * k))):.3f}   ch0 file: "
          f"adv_frame.bin   ->  {a.out}/")
    print("transmit: frame_file=adv_frame.bin (fixed), swap pert_file per run, "
          "epsilon=1.0, USRP gain fixed across the sweep.")


if __name__ == '__main__':
    main()

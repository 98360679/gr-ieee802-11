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


def _tag(psr):
    return f"m{abs(int(psr))}" if psr < 0 else "0"


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
    p.add_argument('--out', required=True, help='output directory')
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    frame = np.fromfile(a.frame, dtype=C64)
    base = np.fromfile(a.pert, dtype=C64)
    if len(frame) != len(base):
        raise SystemExit(f"length mismatch: frame {len(frame)} vs pert {len(base)} "
                         "(frame/pert must be the aligned build_adv_replay pair)")

    # ch1 scaled to each target PSR, relative to the PSR baked into --pert
    scaled = {psr: (base * 10 ** ((psr - a.base_psr) / 20)).astype(C64)
              for psr in a.psr}

    # one common factor so the loudest peak across frame + all perts == headroom
    gmax = max(float(np.max(np.abs(frame))),
               max(float(np.max(np.abs(s))) for s in scaled.values()))
    k = a.headroom / gmax
    print(f"global peak (ch0/ch1) {gmax:.3f} -> common DAC-safe scale {k:.4f} "
          f"(target peak {a.headroom})\n")

    (frame * k).astype(C64).tofile(os.path.join(a.out, 'adv_frame.bin'))
    print(f"{'PSR':>5} {'ch1 peak':>9}  file")
    for psr in a.psr:
        out = (scaled[psr] * k).astype(C64)
        fn = f"adv_perturbation_psr_{_tag(psr)}.bin"
        out.tofile(os.path.join(a.out, fn))
        print(f"{psr:>5.0f} {float(np.max(np.abs(out))):>9.3f}  {fn}")
    print(f"\nframe peak {float(np.max(np.abs(frame * k))):.3f}   ch0 file: "
          f"adv_frame.bin   ->  {a.out}/")
    print("transmit: frame_file=adv_frame.bin (fixed), swap pert_file per run, "
          "epsilon=1.0, USRP gain fixed across the sweep.")


if __name__ == '__main__':
    main()

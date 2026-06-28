#!/usr/bin/env python3
"""
exp3_assemble_pert.py — build a new ch1 (adv_perturbation) for an EXISTING adv_frame
────────────────────────────────────────────────────────────────────────────────────
ch0 (adv_frame.bin = the legit device_6 frames) is identical regardless of which
attack the perturbation encodes, so to make a new attack bundle we only need a new
ch1. This reads the existing adv_index.csv (per-frame data-region layout) + a
per-frame pert-dir (<id>.bin) and places each delta into its frame's data region,
producing an adv_perturbation.bin sample-aligned to the existing adv_frame.bin.
No TX recording / T91 needed (reuses what build_adv_replay already emitted).
"""
import os, csv, argparse, numpy as np
C64 = np.complex64


def load(p):
    return np.load(p).astype(C64) if p.endswith('.npy') else np.fromfile(p, dtype=C64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adv-frame', required=True, help='existing ch0 adv_frame.bin (for length)')
    ap.add_argument('--adv-index', required=True, help='existing adv_index.csv')
    ap.add_argument('--pert-dir', required=True)
    ap.add_argument('--pert-glob', default='{id}.bin')
    ap.add_argument('--out-pert', required=True)
    a = ap.parse_args()

    N = len(np.fromfile(a.adv_frame, dtype=C64))
    out = np.zeros(N, dtype=C64)
    rows = list(csv.DictReader(open(a.adv_index)))
    placed = miss = 0
    for r in rows:
        fid = int(r['frame_id']); off = int(r['pert_offset']); reglen = int(r['pert_region_len'])
        fp = os.path.join(a.pert_dir, a.pert_glob.format(id=fid))
        if not os.path.exists(fp):
            miss += 1; continue
        d = load(fp); use = min(len(d), reglen)
        out[off:off + use] = d[:use]; placed += 1
    out.tofile(a.out_pert)
    print(f"placed {placed} deltas ({miss} missing) -> {a.out_pert}  ({N} samples, "
          f"matches adv_frame)")


if __name__ == '__main__':
    main()

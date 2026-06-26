#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_adv_replay.py  --  Step 5 of the OTA adversarial pipeline.

Takes the TX recording (frame.bin + frame_index.csv produced by wifi_tx's
frame_recorder), the list of frame IDs the RX actually decoded, and the RX's
per-frame perturbation(s), and builds the two sample-aligned files that
wifi_adversary_tx replays on its 2-channel MIMO USRP:

    out-frame  -> channel 0 (the selected TX frames, concatenated)
    out-pert   -> channel 1 (perturbation, padded so each segment lines up
                  sample-for-sample with its frame on channel 0)

Both outputs are complex64 (GNU Radio fc32 / gr_complex), identical length, and
aligned by construction (built together in one loop). Only the frames whose IDs
the RX decoded are emitted, in the order the IDs are listed.

Perturbation placement: for each selected frame, the perturbation is written into
that frame's DATA region (the actual modulated samples, [data_offset, data_offset+
data_len)); the front/tail zero-pad of the burst stays zero. Use --align burst to
instead spread the perturbation across the whole burst.

Perturbation sources (pick one; omit both to emit an all-zero ch1 = baseline replay):
    --pert-dir DIR    one file per decoded id (complex64 raw .bin or .npy),
                      filename from --pert-glob (default "{id}.bin")
    --pert-file FILE  a single (universal) perturbation broadcast to every frame
"""
import argparse
import os
import sys
import re
import numpy as np

C64 = np.complex64


def log(msg):
    print(msg, file=sys.stderr)


def read_index(path):
    """Return dict: frame_id -> (burst_offset, burst_len, data_offset, data_len)."""
    idx = {}
    with open(path) as f:
        header = f.readline().strip().split(',')
        expected = ['frame_id', 'burst_offset', 'burst_len', 'data_offset', 'data_len']
        if header != expected:
            log(f"WARNING: index header {header} != expected {expected}; parsing positionally")
        for line in f:
            line = line.strip()
            if not line:
                continue
            fid, boff, blen, doff, dlen = (int(x) for x in line.split(','))
            idx[fid] = (boff, blen, doff, dlen)
    return idx


def read_ids(path):
    """Read decoded frame IDs (ints), preserving order, skipping a header/non-ints."""
    ids = []
    with open(path) as f:
        for line in f:
            for tok in re.split(r'[,\s]+', line.strip()):
                if re.fullmatch(r'\d+', tok):
                    ids.append(int(tok))
    return ids


def load_pert(path):
    if path.endswith('.npy'):
        return np.load(path).astype(C64)
    return np.fromfile(path, dtype=C64)


def main():
    ap = argparse.ArgumentParser(description="Build aligned frame/perturbation replay files (pipeline step 5).")
    ap.add_argument('--frame-bin', default='/dev/shm/frame.bin', help='TX recording (complex64) from frame_recorder')
    ap.add_argument('--index', default='/dev/shm/frame_index.csv', help='frame_index.csv from frame_recorder')
    ap.add_argument('--ids', required=True, help='file of decoded frame IDs from the RX side (one or more ints)')
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--pert-dir', help='directory with one perturbation file per decoded id')
    src.add_argument('--pert-file', help='single perturbation broadcast to every selected frame')
    ap.add_argument('--pert-glob', default='{id}.bin', help='filename pattern in --pert-dir ({id} placeholder)')
    ap.add_argument('--align', choices=['data', 'burst'], default='data',
                    help='place perturbation in the inner data region (default) or across the whole burst')
    ap.add_argument('--anchor', choices=['start', 'center'], default='start',
                    help='where to place a perturbation shorter than its target region')
    ap.add_argument('--out-frame', default='/dev/shm/adv_frame.bin')
    ap.add_argument('--out-pert', default='/dev/shm/adv_perturbation.bin')
    ap.add_argument('--out-index', default='/dev/shm/adv_index.csv')
    args = ap.parse_args()

    if os.path.abspath(args.frame_bin) == os.path.abspath(args.out_frame):
        ap.error("--out-frame must differ from --frame-bin (refusing to overwrite the recording)")

    index = read_index(args.index)
    ids = read_ids(args.ids)
    if not ids:
        ap.error(f"no frame IDs read from {args.ids}")

    frame = np.fromfile(args.frame_bin, dtype=C64)
    log(f"loaded {len(frame)} samples from {args.frame_bin}; index has {len(index)} frames; "
        f"{len(ids)} decoded IDs requested")

    universal = None
    if args.pert_file:
        universal = load_pert(args.pert_file)
        log(f"universal perturbation: {len(universal)} samples from {args.pert_file}")

    out_frame = []
    out_pert = []
    rows = []
    out_off = 0
    n_missing_idx = n_missing_pert = n_pad = n_trunc = 0

    for fid in ids:
        if fid not in index:
            log(f"WARNING: decoded id {fid} not in index; skipping")
            n_missing_idx += 1
            continue
        boff, blen, doff, dlen = index[fid]
        if boff + blen > len(frame):
            log(f"WARNING: id {fid} burst [{boff},{boff+blen}) exceeds frame.bin ({len(frame)}); skipping")
            n_missing_idx += 1
            continue

        burst = frame[boff:boff + blen].copy()

        # target region within the burst for the perturbation
        if args.align == 'data':
            reg_start = doff - boff
            reg_len = dlen
        else:
            reg_start = 0
            reg_len = blen

        # fetch this frame's perturbation
        if universal is not None:
            pert = universal
        elif args.pert_dir:
            fpath = os.path.join(args.pert_dir, args.pert_glob.format(id=fid))
            if not os.path.exists(fpath):
                log(f"WARNING: no perturbation file for id {fid} ({fpath}); using zeros")
                pert = np.zeros(0, dtype=C64)
                n_missing_pert += 1
            else:
                pert = load_pert(fpath)
        else:
            pert = np.zeros(0, dtype=C64)  # baseline: no perturbation

        pert_burst = np.zeros(blen, dtype=C64)
        use = min(len(pert), reg_len)
        if len(pert) > reg_len:
            n_trunc += 1
        elif 0 < len(pert) < reg_len:
            n_pad += 1
        if use > 0:
            if args.anchor == 'center':
                lead = (reg_len - use) // 2
            else:
                lead = 0
            pert_burst[reg_start + lead: reg_start + lead + use] = pert[:use]

        out_frame.append(burst)
        out_pert.append(pert_burst)
        rows.append((fid, out_off, blen, out_off + reg_start, reg_len, use))
        out_off += blen

    if not out_frame:
        log("ERROR: no frames selected; nothing written")
        return 2

    of = np.concatenate(out_frame)
    op = np.concatenate(out_pert)
    assert len(of) == len(op), "internal: frame/pert length mismatch"
    of.tofile(args.out_frame)
    op.tofile(args.out_pert)

    with open(args.out_index, 'w') as f:
        f.write('frame_id,out_offset,burst_len,pert_offset,pert_region_len,pert_samples_used\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')

    log(f"wrote {len(rows)} frames, {len(of)} samples each to:")
    log(f"  ch0 frame : {args.out_frame}")
    log(f"  ch1 pert  : {args.out_pert}")
    log(f"  index     : {args.out_index}")
    if n_missing_idx or n_missing_pert or n_pad or n_trunc:
        log(f"notes: {n_missing_idx} ids skipped (not in index), {n_missing_pert} missing pert files, "
            f"{n_pad} perts zero-padded, {n_trunc} perts truncated to region")
    log("")
    log("Point wifi_adversary_tx at these files:")
    log(f"    frame_file = '{args.out_frame}'")
    log(f"    pert_file  = '{args.out_pert}'")
    return 0


if __name__ == '__main__':
    sys.exit(main())

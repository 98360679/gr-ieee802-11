#!/usr/bin/env python3
"""
exp3_closed_loop_reconstruct.py — inject δ into the PAYLOAD only of decoded frames,
from the pristine TX recording (frame_recorder output). Two modes:

  single : device_6 replays header + (payload+δ) + tail  → one transmit file.
           δ goes into the clean pre-RF payload, header/tail untouched, so the frame
           still syncs + its FRID id decodes, and device_6 stamps its fingerprint ONCE
           (no re-stamp of an already-fingerprinted capture).

  ch1    : build the 2-channel padded perturbation frame. ch1 = zeros in header + tail,
           δ only in the payload region — so when ch0 (device_6 legit, = the frame_run
           bursts) and ch1 superimpose OTA, header/tail stay pure device_6 and δ (adversary
           fingerprint) lands only on the payload. Emits aligned ch0 + ch1 files.

Payload region per frame (boundary B, id-preserving):
     payload = [ data_offset + SPLIT : data_offset + data_len - TAIL ]
where SPLIT (default 1360) skips preamble+L-SIG+SERVICE+MAC+FRID-id, TAIL leaves the
trailing FCS/tail symbols clean. Geometry comes from frame_index_run_*.csv
(frame_id, burst_offset, burst_len, data_offset, data_len).

  python3 exp3_closed_loop_reconstruct.py --mode single \
      --frame-run  .../device_6/frame_run_1.bin \
      --frame-index .../device_6/frame_index_run_1.csv \
      --pert-dir   .../delta_dev6_to_dev4_psr-15 \
      --decoded-ids .../rx_frames_run_1.jsonl \
      --split 1360 --tail 0 --out .../closed_loop_tx.bin
"""
import os, csv, json, re, argparse, numpy as np
C64 = np.complex64


def load_index(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        rows[int(r['frame_id'])] = (int(r['burst_offset']), int(r['burst_len']),
                                    int(r['data_offset']), int(r['data_len']))
    return rows


def load_ids(path):
    txt = open(path).read()
    if path.endswith('.jsonl'):
        ids = []
        for l in txt.splitlines():
            try:
                r = json.loads(l)
                if r.get('frame_id') is not None:
                    ids.append(int(r['frame_id']))
            except Exception:
                pass
        return ids
    if 'frame_id' in txt.splitlines()[0]:
        return [int(r['frame_id']) for r in csv.DictReader(open(path))]
    return [int(t) for t in re.findall(r'\d+', txt)]


def payload_bounds(burst_off, data_off, data_len, split, tail):
    """absolute sample bounds of the payload region within frame_run."""
    lo = data_off + split
    hi = data_off + data_len - tail
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['single', 'ch1'], required=True)
    ap.add_argument('--frame-run', required=True)
    ap.add_argument('--frame-index', required=True)
    ap.add_argument('--pert-dir', required=True)
    ap.add_argument('--pert-glob', default='{id}.bin')
    ap.add_argument('--decoded-ids', default=None,
                    help='ids to attack (jsonl/csv/txt); default = every id in the index')
    ap.add_argument('--split', type=int, default=1360, help='header length into the frame (boundary B)')
    ap.add_argument('--tail', type=int, default=0, help='samples to leave clean at the frame end')
    ap.add_argument('--gap', type=int, default=0, help='zero samples between output bursts')
    ap.add_argument('--out', required=True, help='single: one .bin; ch1: dir for ch0/ch1 .bin')
    a = ap.parse_args()

    fr = np.fromfile(a.frame_run, dtype=C64)
    idx = load_index(a.frame_index)
    ids = load_ids(a.decoded_ids) if a.decoded_ids else sorted(idx)
    ids = [i for i in ids if i in idx]
    print(f"{len(fr)} frame_run samples | {len(idx)} indexed frames | {len(ids)} target ids | "
          f"boundary: data_offset+{a.split} .. data_offset+data_len-{a.tail}")

    gap = np.zeros(a.gap, C64)
    ch0_out, ch1_out, single_out = [], [], []
    used = miss = 0
    for fid in ids:
        boff, blen, doff, dlen = idx[fid]
        burst = fr[boff:boff + blen].copy()
        if len(burst) < blen:
            miss += 1; continue
        lo, hi = payload_bounds(boff, doff, dlen, a.split, a.tail)
        plen = hi - lo
        rlo, rhi = lo - boff, hi - boff            # payload bounds relative to the burst
        dpath = os.path.join(a.pert_dir, a.pert_glob.format(id=fid))
        d = np.fromfile(dpath, dtype=C64) if os.path.exists(dpath) else None
        # δ is a whole-active-region file (index 0 = frame onset); its PAYLOAD portion
        # is d[split:] — that is what goes into the payload region [data_offset+split:].
        use = min(plen, max(0, len(d) - a.split)) if d is not None else 0
        dpay = d[a.split:a.split + use] if use else None

        if a.mode == 'single':
            if use:
                burst[rlo:rlo + use] += dpay; used += 1
            single_out += [burst, gap]
        else:  # ch1: ch0 = pristine burst; ch1 = zeros except payload
            ch1 = np.zeros(blen, C64)
            if use:
                ch1[rlo:rlo + use] = dpay; used += 1
            ch0_out += [burst, gap]; ch1_out += [ch1, gap]

    if a.mode == 'single':
        np.concatenate(single_out).astype(C64).tofile(a.out) if single_out else None
        print(f"single-channel: {len(ids)-miss} frames ({used} with δ, {miss} missing) -> {a.out}")
    else:
        os.makedirs(a.out, exist_ok=True)
        np.concatenate(ch0_out).astype(C64).tofile(os.path.join(a.out, 'ch0_legit.bin'))
        np.concatenate(ch1_out).astype(C64).tofile(os.path.join(a.out, 'ch1_pert_padded.bin'))
        print(f"2-channel: {len(ids)-miss} frames ({used} with δ) -> {a.out}/ch0_legit.bin + ch1_pert_padded.bin")
        print("  header+tail of ch1 are zero → OTA superposition keeps them pure device_6; δ only on payload")


if __name__ == '__main__':
    main()

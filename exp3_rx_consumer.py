#!/usr/bin/env python3
"""
exp3_rx_consumer.py — read the RX decoded-frame log (rx_frames.jsonl)
─────────────────────────────────────────────────────────────────────

Step 2 of the closed-loop adversarial pipeline. The RX flowgraph
(gr-ieee802-11-maint-3.10/examples/wifi_rx.grc, block `frame_logger`) appends one
JSON record per CRC-valid frame that carries our FRID magic:

    {"seq": 1, "t": 1750000000.0, "frame_id": 7, "len": 524, "payload_hex": "..."}

  * frame_id    : uint32 the TX stamped into the MSDU (b'FRID'+uint32_be+'x'*filler)
  * payload_hex : the full decoded blob = 24-byte MAC header + MSDU (FCS stripped)

This module is the entry point for the perturbation stage: it loads that log,
recovers the decoded MSDU per frame (the application bytes the RX actually got),
and emits a clean, transferable bundle keyed by frame_id. The downstream crafter
(exp3_make_perturbation_ber.py) turns each frame into a per-frame perturbation;
the frame_ids in the bundle are what the TX retransmits in the attack pass
(step 4).

Usage:
  python3 exp3_rx_consumer.py --log /tmp/rx_frames.jsonl               # summary
  python3 exp3_rx_consumer.py --log /tmp/rx_frames.jsonl --export out  # write bundle
"""

import os
import sys
import json
import argparse

# 802.11 data-frame MAC header, stripped to reach the MSDU. Must match the
# frame_logger block in wifi_rx.grc and the TX in exp3_wifi_tx.py.
MAC_HDR_LEN = 24
MAGIC = b'FRID'


class RxFrame:
    """One decoded frame recovered from the RX log."""

    __slots__ = ('frame_id', 'seq', 't', 'blob', 'msdu')

    def __init__(self, frame_id, seq, t, blob):
        self.frame_id = frame_id      # int  — uint32 stamped by the TX
        self.seq = seq                # int  — RX-side arrival counter
        self.t = t                    # float — RX wall-clock timestamp
        self.blob = blob              # bytes — full decoded frame (MAC hdr + MSDU)
        self.msdu = blob[MAC_HDR_LEN:]  # bytes — application payload only

    @property
    def has_magic(self):
        return self.msdu[:4] == MAGIC

    def __repr__(self):
        return (f"RxFrame(frame_id={self.frame_id}, seq={self.seq}, "
                f"len={len(self.blob)}, msdu={len(self.msdu)}B)")


def load_rx_frames(path, dedup=True):
    """Parse rx_frames.jsonl into RxFrame records.

    Malformed/blank lines are skipped (counted in stats, not raised). With
    dedup=True, frames sharing a frame_id collapse to the LAST decode seen in
    the file (the freshest channel realization); duplicates are reported by
    load_rx_frames_with_stats. Records are returned sorted by frame_id.
    """
    return load_rx_frames_with_stats(path, dedup=dedup)[0]


def load_rx_frames_with_stats(path, dedup=True):
    """Like load_rx_frames but also returns a stats dict.

    stats = {n_lines, n_bad, n_records, n_unique, n_dups, id_min, id_max}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"RX log not found: {path}")

    records = []
    n_lines = n_bad = 0
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                d = json.loads(line)
                frame_id = int(d['frame_id'])
                blob = bytes.fromhex(d['payload_hex'])
                rec = RxFrame(frame_id=frame_id,
                              seq=int(d.get('seq', -1)),
                              t=float(d.get('t', 0.0)),
                              blob=blob)
            except (ValueError, KeyError, TypeError):
                n_bad += 1
                continue
            records.append(rec)

    n_records = len(records)
    n_dups = 0
    if dedup:
        by_id = {}                      # last write wins (freshest decode)
        for rec in records:
            if rec.frame_id in by_id:
                n_dups += 1
            by_id[rec.frame_id] = rec
        records = list(by_id.values())

    records.sort(key=lambda r: r.frame_id)
    ids = [r.frame_id for r in records]
    stats = {
        'n_lines': n_lines,
        'n_bad': n_bad,
        'n_records': n_records,
        'n_unique': len(records),
        'n_dups': n_dups,
        'id_min': min(ids) if ids else None,
        'id_max': max(ids) if ids else None,
    }
    return records, stats


def export_bundle(records, out_dir):
    """Write the transferable bundle consumed by the perturbation/TX stages.

      <out_dir>/frames.json    frame_id -> {seq, t, msdu_hex, blob_hex, len}
      <out_dir>/frame_ids.txt  one frame_id per line (the attack-pass set)

    Returns the two paths written.
    """
    os.makedirs(out_dir, exist_ok=True)
    frames = {
        str(r.frame_id): {
            'seq': r.seq,
            't': r.t,
            'len': len(r.blob),
            'msdu_hex': r.msdu.hex(),
            'blob_hex': r.blob.hex(),
        }
        for r in records
    }
    frames_path = os.path.join(out_dir, 'frames.json')
    ids_path = os.path.join(out_dir, 'frame_ids.txt')
    with open(frames_path, 'w') as f:
        json.dump(frames, f, indent=2, sort_keys=True)
    with open(ids_path, 'w') as f:
        f.write('\n'.join(str(r.frame_id) for r in records))
        if records:
            f.write('\n')
    return frames_path, ids_path


def print_summary(records, stats, n_show=10):
    print("=" * 60)
    print("RX decoded-frame log summary")
    print("=" * 60)
    print(f"  lines read      : {stats['n_lines']}")
    print(f"  malformed lines : {stats['n_bad']}")
    print(f"  records parsed  : {stats['n_records']}")
    print(f"  unique frame_ids: {stats['n_unique']}")
    print(f"  duplicate ids   : {stats['n_dups']} (collapsed to freshest decode)")
    if stats['id_min'] is not None:
        print(f"  frame_id range  : {stats['id_min']} .. {stats['id_max']}")
        ids = set(r.frame_id for r in records)
        full = set(range(stats['id_min'], stats['id_max'] + 1))
        missing = sorted(full - ids)
        if missing:
            head = ', '.join(map(str, missing[:20]))
            more = '' if len(missing) <= 20 else f" (+{len(missing) - 20} more)"
            print(f"  missing ids     : {len(missing)} [{head}{more}]")
        else:
            print(f"  missing ids     : none (contiguous)")
    n_bad_magic = sum(1 for r in records if not r.has_magic)
    if n_bad_magic:
        print(f"  WARNING: {n_bad_magic} record(s) lack the FRID magic in the MSDU")
    print("-" * 60)
    for r in records[:n_show]:
        ascii_id = r.msdu[4:8].hex() if r.has_magic else '????'
        print(f"  id={r.frame_id:<10} seq={r.seq:<5} len={len(r.blob):<4} "
              f"msdu[:8]={r.msdu[:8].hex()}")
    if len(records) > n_show:
        print(f"  ... ({len(records) - n_show} more)")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(
        description="Read the RX decoded-frame log (rx_frames.jsonl)")
    p.add_argument('--log', default='/tmp/rx_frames.jsonl',
                   help='path to the RX frame_logger JSONL (default: /tmp/rx_frames.jsonl)')
    p.add_argument('--export', metavar='OUT_DIR', default=None,
                   help='write frames.json + frame_ids.txt bundle to OUT_DIR')
    p.add_argument('--no-dedup', action='store_true',
                   help='keep every record (do not collapse repeated frame_ids)')
    p.add_argument('--show', type=int, default=10,
                   help='number of records to list in the summary')
    a = p.parse_args()

    try:
        records, stats = load_rx_frames_with_stats(a.log, dedup=not a.no_dedup)
    except FileNotFoundError as e:
        sys.exit(str(e))

    print_summary(records, stats, n_show=a.show)

    if not records:
        sys.exit("No usable frames in the log — nothing to export.")

    if a.export:
        frames_path, ids_path = export_bundle(records, a.export)
        print(f"\nWrote bundle:")
        print(f"  {frames_path}   ({len(records)} frames)")
        print(f"  {ids_path}      (frame_id set for the attack pass)")


if __name__ == '__main__':
    main()

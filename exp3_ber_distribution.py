#!/usr/bin/env python3
"""
exp3_ber_distribution.py — per-frame bit-error distribution from a saved
exp3_ber_eval.py decode dump (--save-dump). Answers "is the BER spread evenly
or concentrated in a few bad frames?" without re-decoding the capture.

Leads with the ERROR-FREE FRAME RATE as the primary metric: on this rig errors
are bimodal (a frame is either clean or heavily corrupted), so the fraction of
clean frames is more interpretable than raw BER.

Run:
  python3 exp3_ber_distribution.py /tmp/clean_dump.txt
"""
import sys
import numpy as np
from exp3_ber_eval import parse_frames, _is_ours, _POPCOUNT, MAC_HEADER, PDU_LENGTH

dump = open(sys.argv[1]).read()
frames = [m for m in parse_frames(dump) if _is_ours(m)]
P = np.array([list(m[MAC_HEADER:MAC_HEADER + PDU_LENGTH]) for m in frames
             if len(m) >= MAC_HEADER + PDU_LENGTH], dtype=np.uint8)
F, L = P.shape
ref = np.array([np.bincount(P[:, j]).argmax() for j in range(L)], dtype=np.uint8)
diff = np.bitwise_xor(P, ref)
biterr = _POPCOUNT[diff].sum(axis=1)            # bit errors per frame
total = int(biterr.sum())
bits = F * L * 8

n_clean = int((biterr == 0).sum())
ef_rate = n_clean / F if F else 0.0
print("=" * 56)
print(f">>> ERROR-FREE FRAME RATE : {ef_rate*100:.1f}%  ({n_clean}/{F})   <<<  PRIMARY")
print(f"    frame error rate      : {(1-ef_rate)*100:.1f}%  ({F-n_clean}/{F})")
print("=" * 56)
print(f"aligned frames        : {F}   (bits compared {bits:,})")
print(f"total bit errors      : {total:,}   BER {total/bits:.3e}  (secondary)")
print(f"frames with errors    : {(biterr>0).sum()}")
print()
print("per-frame bit errors (only frames with >0):")
nz = biterr[biterr > 0]
if len(nz):
    print(f"  min/median/mean/max : {nz.min()} / {int(np.median(nz))} / {nz.mean():.1f} / {nz.max()}")
    # concentration: how much of total error do the worst K frames hold?
    s = np.sort(biterr)[::-1]
    for k in (1, 3, 5, 10):
        if k <= F:
            print(f"  top {k:2d} worst frames   : {100*s[:k].sum()/total:.1f}% of all bit errors")
    # histogram buckets
    print("  distribution:")
    edges = [1, 10, 50, 200, 1000, 4001]
    lo = 1
    for hi in edges[1:]:
        c = int(((biterr >= lo) & (biterr < hi)).sum())
        print(f"    {lo:4d}–{hi-1:4d} errs : {c} frames")
        lo = hi

#!/usr/bin/env python3
"""Decode ONE OTA capture (slice) and print its BER vs a saved δ-off reference.
Run per file in a subprocess so a gr-ieee80211 segfault on a noisy capture doesn't cascade."""
import os, sys, tempfile, pickle, argparse, numpy as np
import exp3_ber_eval as B
from exp3_cvnn_ber import ber_fer, bits
C64 = np.complex64
MAC = getattr(B, 'MAC_HEADER', 24); PDU = getattr(B, 'PDU_LENGTH', 500)


def decode_file(path, nsamp):
    x = np.fromfile(path, dtype=C64, count=nsamp)
    tmp = tempfile.mktemp('.bin'); x.tofile(tmp)
    dump = tempfile.mktemp('.txt'); sys.stdout.flush(); saved = os.dup(1)
    with open(dump, 'w') as fo:
        os.dup2(fo.fileno(), 1)
        try:
            B.ber_rx(tmp).run()
        finally:
            sys.stdout.flush(); os.dup2(saved, 1); os.close(saved)
    dec = list(B.parse_frames(open(dump).read())); os.remove(tmp); os.remove(dump)
    return [bytes(d[MAC:MAC + PDU]) for d in dec if B._is_ours(d)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--ref', help='pickle of reference MSDUs; if --save-ref, write instead')
    ap.add_argument('--save-ref', action='store_true')
    ap.add_argument('--nsamp', type=int, default=15_000_000)
    a = ap.parse_args()
    ms = decode_file(a.file, a.nsamp)
    if a.save_ref:
        pickle.dump(ms, open(a.ref, 'wb'))
        print(f"REF {len(ms)}")
        return
    ref = [bits(x) for x in pickle.load(open(a.ref, 'rb'))]
    b, _ = ber_fer(ref, ms) if ms else (0.5, 1.0)
    print(f"NF {len(ms)} BER {b:.3e}")


if __name__ == '__main__':
    main()

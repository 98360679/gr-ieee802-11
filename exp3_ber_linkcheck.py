#!/usr/bin/env python3
"""
exp3_ber_linkcheck.py — link-stealth (FER + BER) across an OTA attack PSR sweep
────────────────────────────────────────────────────────────────────────────────
Computes per-PSR FER (CRC-fail rate) and BER (over the known 0x78 MSDU fill)
straight from the recapture's ber_frames_psr_*.jsonl decode logs — no re-decode of
the IQ. Flat metrics across PSR => the perturbation power does not degrade the link
(link-stealthy at every level). See [[exp3-closed-loop-pipeline]] for the 2026-06-27
result: FER ~0.12 / BER ~4e-2 flat from PSR 0 to -30.
"""
import os, json, argparse, numpy as np

W0, W1 = 40, 510          # MSDU fill byte window (all 0x78 on this rig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/media/nghoselab/T9/Data/session13/attacked/6_27_2026/device_6')
    ap.add_argument('--psr', type=int, nargs='+', default=[0, -5, -10, -15, -20, -25, -30])
    a = ap.parse_args()
    print(f"{'PSR':>5} {'frames':>7} {'crc_ok%':>8} {'FER':>7} {'BER':>9} {'errBER':>8}")
    out = []
    for psr in a.psr:
        fn = os.path.join(a.dir, f"ber_frames_psr_{abs(psr)}.jsonl")
        if not os.path.exists(fn):
            print(f"{psr:>5}  MISSING {fn}"); continue
        n = ok = be = bt = efe = eft = 0
        for l in open(fn):
            l = l.strip()
            if not l:
                continue
            o = json.loads(l); n += 1; c = bool(o.get('crc_ok')); ok += c
            h = o.get('payload_hex')
            if not h:
                continue
            b = np.frombuffer(bytes.fromhex(h), dtype=np.uint8)
            if len(b) < W1:
                continue
            err = np.unpackbits(b[W0:W1]) != np.unpackbits(np.full(W1 - W0, 0x78, np.uint8))
            be += int(err.sum()); bt += err.size
            if not c:
                efe += int(err.sum()); eft += err.size
        fer = 1 - ok / n if n else float('nan')
        print(f"{psr:>5} {n:>7} {100*ok/n:>7.1f}% {fer:>7.3f} {be/bt:>9.2e} "
              f"{(efe/eft if eft else 0):>8.2e}")
        out.append(dict(psr=psr, frames=n, crc_ok=ok, fer=fer, ber=be/bt if bt else None))
    json.dump(out, open("ber_linkcheck.json", "w"), indent=2)
    print("\nFlat across PSR => delta power does not degrade the link (link-stealthy).")


if __name__ == '__main__':
    main()

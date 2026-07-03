#!/usr/bin/env python3
"""
exp3_cvnn_ber.py — link-stealth BER for the CVNN attack (Exp 1), per (ε, attack).

For each attack config, craft δ per frame, decode clean vs frame+δ through the validated
gr-ieee80211 chain, and compute BER = bit differences between the perturbed decode and its
matching clean decode (best Hamming match -> each perturbed frame paired to its own clean
origin, robust to drops and needs no known-TX bits / mode reference). BER table parallels
the fooling table (PGD/FGSM × targeted/untargeted × ε).

  python3 exp3_cvnn_ber.py --model fingerprint_cvnn_7_03_attack.pt \
      --capture .../7_03_2026/device_6/clean_run_1.bin --device 6 --target 4 \
      --epsilons 0.05 ... 0.26 --nframes 15
"""
import os, sys, tempfile, argparse, numpy as np, torch
import exp3_ber_eval as B
from exp3_cvnn_attack import load_cvnn, craft, predict, C64
from exp3_fp_model import NUM_CLASSES
from exp3_extract_frames import extract_frames_for_file

MAC_HEADER = getattr(B, 'MAC_HEADER', 24)
PDU_LENGTH = getattr(B, 'PDU_LENGTH', 500)


def decode_capture(frames, gap=4000):
    """Decode a list of complex frames -> list of MSDU bytes (ours-filtered)."""
    parts = []
    for f in frames:
        parts.append(np.asarray(f, C64)); parts.append(np.zeros(gap, C64))
    tmp = tempfile.mktemp(suffix='.bin'); np.concatenate(parts).astype(C64).tofile(tmp)
    dump = tempfile.mktemp(suffix='.txt')
    sys.stdout.flush(); saved = os.dup(1)
    with open(dump, 'w') as fo:
        os.dup2(fo.fileno(), 1)
        try:
            B.ber_rx(tmp).run()
        finally:
            sys.stdout.flush(); os.dup2(saved, 1); os.close(saved)
    dec = list(B.parse_frames(open(dump).read()))
    os.remove(tmp); os.remove(dump)
    return [bytes(d[MAC_HEADER:MAC_HEADER + PDU_LENGTH]) for d in dec if B._is_ours(d)]


def bits(msdu):
    return np.unpackbits(np.frombuffer(msdu, np.uint8))


def ber_fer(clean_bits, pert_msdus):
    """Return (BER, FER). BER = bit-errors/bits (each perturbed frame best-matched to a clean
    frame); 0 observed errors -> rule-of-3 upper bound 2.996/N; no decode at all -> 0.5 (total
    link failure, random). FER = fraction of the cleanly-decodable frames that δ breaks (fail
    to decode OR decode with >=1 bit error = CRC fail)."""
    tot_err = tot_bits = n_correct = 0
    for pm in pert_msdus:
        pb = bits(pm); bestd, bestL = 10 ** 12, 0
        for cb in clean_bits:
            L = min(len(cb), len(pb)); d = int((cb[:L] != pb[:L]).sum())
            if d < bestd:
                bestd, bestL = d, L
        tot_err += bestd; tot_bits += bestL
        if bestd == 0:
            n_correct += 1
    ber = 0.5 if not tot_bits else (tot_err if tot_err > 0 else 2.996) / tot_bits
    fer = 1.0 - n_correct / max(1, len(clean_bits))
    return ber, min(1.0, max(0.0, fer))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--capture', required=True)
    ap.add_argument('--device', type=int, default=6); ap.add_argument('--target', type=int, default=4)
    ap.add_argument('--nframes', type=int, default=15); ap.add_argument('--steps', type=int, default=80)
    ap.add_argument('--step-frac', type=float, default=0.1)
    ap.add_argument('--epsilons', type=float, nargs='+',
                    default=[round(0.05 + 0.01 * i, 2) for i in range(22)])
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = load_cvnn(a.model, dev); true, tgt = a.device - 1, a.target - 1

    allf = extract_frames_for_file(a.capture, floor_pct=20.0, thr_mult=2.0)[0]
    frames = [np.asarray(f, C64) for f in allf
              if predict(m, torch.tensor(np.asarray(f, C64), device=dev)) == true][:a.nframes]
    clean_msdus = decode_capture(frames)
    clean_bits = [bits(x) for x in clean_msdus]
    print(f"{a.model}  BER (link stealth), device_{a.device}->device_{a.target}\n"
          f"  {len(frames)} frames, {len(clean_msdus)} clean-decoded reference MSDUs\n")

    tag = 'PGD→d%d PGDoff FGSM→d%d FGSMoff' % (a.target, a.target)
    rows = []
    for eps in a.epsilons:
        r = {}
        for method in ('pgd', 'fgsm'):
            for mode, target in (('t', tgt), ('u', None)):
                pert = [fn + craft(m, fn, true, target, eps, method, a.steps, a.step_frac, dev).cpu().numpy()
                        for fn in frames]
                r[(method, mode)] = ber_fer(clean_bits, decode_capture(pert))
        rows.append((eps, r))
        print(f"[eps {eps:.3f} done]")
    order = [('pgd', 't'), ('pgd', 'u'), ('fgsm', 't'), ('fgsm', 'u')]
    print(f"\n=== BER ===\n{'eps':>6} | {tag}")
    for eps, r in rows:
        print(f"{eps:>6.3f} | " + " ".join(f"{r[k][0]:>8.2e}" for k in order))
    print(f"\n=== FER ===\n{'eps':>6} | {tag}")
    for eps, r in rows:
        print(f"{eps:>6.3f} | " + " ".join(f"{r[k][1]:>8.3f}" for k in order))


if __name__ == '__main__':
    main()

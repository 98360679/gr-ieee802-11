#!/usr/bin/env python3
"""
exp3_rx_eval.py — fingerprint receiver + attack evaluation
──────────────────────────────────────────────────────────

The matching receiver for the per-frame attack. It detects frames in a received
IQ stream, windows + classifies them with the SAME model and preprocessing used
for training/attack (exp3_fp_model — single source of truth), and reports the
fingerprint result. Point it at a clean capture and at an attacked capture to
measure how much the adversary degraded device identification.

Three sources:
  --source usrp   capture live from a USRP, then analyze   (needs hardware)
  --source file   analyze an existing complex64 capture     (e.g. an Rx tap)
  --source selftest  no hardware: build frame.bin + eps*perturbation.bin
                     (optionally + AWGN), then run the full Rx pipeline. Verifies
                     detection + windowing + the attack end-to-end on this box.

Examples:
  # No hardware — confirm the Rx pipeline + attack agree with the crafter:
  .venv/bin/python exp3_rx_eval.py --source selftest --true-device 1 --epsilon 1.0

  # Offline, on a captured Rx tap:
  .venv/bin/python exp3_rx_eval.py --source file --file /tmp/rx_clean.bin --true-device 1
  .venv/bin/python exp3_rx_eval.py --source file --file /tmp/rx_attacked.bin --true-device 1

  # Live capture for 20 s then analyze:
  .venv/bin/python exp3_rx_eval.py --source usrp --addr 192.168.10.6 \
        --freq 2.45e9 --gain 0.6 --duration 20 --true-device 1
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from exp3_fp_model import (FingerprintCNN, iq_to_input, NUM_CLASSES,
                           DEVICE_NAMES, WIN, ACTIVE, FS)
from exp3_train_fingerprint import LOCAL_PT, SAVE_PT

_EPS = 1e-12


# ── frame detection ───────────────────────────────────────────────────
def detect_bursts(x, smooth=256, min_active=11000, guard=0.30):
    """Find frame bursts in a continuous IQ stream.

    Returns list of (start, end) sample indices. A burst is a contiguous span
    whose smoothed power is clearly above the noise floor and lasts at least
    min_active samples (~ one frame's active region). Mirrors the envelope logic
    used to segment the training frames (find_bursts.py).
    """
    P = np.abs(x) ** 2
    n = len(P) // smooth
    if n == 0:
        return []
    env = P[:n * smooth].reshape(n, smooth).mean(1)
    floor = np.median(env)
    thr = max(floor * 6, env.mean() * 3)
    active = env > thr
    bursts = []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            s, e = i * smooth, j * smooth
            if e - s >= min_active:
                bursts.append((s, e))
            i = j
        else:
            i += 1
    # refine each start to the sample-level rising edge within the coarse span
    refined = []
    for s, e in bursts:
        seg = np.abs(x[s:e])
        pk = seg.max() + _EPS
        rise = np.argmax(seg > guard * pk)
        refined.append((s + int(rise), e))
    return refined


def windows_from_burst(x, start, align_search=64, model=None):
    """Window ACTIVE samples from `start` into 14×1024, optionally searching a
    small alignment offset for the most confident classification."""
    offsets = [0] if (align_search == 0 or model is None) else \
        range(-align_search, align_search + 1, 16)
    best = None
    for off in offsets:
        s = start + off
        if s < 0 or s + ACTIVE > len(x):
            continue
        w = np.stack([x[s + k:s + k + WIN] for k in range(0, ACTIVE - WIN + 1, WIN)])
        xin = iq_to_input(w)
        if model is None:
            return xin, s
        with torch.no_grad():
            conf = F.softmax(model(torch.from_numpy(xin)), 1).numpy()
        score = conf.max(1).mean()                # how confident overall
        if best is None or score > best[0]:
            best = (score, xin, s)
    return (None, None) if best is None else (best[1], best[2])


# ── classify + report ─────────────────────────────────────────────────
def classify_stream(x, model, align_search=64):
    bursts = detect_bursts(x)
    results = []
    for s, _e in bursts:
        xin, _ = windows_from_burst(x, s, align_search, model)
        if xin is None:
            continue
        with torch.no_grad():
            prob = F.softmax(model(torch.from_numpy(xin)), 1).numpy()
        votes = prob.argmax(1)
        dev = int(np.bincount(votes, minlength=NUM_CLASSES).argmax())
        results.append((dev, float(prob[:, dev].mean()), len(votes)))
    return results


def report(results, true_device, label):
    print(f"\n──── {label} ────")
    if not results:
        print("  no frames detected.")
        return
    devs = np.array([r[0] for r in results])
    print(f"  frames detected: {len(devs)}")
    hist = np.bincount(devs, minlength=NUM_CLASSES)
    for d in range(NUM_CLASSES):
        if hist[d]:
            print(f"    -> {DEVICE_NAMES[d]}: {hist[d]}  ({hist[d]/len(devs)*100:.0f}%)")
    if true_device is not None:
        tl = true_device - 1
        acc = (devs == tl).mean()
        print(f"  TRUE device = {DEVICE_NAMES[tl]}")
        print(f"  fingerprint accuracy : {acc*100:.1f}%")
        print(f"  attack fooling rate  : {(1-acc)*100:.1f}%")
        if (devs != tl).any():
            mis = np.bincount(devs[devs != tl], minlength=NUM_CLASSES)
            tgt = int(mis.argmax())
            print(f"  most common misID    : {DEVICE_NAMES[tgt]} "
                  f"({mis[tgt]}/{(devs!=tl).sum()} of errors)")


# ── sources ───────────────────────────────────────────────────────────
def src_file(path):
    return np.fromfile(path, dtype=np.complex64)


def src_selftest(args):
    """Build a received stream digitally from frame.bin + eps*perturbation.bin."""
    frame = np.fromfile(args.frame, dtype=np.complex64)
    pert = np.fromfile(args.pert, dtype=np.complex64)
    if len(frame) != len(pert):
        sys.exit("selftest: frame.bin and perturbation.bin differ in length.")
    rx = frame + args.epsilon * pert
    rx = np.tile(rx, args.reps)                          # a few looped frames
    if args.snr_db is not None:
        p = np.mean(np.abs(rx[np.abs(rx) > 0]) ** 2)
        n = np.sqrt(p / (2 * 10 ** (args.snr_db / 10)))
        rx = rx + n * (np.random.randn(*rx.shape) + 1j * np.random.randn(*rx.shape))
    return rx.astype(np.complex64)


def src_usrp(args):
    """Capture `duration` s from a USRP to a temp file, return the samples."""
    from gnuradio import gr, blocks, uhd
    import time
    out = args.capture or "/tmp/exp3_rx_capture.bin"

    class rx(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self, "exp3 rx", catch_exceptions=True)
            self.u = uhd.usrp_source(args.dev,
                                     uhd.stream_args(cpu_format="fc32", channels=[0]))
            self.u.set_samp_rate(FS)
            self.u.set_center_freq(uhd.tune_request(args.freq), 0)
            self.u.set_antenna(args.antenna, 0)
            self.u.set_normalized_gain(args.gain, 0)
            self.sink = blocks.file_sink(gr.sizeof_gr_complex, out, False)
            self.connect((self.u, 0), (self.sink, 0))

    tb = rx(); tb.start(); time.sleep(args.duration); tb.stop(); tb.wait()
    print(f"  captured -> {out}")
    return np.fromfile(out, dtype=np.complex64)


def main():
    p = argparse.ArgumentParser(description="Exp3 fingerprint Rx + attack eval")
    p.add_argument('--source', choices=['usrp', 'file', 'selftest'], default='selftest')
    p.add_argument('--file', help='complex64 capture (source=file)')
    p.add_argument('--true-device', type=int, default=None, help='device 1..6 being transmitted')
    p.add_argument('--model', default=LOCAL_PT if os.path.exists(LOCAL_PT) else SAVE_PT)
    p.add_argument('--align-search', type=int, default=64, help='+/- sample offset search (0=off)')
    p.add_argument('--label', default='capture')
    # selftest
    p.add_argument('--frame', default='/dev/shm/frame.bin')
    p.add_argument('--pert', default='/dev/shm/perturbation.bin')
    p.add_argument('--epsilon', type=float, default=1.0)
    p.add_argument('--reps', type=int, default=8)
    p.add_argument('--snr-db', type=float, default=None, help='add AWGN at this SNR (selftest)')
    # usrp
    p.add_argument('--addr', default='192.168.10.6')
    p.add_argument('--dev', default=None, help='full UHD args; overrides --addr')
    p.add_argument('--freq', type=float, default=2.45e9)
    p.add_argument('--antenna', default='J1')
    p.add_argument('--gain', type=float, default=0.6)
    p.add_argument('--duration', type=float, default=20.0)
    p.add_argument('--capture', default=None, help='where to save the USRP capture')
    args = p.parse_args()
    if args.dev is None:
        args.dev = f"addr={args.addr}"

    model = FingerprintCNN(NUM_CLASSES)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()

    if args.source == 'file':
        if not args.file:
            p.error("--source file requires --file")
        x = src_file(args.file); label = f"FILE {os.path.basename(args.file)}"
    elif args.source == 'usrp':
        x = src_usrp(args); label = f"USRP {args.dev} @ {args.freq/1e9:.3f}GHz"
    else:
        x = src_selftest(args)
        label = f"SELFTEST eps={args.epsilon}" + (
            f" snr={args.snr_db}dB" if args.snr_db is not None else "")

    print(f"loaded {len(x)} samples ({len(x)/FS*1e3:.1f} ms)  model={os.path.basename(args.model)}")
    results = classify_stream(x, model, args.align_search)
    report(results, args.true_device, label)


if __name__ == '__main__':
    main()

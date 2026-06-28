#!/usr/bin/env python3
"""
exp3_rebaseline.py — re-validate the fingerprint model on NEW-RX clean captures
───────────────────────────────────────────────────────────────────────────────
After swapping the receiver USRP, the model trained through the OLD receiver may
not transfer (receiver-induced domain shift: the new RX's CFO / I-Q imbalance /
filter shape distort every TX differently than the chain the model learned on).

Capture a CLEAN run per victim device through the NEW RX, point this at them, and
it reports per-device + overall window/frame accuracy and the DROP vs the model's
own committed baseline. It uses the EXACT Stage-1 extraction (exp3_extract_frames)
and the model's own non-overlapping windowing, so the numbers are directly
comparable to training. No attack here — clean captures only.

Decision rule:
  * accuracy holds near baseline  -> model transferred, keep using it.
  * accuracy dropped              -> recollect victim training data through the
                                     new RX and retrain (old frames_dev*.npz are
                                     old-RX data and can't be reused).

Input (either form, mirrors exp3_eval_capture):
  --root DIR                  scan DIR/Device_<d>/*.bin  (or device_<d>)
  --capture FILE --device D   one capture with explicit label (repeatable)

Run:
  python3 exp3_rebaseline.py --root /media/nghoselab/T9/Data/session13/newrx_clean
  python3 exp3_rebaseline.py --capture dev5_newrx.bin --device 5
"""
import os
import sys
import glob
import re
import json
import argparse
import numpy as np
import torch

from exp3_fp_model import frame_to_windows, iq_to_input, WIN
from exp3_extract_frames import extract_frames_for_file
from exp3_make_perturbation import load_fp_model, default_model


def collect(args):
    """Return [(path, device_id), ...] from --root and/or --capture/--device."""
    items = []
    if args.root:
        for dd in sorted(glob.glob(os.path.join(args.root, '*'))):
            if not os.path.isdir(dd):
                continue
            m = re.search(r'device[_\-]?(\d+)', os.path.basename(dd), re.I)
            if not m:
                continue
            d = int(m.group(1))
            for b in sorted(glob.glob(os.path.join(dd, '*.bin'))):
                items.append((b, d))
    caps = args.capture or []
    devs = args.device or []
    if len(caps) != len(devs):
        sys.exit("--capture and --device must be paired 1:1")
    for c, d in zip(caps, devs):
        items.append((c, int(d)))
    if not items:
        sys.exit("No captures found. Use --root or --capture/--device.")
    return items


def eval_file(model, n_classes, path, floor_pct=None, thr_mult=6.0):
    """Per-window predictions (flat) and per-frame voted class for one capture.

    floor_pct: robust noise-floor percentile for the burst detector (use ~20 for
    attack recaptures where the RX AGC pumps the silent gaps up; None = median,
    as for the clean training-style captures)."""
    frames = extract_frames_for_file(path, floor_pct=floor_pct, thr_mult=thr_mult)[0]
    win_preds, frame_preds = [], []
    for fr in frames:
        w = frame_to_windows(fr, hop=WIN)              # non-overlapping (val rule)
        x = torch.from_numpy(iq_to_input(w))
        with torch.no_grad():
            pred = model(x).argmax(1).numpy()
        win_preds.extend(pred.tolist())
        frame_preds.append(int(np.bincount(pred, minlength=n_classes).argmax()))
    return win_preds, frame_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None)
    ap.add_argument('--capture', action='append',
                    help='capture .bin (repeatable, pair with --device)')
    ap.add_argument('--device', action='append',
                    help='device label for the matching --capture')
    ap.add_argument('--model', default=default_model(),
                    help='fingerprint .pt to evaluate (default: 4-device model)')
    ap.add_argument('--out', default='rebaseline_newrx.json')
    a = ap.parse_args()

    model, n_classes, name_to_idx, idx_to_name = load_fp_model(a.model)
    cfgp = os.path.splitext(a.model)[0] + '.json'
    base = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
    base_w, base_f = base.get('val_window_acc'), base.get('val_frame_acc')

    print(f"Model {os.path.basename(a.model)} ({n_classes} classes)")
    if base_f is not None:
        print(f"Committed baseline: window {base_w:.4f}  frame {base_f:.4f}")
    print()

    items = collect(a)
    C = np.zeros((n_classes, n_classes), int)          # frame confusion
    per_dev = {}
    tw_ok = tw = tf_ok = tf = 0
    for path, dev in items:
        name = f'device_{dev}'
        if name not in name_to_idx:
            print(f"  SKIP {os.path.basename(path)}: {name} not a class in this "
                  f"model ({sorted(name_to_idx)})")
            continue
        true = name_to_idx[name]
        wp, fp = eval_file(model, n_classes, path)
        if not fp:
            print(f"  WARN {os.path.basename(path)}: no frames detected")
            continue
        wp, fp = np.array(wp), np.array(fp)
        w_ok, f_ok = int((wp == true).sum()), int((fp == true).sum())
        d = per_dev.setdefault(name, dict(w_ok=0, w=0, f_ok=0, f=0, files=0))
        d['w_ok'] += w_ok; d['w'] += len(wp); d['f_ok'] += f_ok
        d['f'] += len(fp); d['files'] += 1
        for p in fp:
            C[true, p] += 1
        tw_ok += w_ok; tw += len(wp); tf_ok += f_ok; tf += len(fp)
        print(f"  {name}: {len(fp):>4} frames  win {w_ok/len(wp):.3f}  "
              f"frame {f_ok/len(fp):.3f}  ({os.path.basename(path)})")

    if tf == 0:
        sys.exit("\nNo usable frames — nothing to score.")

    print("\n=== per-device ===")
    dev_rows = {}
    for name in sorted(per_dev):
        d = per_dev[name]
        wa, fa = d['w_ok'] / d['w'], d['f_ok'] / d['f']
        dev_rows[name] = dict(frames=d['f'], windows=d['w'],
                              window_acc=wa, frame_acc=fa, files=d['files'])
        print(f"  {name:>9}: frames {d['f']:>4}  window {wa:.3f}  frame {fa:.3f}")

    ov_w, ov_f = tw_ok / tw, tf_ok / tf
    print(f"\nOVERALL  window {ov_w:.4f}  frame {ov_f:.4f}")
    if base_f is not None:
        print(f"BASELINE window {base_w:.4f}  frame {base_f:.4f}")
        print(f"DROP     window {base_w - ov_w:+.4f}  frame {base_f - ov_f:+.4f}")
        verdict = ("TRANSFERRED (within ~2pts)" if base_f - ov_f <= 0.02
                   else "DEGRADED -> recollect + retrain on new-RX data")
        print(f"VERDICT  {verdict}")

    print("\n=== FRAME confusion (rows=true, cols=pred) ===")
    names = [idx_to_name[i] for i in range(n_classes)]
    print("true\\pred " + "".join(f"{n:>9}" for n in names))
    for i in range(n_classes):
        row = "".join(f"{C[i, j]:>9}" for j in range(n_classes))
        print(f"{names[i]:>9}{row}   n={C[i].sum()}")

    report = dict(
        model=os.path.basename(a.model), n_classes=n_classes,
        baseline=dict(window=base_w, frame=base_f),
        overall=dict(window=ov_w, frame=ov_f),
        drop=dict(window=None if base_w is None else base_w - ov_w,
                  frame=None if base_f is None else base_f - ov_f),
        per_device=dev_rows, frame_confusion=C.tolist(), class_order=names)
    with open(a.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == '__main__':
    main()

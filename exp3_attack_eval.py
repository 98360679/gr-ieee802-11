#!/usr/bin/env python3
"""
exp3_attack_eval.py — measure OTA fooling of the fingerprinter on an attacked recapture
────────────────────────────────────────────────────────────────────────────────────────
Step 6 (closed-loop): after the 2-channel MIMO adversary TX (wifi_adversary_tx)
replays  adv_frame.bin (legit device frames, ch0) + adv_perturbation.bin (δ, ch1)
and you re-capture the air, point this at the attacked .bin (true legit device
known) to measure the REAL over-the-air fooling against the 6-class model:

  * fooling rate = fraction of frames NOT classified as the true legit device
  * the dominant misclassification target (the digital craft pushed device_6 -> device_1)
  * optional clean-vs-attacked side-by-side if you pass --clean

Reuses the exact Stage-1 extraction + model windowing (via exp3_rebaseline), so
numbers are comparable to training and the clean baseline.

Run (after the replay + recapture exists):
  python3 exp3_attack_eval.py --attacked attacked_dev6.bin --device 6
  python3 exp3_attack_eval.py --attacked attacked_dev6.bin --device 6 \
        --clean /media/nghoselab/T9/Data/session13/train/device_6/clean_run_3.bin
"""
import os
import sys
import json
import argparse
import numpy as np

from exp3_rebaseline import eval_file
from exp3_make_perturbation import load_fp_model
from exp3_train_fingerprint import LOCAL_PT          # the canonical retrained (6-class) model


def summarize(tag, frame_preds, true_idx, idx_to_name, n_classes):
    fp = np.array(frame_preds)
    n = len(fp)
    acc = float((fp == true_idx).mean()) if n else 0.0
    fooling = 1.0 - acc
    dist = {idx_to_name.get(i, f'class{i}'): int((fp == i).sum())
            for i in range(n_classes)}
    wrong = {i: int((fp == i).sum()) for i in range(n_classes) if i != true_idx}
    tgt = max(wrong, key=wrong.get) if any(wrong.values()) else None
    print(f"  [{tag}] {n} frames   acc(as {idx_to_name[true_idx]}) {acc:.3f}   "
          f"fooling {fooling:.3f}")
    print(f"        dist: {dist}")
    if tgt is not None and wrong[tgt] > 0:
        print(f"        dominant target: {idx_to_name[tgt]} "
              f"({wrong[tgt]}/{n} = {wrong[tgt]/n:.2f})")
    return dict(frames=n, acc=acc, fooling=fooling, dist=dist,
                dominant_target=(idx_to_name.get(tgt) if tgt is not None else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attacked', required=True,
                    help='attacked recapture .bin (legit+adversary OTA)')
    ap.add_argument('--clean', default=None,
                    help='optional clean capture of the same device for comparison')
    ap.add_argument('--device', type=int, required=True,
                    help='true legit device id (e.g. 6)')
    ap.add_argument('--model', default=LOCAL_PT,
                    help='fingerprint .pt (default: canonical retrained 6-class model)')
    ap.add_argument('--out', default='attack_eval.json')
    a = ap.parse_args()

    model, n_classes, name_to_idx, idx_to_name = load_fp_model(a.model)
    name = f'device_{a.device}'
    if name not in name_to_idx:
        sys.exit(f"{name} is not a class in {os.path.basename(a.model)} "
                 f"({sorted(name_to_idx)})")
    true = name_to_idx[name]
    print(f"Model {os.path.basename(a.model)} ({n_classes} classes); "
          f"legit TX {name} -> class {true}\n")

    report = {'model': os.path.basename(a.model), 'device': name}
    if a.clean:
        _, fc = eval_file(model, n_classes, a.clean)
        report['clean'] = summarize('clean', fc, true, idx_to_name, n_classes)
    _, fa = eval_file(model, n_classes, a.attacked)
    report['attacked'] = summarize('attacked', fa, true, idx_to_name, n_classes)

    if a.clean:
        drop = report['clean']['acc'] - report['attacked']['acc']
        print(f"\n  CLEAN acc {report['clean']['acc']:.3f}  ->  "
              f"ATTACKED acc {report['attacked']['acc']:.3f}   "
              f"(accuracy drop {drop:+.3f} = OTA fooling gain)")
        report['accuracy_drop'] = drop

    json.dump(report, open(a.out, 'w'), indent=2)
    print(f"\nWrote {a.out}")


if __name__ == '__main__':
    main()

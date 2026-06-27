#!/usr/bin/env python3
"""
exp3_psr_sweep_eval.py — OTA fooling vs PSR for the device_6 attacked recaptures
────────────────────────────────────────────────────────────────────────────────
Re-eval driver for the session13 re-run: evaluates the clean device_6 baseline +
every attacked recapture in one process (model loaded once), reusing the exact
Stage-1 extraction + windowing (exp3_rebaseline.eval_file) and the summarize()
breakdown from exp3_attack_eval, so numbers match training / the prior runs.

Writes one attack_eval_psr_<lvl>.json per PSR (same schema as exp3_attack_eval)
plus a combined psr_sweep_eval.json with the fooling-vs-PSR table.
"""
import os
import json
import numpy as np

from exp3_rebaseline import eval_file
from exp3_make_perturbation import load_fp_model
from exp3_train_fingerprint import LOCAL_PT
from exp3_attack_eval import summarize

ATTACK_DIR = "/media/nghoselab/T9/Data/session13/attacked/device_6"
CLEAN = "/media/nghoselab/T9/Data/session13/train/device_6/clean_run_3.bin"
DEVICE = 6
FLOOR_PCT = 20.0

# file-name PSR tag  ->  true PSR in dB  (file '12' is actually -15 dB; user typo)
PSR_FILES = [
    ("adv_psr_0.bin",    0),
    ("adv_psr_5.bin",   -5),
    ("adv_psr_10.bin", -10),
    ("adv_psr_12.bin", -15),
    ("adv_psr_20.bin", -20),
    ("adv_psr_25.bin", -25),
    ("adv_psr_30.bin", -30),
]


def _tag(psr):
    return f"m{abs(int(psr))}" if psr < 0 else "0"


def main():
    model, n_classes, name_to_idx, idx_to_name = load_fp_model(LOCAL_PT)
    name = f"device_{DEVICE}"
    true = name_to_idx[name]
    print(f"Model {os.path.basename(LOCAL_PT)} ({n_classes} classes); "
          f"legit TX {name} -> class {true}\n")

    # clean baseline once
    print("=== clean baseline (clean_run_3) ===")
    _, fc = eval_file(model, n_classes, CLEAN, floor_pct=FLOOR_PCT)
    clean = summarize("clean", fc, true, idx_to_name, n_classes)
    clean_acc = clean["acc"]

    sweep = []
    for fn, psr in PSR_FILES:
        path = os.path.join(ATTACK_DIR, fn)
        print(f"\n=== PSR {psr:+d} dB  ({fn}) ===")
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        _, fa = eval_file(model, n_classes, path, floor_pct=FLOOR_PCT)
        rep = summarize("attacked", fa, true, idx_to_name, n_classes)
        drop = clean_acc - rep["acc"]
        print(f"        clean acc {clean_acc:.3f} -> attacked acc {rep['acc']:.3f}"
              f"   (accuracy drop {drop:+.3f} = OTA fooling gain)")

        out = {"model": os.path.basename(LOCAL_PT), "device": name,
               "psr_db": psr, "file": fn, "floor_pct": FLOOR_PCT,
               "clean": clean, "attacked": rep, "accuracy_drop": drop}
        outfn = f"attack_eval_psr_{_tag(psr)}.json"
        json.dump(out, open(outfn, "w"), indent=2)
        print(f"        wrote {outfn}")

        sweep.append({"psr_db": psr, "file": fn, "frames": rep["frames"],
                      "acc": rep["acc"], "fooling": rep["fooling"],
                      "accuracy_drop": drop,
                      "dominant_target": rep["dominant_target"],
                      "dist": rep["dist"]})

    summary = {"model": os.path.basename(LOCAL_PT), "device": name,
               "floor_pct": FLOOR_PCT, "clean_acc": clean_acc,
               "clean_frames": clean["frames"], "sweep": sweep}
    json.dump(summary, open("psr_sweep_eval.json", "w"), indent=2)

    print("\n\n================  PSR SWEEP SUMMARY  ================")
    print(f"clean device_6 acc {clean_acc:.3f} ({clean['frames']} frames)\n")
    print(f"{'PSR(dB)':>7} {'frames':>7} {'acc':>7} {'fooling':>8} "
          f"{'drop':>7}  dominant_target")
    for s in sweep:
        print(f"{s['psr_db']:>7d} {s['frames']:>7d} {s['acc']:>7.3f} "
              f"{s['fooling']:>8.3f} {s['accuracy_drop']:>+7.3f}  "
              f"{s['dominant_target']}")
    print("\nWrote psr_sweep_eval.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Eval the EOT per-frame OTA recapture sweep (device_6 -> device_4, fine-tuned model)."""
import os, json, numpy as np
from exp3_rebaseline import eval_file
from exp3_make_perturbation import load_fp_model

MODEL = "fingerprint_cnn_ft20260627.pt"
AD = "/media/nghoselab/T9/Data/session13/attacked/6_27_2026/device_6"
CLEAN = "/media/nghoselab/T9/Data/session13/train/6_27_2026/device_6/clean_run_1.bin"
FLOOR = 20.0
PSR = [("adv_psr_0.bin", 0), ("adv_psr_5.bin", -5), ("adv_psr_10.bin", -10),
       ("adv_psr_15.bin", -15), ("adv_psr_20.bin", -20), ("adv_psr_25.bin", -25),
       ("adv_psr_30.bin", -30)]


def summ(fp, true, tgt, i2n, nc):
    fp = np.array(fp); n = len(fp)
    acc = float((fp == true).mean()) if n else 0
    hit = float((fp == tgt).mean()) if n else 0
    dist = {i2n.get(i, i): int((fp == i).sum()) for i in range(nc)}
    return n, acc, 1 - acc, hit, dist


def main():
    m, nc, n2i, i2n = load_fp_model(MODEL)
    true, tgt = n2i['device_6'], n2i['device_4']
    print(f"{MODEL}  legit device_6 (class {true})  target device_4 (class {tgt})\n")
    _, fc = eval_file(m, nc, CLEAN, floor_pct=None)
    n, acc, foo, hit, dist = summ(fc, true, tgt, i2n, nc)
    print(f"clean  : {n} fr  acc {acc:.3f}  device_4 {hit:.3f}")
    rows = []
    for fn, psr in PSR:
        p = os.path.join(AD, fn)
        if not os.path.exists(p):
            print(f"  MISSING {fn}"); continue
        _, fa = eval_file(m, nc, p, floor_pct=FLOOR)
        n, acc, foo, hit, dist = summ(fa, true, tgt, i2n, nc)
        dom = max((k for k in dist if k != 'device_6'), key=lambda k: dist[k])
        print(f"PSR {psr:>4}: {n:>5} fr  acc(dev6) {acc:.3f}  fooling {foo:.3f}  "
              f"->device_4 {hit:.3f}  dom {dom}({dist[dom]})")
        rows.append(dict(psr=psr, file=fn, frames=n, acc=acc, fooling=foo,
                         device_4_hit=hit, dist=dist, dominant=dom))
    json.dump({"model": MODEL, "clean_acc": acc, "sweep": rows},
              open("eot_attack_eval.json", "w"), indent=2)
    print("\n=== SUMMARY (target = device_4) ===")
    print(f"{'PSR':>5} {'frames':>7} {'fooling':>8} {'->dev4':>8}  dominant")
    for r in rows:
        print(f"{r['psr']:>5} {r['frames']:>7} {r['fooling']:>8.3f} "
              f"{r['device_4_hit']:>8.3f}  {r['dominant']}")
    print("\nWrote eot_attack_eval.json")


if __name__ == "__main__":
    main()

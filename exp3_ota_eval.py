#!/usr/bin/env python3
"""
exp3_ota_eval.py — evaluate an OTA attack recapture PSR sweep against a fingerprint model
──────────────────────────────────────────────────────────────────────────────────────────
General eval for the closed-loop OTA recaptures (adv_psr_{0,5,10,15,20,25,30}.bin in DIR).
Reports per-PSR: frames, fooling (off the legit device), optional target-hit, and the
class distribution. Use thr_mult=2 (+ floor_pct=20) for low-SNR recaptures whose frames
fragment under the default *6 (tell-tale: rx_frames.jsonl ~1000 decoded, extractor ~0).

  python3 exp3_ota_eval.py --dir <recapture_dir> --model fingerprint_cnn_ft20260628.pt \
      --device 6 --target 4 --thr-mult 2

Finding (session13): TARGETED transfers OTA (~73% @-15 on the crisp 6-28 model); pure
untargeted and runner-up do NOT (phase-fragile / target doesn't transfer). See
[[exp3-closed-loop-pipeline]].
"""
import os, re, glob, json, argparse, numpy as np
from exp3_rebaseline import eval_file
from exp3_make_perturbation import load_fp_model


def _psr_from_name(fn):
    """adv_psr_m40 -> -40, adv_psr_p10 -> +10, adv_psr_0 -> 0, adv_psr_15 -> -15
    (bare number = negative, old convention)."""
    t = re.search(r'adv_psr_(m|p)?(\d+)', os.path.basename(fn))
    if not t:
        return None
    sign, n = t.group(1), int(t.group(2))
    return n if sign == 'p' else (-n if (sign == 'm' or n) else 0)


def discover(d):
    """all adv_psr_*.bin in DIR, sorted by PSR ascending."""
    out = [(os.path.basename(p), _psr_from_name(p)) for p in glob.glob(os.path.join(d, "adv_psr_*.bin"))]
    return sorted([(f, p) for f, p in out if p is not None], key=lambda x: x[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="recapture dir with adv_psr_*.bin")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", type=int, default=6, help="legit TX device id")
    ap.add_argument("--target", type=int, default=None, help="targeted attack: report ->device_<target> hit")
    ap.add_argument("--clean", default=None, help="optional clean capture for a baseline row")
    ap.add_argument("--thr-mult", type=float, default=2.0)
    ap.add_argument("--floor-pct", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    m, nc, n2i, i2n = load_fp_model(a.model)
    true = n2i[f"device_{a.device}"]
    tgt = n2i[f"device_{a.target}"] if a.target is not None else None
    print(f"{os.path.basename(a.model)}  legit device_{a.device}"
          + (f"  target device_{a.target}" if tgt is not None else "  (untargeted)"))

    def summ(fa):
        fa = np.array(fa); n = len(fa)
        foo = 1 - float((fa == true).mean()) if n else float("nan")
        hit = float((fa == tgt).mean()) if (n and tgt is not None) else None
        dist = {i2n.get(i, i): int((fa == i).sum()) for i in range(nc)}
        return n, foo, hit, dist

    if a.clean:
        n, foo, hit, _ = summ(eval_file(m, nc, a.clean, floor_pct=None)[1])
        print(f"clean: {n} fr  device_{a.device} acc {1-foo:.3f}")

    hdr = f"{'PSR':>5} {'frames':>7} {'fooling':>8}" + (f" {'->dev'+str(a.target):>8}" if tgt is not None else "")
    print("\n" + hdr + "  distribution")
    rows = []
    for fn, psr in discover(a.dir):
        p = os.path.join(a.dir, fn)
        n, foo, hit, dist = summ(eval_file(m, nc, p, floor_pct=a.floor_pct, thr_mult=a.thr_mult)[1])
        dd = {k: v for k, v in sorted(dist.items(), key=lambda x: -x[1]) if v}
        line = f"{psr:>5} {n:>7} {foo:>8.3f}" + (f" {hit:>8.3f}" if hit is not None else "")
        print(f"{line}  {dd}")
        rows.append(dict(psr=psr, frames=n, fooling=foo, target_hit=hit, dist=dist))

    if a.out:
        json.dump({"model": os.path.basename(a.model), "dir": a.dir,
                   "device": a.device, "target": a.target, "sweep": rows},
                  open(a.out, "w"), indent=2)
        print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()

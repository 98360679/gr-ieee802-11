#!/usr/bin/env python3
"""
exp3_eot_pertdir.py — craft per-frame EOT-robust targeted delta from a CLEAN capture
────────────────────────────────────────────────────────────────────────────────────
Frame source = a clean RX capture of the legit device (model domain, classifies
correctly), e.g. train/6_27_2026/device_6/clean_run_1.bin. For each device_6 frame
the model calls correctly, craft an alignment-robust (EOT) targeted delta toward
device_4 and write the BARE data-region delta as <out>/<i>.bin — the per-frame
perturbation files build_adv_replay consumes via --pert-dir.
"""
import os
import argparse
import numpy as np
import torch

from exp3_make_perturbation import load_fp_model, predict_frame
from exp3_make_perturbation_eot import craft_eot, hit_under
from exp3_extract_frames import extract_frames_for_file
from exp3_fp_model import PRE_ROLL, ACTIVE

C64 = np.complex64
DEF_CAP = "/media/nghoselab/T9/Data/session13/train/6_27_2026/device_6/clean_run_1.bin"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture', default=DEF_CAP)
    p.add_argument('--model', default='fingerprint_cnn_ft20260627.pt')
    p.add_argument('--device', type=int, default=6)
    p.add_argument('--target', type=int, default=4)
    p.add_argument('--untargeted', action='store_true',
                   help='pure untargeted PGD (ascend CE on true) — alignment-fragile')
    p.add_argument('--runner-up', action='store_true',
                   help='robust untargeted: per frame, target the model nearest wrong class')
    p.add_argument('--method', choices=['pgd', 'fgsm'], default='pgd',
                   help='attack: iterative EOT-PGD (default) or single-step EOT-FGSM')
    p.add_argument('--psr', type=float, default=-20.0)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--n-eot', type=int, default=12)
    p.add_argument('--shift', type=float, default=1.5)
    p.add_argument('--phase', type=float, default=90.0)
    p.add_argument('--max-frames', type=int, default=10000)
    p.add_argument('--ids-csv', default='/media/nghoselab/T9/Data/session13/ota_dev6/eot_t4_ids.txt',
                   help='TX frame ids: a frame_index CSV (frame_id column) OR a plain one-id-per-line '
                        'file. Outputs are named <frame_id>.bin for build_adv_replay --pert-glob {id}.bin')
    p.add_argument('--out', default='/media/nghoselab/T9/Data/session13/ota_dev6/eot_t4_pertdir')
    a = p.parse_args()
    import csv, re
    txt = open(a.ids_csv).read()
    if 'frame_id' in txt.splitlines()[0]:                       # CSV with header
        ids = [int(r['frame_id']) for r in csv.DictReader(open(a.ids_csv))]
    else:                                                       # plain one-int-per-line
        ids = [int(t) for t in re.findall(r'\d+', txt)]
    print(f"{len(ids)} TX frame_ids to cover (range {min(ids)}..{max(ids)})")

    model, nc, n2i, i2n = load_fp_model(a.model)
    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(DEV)
    true, tgt = n2i[f'device_{a.device}'], n2i[f'device_{a.target}']
    target_label = None if a.untargeted else tgt
    mi = 1 if (a.untargeted or a.runner_up) else 0   # [1]=off-true, [0]=target-hit
    mode = ("UNTARGETED off device_%d" % a.device if a.untargeted
            else "RUNNER-UP (per-frame nearest wrong class)" if a.runner_up
            else f"device_{a.device}->device_{a.target}")
    os.makedirs(a.out, exist_ok=True)
    print(f"{a.model}: {mode}  PSR {a.psr}  "
          f"EOT shift±{a.shift}/phase±{a.phase}  -> {a.out}")

    allf = extract_frames_for_file(a.capture)[0]
    frames = [f for f in allf if predict_frame(model, f, nc)[0] == true]   # clean device_6 only
    print(f"extracted {len(allf)} frames, {len(frames)} clean device_6, from "
          f"{os.path.basename(a.capture)}; crafting one EOT delta per TX id "
          f"(cycling frames if fewer than ids)")

    n_ok = n_done = 0
    hits = []
    for j, fid in enumerate(ids):
        if n_done >= a.max_frames:
            break
        fr = frames[j % len(frames)]           # per-id delta; frames near-identical, EOT makes each robust
        if a.runner_up:                        # auto-target this frame's nearest wrong class
            _, prob = predict_frame(model, fr, nc)
            mp = prob.mean(0).copy(); mp[true] = -1.0; tl = int(mp.argmax())
        else:
            tl = target_label
        d = craft_eot(model, fr, true, tl, a.psr, DEV,
                      a.steps, n_eot=a.n_eot, shift=a.shift, phase_deg=a.phase,
                      method=a.method)
        bare = d[PRE_ROLL:PRE_ROLL + ACTIVE].astype(C64)   # data-region delta for build_adv_replay
        bare.tofile(os.path.join(a.out, f"{fid}.bin"))     # named by frame_id
        h0 = hit_under(model, fr, d, true, tgt, nc)[mi]
        h1 = hit_under(model, fr, d, true, tgt, nc, tau=1.0)[mi]
        hp = hit_under(model, fr, d, true, tgt, nc, deg=90)[mi]
        hits.append((h0, h1, hp))
        n_ok += h0; n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done}/{len(ids)} crafted  (nominal hit {n_ok}/{n_done})")

    H = np.array(hits) if hits else np.zeros((0, 3))
    print(f"\ncrafted {n_done} per-frame EOT deltas -> {a.out}/*.bin")
    if len(H):
        lbl = f"off device_{a.device}" if (a.untargeted or a.runner_up) else f"->device_{a.target}"
        print(f"  {lbl}:  nominal {H[:,0].mean()*100:.0f}%   "
              f"@1-sample {H[:,1].mean()*100:.0f}%   @90deg {H[:,2].mean()*100:.0f}%")
    print("Feed to build_adv_replay: --pert-dir "
          f"{a.out} --pert-glob '{{id}}.bin'  (with the TX frame.bin + frame_index.csv + ids)")


if __name__ == '__main__':
    main()

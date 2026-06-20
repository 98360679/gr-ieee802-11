#!/usr/bin/env python3
"""
exp3_make_perturbation_ber.py — craft a per-frame PGD perturbation AGAINST AN
EXISTING (decodable) frame.bin, for the BER experiment.

Unlike exp3_make_perturbation.py (which picks a captured device frame from the
fingerprint dataset AND writes its own frame.bin), this version:
  * READS the already-deployed /dev/shm/frame.bin (the decodable 'EXP3'+'A'
    synthetic frame) as the signal s — does NOT overwrite it,
  * maps that frame's burst into the fingerprint model's window layout
    (FRAME_LEN/PRE_ROLL/ACTIVE), runs the SAME untargeted PGD vs the CNN,
  * writes ONLY perturbation.bin, length L, sample-for-sample aligned to
    frame.bin (delta placed at the burst's offset), at the design PSR.

The CNN was trained on real device frames, so fooling a synthetic frame is a
weaker attack — that caveat is accepted; the point here is a perturbation that
stays per-frame aligned to the decodable frame so OTA BER is measurable.

Run:
  .venv/bin/python exp3_make_perturbation_ber.py --psr -20 \
        --frame /dev/shm/frame.bin --out-pert /dev/shm/perturbation.bin
"""

import os
import argparse
import numpy as np
import torch

from exp3_fp_model import (FingerprintCNN, NUM_CLASSES, FRAME_LEN, PRE_ROLL,
                           ACTIVE, WIN, FS, _EPS)
from exp3_make_perturbation import craft, predict_frame
from exp3_train_fingerprint import LOCAL_PT, SAVE_PT


def main():
    p = argparse.ArgumentParser(description="BER-mode perturbation vs an existing frame.bin")
    p.add_argument('--frame', default='/dev/shm/frame.bin',
                   help='EXISTING decodable frame.bin (read-only, the signal s)')
    p.add_argument('--out-pert', default='/dev/shm/perturbation.bin')
    p.add_argument('--psr', type=float, default=-20.0, help='perturbation-to-signal ratio dB')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--step-frac', type=float, default=0.1)
    p.add_argument('--model', default=LOCAL_PT if os.path.exists(LOCAL_PT) else SAVE_PT)
    a = p.parse_args()

    model = FingerprintCNN(NUM_CLASSES)
    model.load_state_dict(torch.load(a.model, map_location='cpu'))
    model.eval()

    # ── load the deployed frame (the signal); keep its length L and amplitude ──
    fbin = np.fromfile(a.frame, dtype=np.complex64)
    L = len(fbin)
    nz = np.nonzero(np.abs(fbin) > 0.0)[0]
    if len(nz) == 0:
        raise SystemExit(f"{a.frame} is all zeros.")
    burst_start = int(nz[0])

    # Map the burst into the model's FRAME_LEN layout so the burst sits at
    # PRE_ROLL (exactly where the CNN expects the active region to begin).
    model_off = burst_start - PRE_ROLL
    if model_off < 0:
        raise SystemExit(f"burst_start={burst_start} < PRE_ROLL={PRE_ROLL}; "
                         f"frame has too little front pad.")
    seg = fbin[model_off: model_off + FRAME_LEN]
    if len(seg) < FRAME_LEN:                       # pad tail if frame runs short
        seg = np.concatenate([seg, np.zeros(FRAME_LEN - len(seg), np.complex64)])
    model_frame = seg.astype(np.complex64)

    # Untargeted: push the CNN away from its CURRENT prediction on this frame.
    dev_clean, _ = predict_frame(model, model_frame)
    true_label = int(dev_clean)
    print(f"frame={a.frame}  L={L}  burst_start={burst_start}  "
          f"model_off={model_off}")
    print(f"CNN clean prediction: device_{dev_clean+1} (used as untargeted label)")

    delta_full, starts = craft(model, model_frame, true_label,
                               a.psr, a.steps, a.step_frac)

    # ── place delta back into a length-L buffer, aligned to frame.bin ──
    pert = np.zeros(L, dtype=np.complex64)
    pert[model_off: model_off + FRAME_LEN] = delta_full
    pert.tofile(a.out_pert)

    # ── verify: prediction flip + achieved PSR over the active windows ──
    dev_adv, padv = predict_frame(model, model_frame + delta_full)
    flipped = (padv.argmax(1) != true_label).mean()
    act = slice(PRE_ROLL, PRE_ROLL + ACTIVE)
    sig_p = float(np.mean(np.abs(model_frame[act]) ** 2))
    prt_p = float(np.mean(np.abs(delta_full[act]) ** 2))
    achieved = 10 * np.log10((prt_p + _EPS) / (sig_p + _EPS))

    print(f"  perturbed prediction: device_{dev_adv+1}  "
          f"(windows flipped {flipped*100:.0f}%)")
    print(f"  achieved PSR = {achieved:.2f} dB (target {a.psr})")
    print(f"Wrote {a.out_pert}  (length {L}, frame.bin untouched)")
    print(f"  -> TX: loop frame.bin on ch0, perturbation.bin*epsilon on ch1")


if __name__ == '__main__':
    main()

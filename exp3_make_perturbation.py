#!/usr/bin/env python3
"""
exp3_make_perturbation.py — craft a per-frame PGD perturbation vs the fingerprint CNN
──────────────────────────────────────────────────────────────────────────────────

White-box, untargeted PGD against the model we trained (exp3_train_fingerprint.py),
so the perturbation is faithful to the preprocessing it will face at the receiver.

OTA picture (per-frame):
  * legit radio (ch0) REPLAYS one captured device frame  -> the "signal" s
  * adversary radio (ch1) transmits the perturbation      -> delta
  * at the Rx they add; the fingerprint CNN should misclassify the device.

The receiver normalizes each 1024-window to unit RMS, so absolute scale is
irrelevant — only the perturbation-to-signal RATIO (PSR) matters. We therefore
craft delta under a per-window PSR budget and bake that PSR into perturbation.bin
RELATIVE to frame.bin (both written at true relative amplitude). In the .grc,
epsilon=1.0 with equal gains/path-loss realizes the design PSR; epsilon fine-tunes.

Outputs (complex64 @ 5 MHz, both length L, sample-for-sample aligned):
  frame.bin         legit replay signal  (unit-RMS over active, padded to L)
  perturbation.bin  delta                (scaled to PSR vs frame, padded to L)

Run (after training):
  .venv/bin/python exp3_make_perturbation.py --device 1 --psr -20 \
        --out-frame /dev/shm/frame.bin --out-pert /dev/shm/perturbation.bin
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from exp3_fp_model import (FingerprintCNN, frame_to_windows, torch_iq_to_input,
                           iq_to_input, NUM_CLASSES, DEVICE_NAMES,
                           WIN, ACTIVE, PRE_ROLL, FRAME_LEN, FS)
from exp3_train_fingerprint import load_frames, LOCAL_PT, SAVE_PT

_EPS = 1e-12

# The 4-device model (device_2 dropped) is the default attack target now; fall
# back to the 5-class model, then the drive copy, if it isn't present locally.
LOCAL_4DEV = os.path.splitext(LOCAL_PT)[0] + '_4dev.pt'


def default_model():
    for p in (LOCAL_4DEV, LOCAL_PT, SAVE_PT):
        if os.path.exists(p):
            return p
    return LOCAL_PT


def load_fp_model(model_path):
    """Load a FingerprintCNN sized to the checkpoint, with the device<->class
    label map taken from the model's sidecar JSON config.

    The 4-device model remaps kept devices to contiguous class indices
    (device_1->0, device_3->1, device_4->2, device_5->3), so we must NOT assume
    the old 5-class `device_d -> d-1` rule. We read `class_labels` from
    <model>.json when present; otherwise fall back to that identity rule.

    Returns (model, n_classes, name_to_idx, idx_to_name).
    """
    state = torch.load(model_path, map_location='cpu')
    n_classes = state['head.5.weight'].shape[0]        # final Linear rows
    cfg_path = os.path.splitext(model_path)[0] + '.json'
    name_to_idx = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            name_to_idx = {k: int(v)
                           for k, v in json.load(f).get('class_labels', {}).items()}
    if not name_to_idx:                                # no/empty config -> 5-class identity
        name_to_idx = {f'device_{i}': i - 1 for i in range(1, n_classes + 1)}
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    model = FingerprintCNN(n_classes)
    model.load_state_dict(state)
    model.eval()
    return model, n_classes, name_to_idx, idx_to_name


def predict_frame(model, frame, n_classes=NUM_CLASSES):
    """Majority-vote class prediction + mean confidence for one frame.

    Returns the class INDEX (0..n_classes-1) the model voted, not a device id —
    map it back through idx_to_name for display.
    """
    w = frame_to_windows(frame, hop=WIN)
    x = torch.from_numpy(iq_to_input(w))
    dvc = next(model.parameters()).device          # follow the model (cpu or cuda)
    with torch.no_grad():
        p = F.softmax(model(x.to(dvc)), dim=1).cpu().numpy()
    votes = p.argmax(1)
    dev = np.bincount(votes, minlength=n_classes).argmax()
    return dev, p


def craft(model, frame, true_label, psr_db, steps, step_frac, target_label=None):
    """Per-frame PGD. Returns delta over the FULL frame (complex).

    target_label=None -> UNTARGETED: ascend CE(true_label), push off the true class.
    target_label=k    -> TARGETED:   descend CE(k), pull toward class k.
    Both stay inside the same per-window PSR L2 ball."""
    # Non-overlapping inference windows over the active region.
    lo = PRE_ROLL
    starts = list(range(lo, lo + ACTIVE - WIN + 1, WIN))      # 14 windows
    s = torch.from_numpy(frame.astype(np.complex64))
    sig = torch.stack([s[a:a + WIN] for a in starts])          # [W,1024] complex
    # Per-window L2 budget for delta from the PSR (power ratio).
    sig_norm = torch.sqrt((sig.real**2 + sig.imag**2).sum(1) + _EPS)   # [W]
    budget = (10.0 ** (psr_db / 20.0)) * sig_norm              # ||delta_w||_2

    if target_label is not None:
        y = torch.full((len(starts),), target_label, dtype=torch.long)
        sign = -1.0                                            # descend toward target
    else:
        y = torch.full((len(starts),), true_label, dtype=torch.long)
        sign = +1.0                                            # ascend off the true class
    d = torch.zeros(len(starts), WIN, 2)                       # [W,1024,2] I/Q
    d.normal_(0, 1e-3); d.requires_grad_(True)

    def windows_input(delta):
        dc = torch.complex(delta[..., 0], delta[..., 1])       # [W,1024]
        return torch_iq_to_input(sig + dc)                     # [W,2,1024]

    for _ in range(steps):
        if d.grad is not None:
            d.grad.zero_()
        logits = model(windows_input(d))
        loss = F.cross_entropy(logits, y)
        loss.backward()
        with torch.no_grad():
            g = d.grad
            gnorm = g.flatten(1).norm(dim=1).clamp_min(_EPS)
            d += sign * (step_frac * budget / gnorm).view(-1, 1, 1) * g   # PGD step
            # Project each window's delta back into its PSR L2 ball.
            dn = d.flatten(1).norm(dim=1)
            scale = torch.clamp(budget / dn.clamp_min(_EPS), max=1.0)
            d *= scale.view(-1, 1, 1)

    delta_full = np.zeros(FRAME_LEN, dtype=np.complex64)
    dd = d.detach().numpy()
    for i, a in enumerate(starts):
        delta_full[a:a + WIN] = dd[i, :, 0] + 1j * dd[i, :, 1]
    return delta_full, starts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=int, default=5,
                   help='legit-TX device id; must be a class in --model '
                        '(4-dev model: 1,3,4,5)')
    p.add_argument('--target', type=int, default=None,
                   help='targeted attack: push --device toward this device id '
                        '(must also be a class in --model). Omit = untargeted.')
    p.add_argument('--psr', type=float, default=-20.0, help='perturbation-to-signal ratio dB')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--step-frac', type=float, default=0.1, help='PGD step as frac of budget')
    p.add_argument('--period-ms', type=float, default=50.0, help='OTA loop length')
    p.add_argument('--model', default=default_model(),
                   help='fingerprint .pt to attack (default: 4-device model)')
    p.add_argument('--out-frame', default='/dev/shm/frame.bin')
    p.add_argument('--out-pert',  default='/dev/shm/perturbation.bin')
    p.add_argument('--eval-n', type=int, default=40, help='frames to measure fooling rate')
    a = p.parse_args()

    model, n_classes, name_to_idx, idx_to_name = load_fp_model(a.model)
    name = f'device_{a.device}'
    if name not in name_to_idx:
        raise SystemExit(
            f"{name} is not a class in {os.path.basename(a.model)} "
            f"(classes: {sorted(name_to_idx)}). Pick a kept device.")
    true_label = name_to_idx[name]
    target_label = None
    if a.target is not None:
        tname = f'device_{a.target}'
        if tname not in name_to_idx:
            raise SystemExit(
                f"target {tname} is not a class in {os.path.basename(a.model)} "
                f"(classes: {sorted(name_to_idx)}).")
        target_label = name_to_idx[tname]
    mode = (f"TARGETED -> device_{a.target} (class {target_label})"
            if target_label is not None else "UNTARGETED")
    print(f"Model {os.path.basename(a.model)} ({n_classes} classes); "
          f"legit TX {name} -> class {true_label}   [{mode}]")

    frames, run = load_frames(a.device)
    # Use held-out run-3 frames; pick the most confident correctly-classified one.
    idx = np.where(run == 3)[0]
    best = None
    for i in idx:
        dev, prob = predict_frame(model, frames[i], n_classes)
        if dev == true_label:
            conf = prob[:, true_label].mean()
            if best is None or conf > best[1]:
                best = (i, conf)
    if best is None:
        raise SystemExit(f"No correctly-classified run-3 frame for {name}.")
    ti = best[0]
    target = frames[ti]
    print(f"Target: {name} frame #{ti}  (clean conf {best[1]:.3f})")

    delta, starts = craft(model, target, true_label, a.psr, a.steps,
                          a.step_frac, target_label)

    # Verify on the crafted frame.
    pert_frame = target.copy(); pert_frame += delta
    dev_clean, _ = predict_frame(model, target, n_classes)
    dev_adv, padv = predict_frame(model, pert_frame, n_classes)
    win_pred = padv.argmax(1)
    flipped = (win_pred != true_label).mean()
    print(f"  clean  -> {idx_to_name.get(dev_clean, f'class{dev_clean}')}")
    print(f"  perturbed -> {idx_to_name.get(dev_adv, f'class{dev_adv}')}   "
          f"(windows flipped {flipped*100:.0f}%)")

    # Fooling rate over many run-3 frames using the SAME-budget per-frame PGD.
    # Untargeted: any flip off the true class. Targeted: landing ON the target.
    n_ok = n_hit = n_tot = 0
    for i in idx[:a.eval_n]:
        d, _ = craft(model, frames[i], true_label, a.psr, a.steps,
                     a.step_frac, target_label)
        dadv, _ = predict_frame(model, frames[i] + d, n_classes)
        n_ok += (dadv != true_label)
        n_hit += (target_label is not None and dadv == target_label)
        n_tot += 1
    print(f"  per-frame PGD fooling rate @ PSR {a.psr} dB: {n_ok}/{n_tot} "
          f"= {n_ok/n_tot*100:.0f}% (off {name})")
    if target_label is not None:
        print(f"  per-frame TARGET-HIT rate (-> device_{a.target}): "
              f"{n_hit}/{n_tot} = {n_hit/n_tot*100:.0f}%")

    # ── Write OTA files: unit-RMS frame + PSR-scaled delta, padded to L ──
    L = int(round(a.period_ms * 1e-3 * FS))
    act = slice(PRE_ROLL, PRE_ROLL + ACTIVE)
    rms_s = np.sqrt(np.mean(np.abs(target[act])**2)) + _EPS
    frame_n = (target / rms_s).astype(np.complex64)            # unit-RMS signal
    delta_n = (delta / rms_s).astype(np.complex64)             # same scale -> PSR preserved

    fbuf = np.zeros(L, dtype=np.complex64); fbuf[:FRAME_LEN] = frame_n
    pbuf = np.zeros(L, dtype=np.complex64); pbuf[:FRAME_LEN] = delta_n
    fbuf.tofile(a.out_frame); pbuf.tofile(a.out_pert)

    sig_p = np.mean(np.abs(frame_n[act])**2)
    pert_p = np.mean(np.abs(delta_n[act])**2)
    print(f"\nWrote {a.out_frame} and {a.out_pert}")
    print(f"  L={L} ({a.period_ms} ms)  frame at offset 0  active {PRE_ROLL}:{PRE_ROLL+ACTIVE}")
    print(f"  achieved PSR = {10*np.log10(pert_p/sig_p):.2f} dB (target {a.psr})")
    print(f"  -> .grc: loop frame.bin on ch0, perturbation.bin*epsilon(=1.0) on ch1")


if __name__ == '__main__':
    main()

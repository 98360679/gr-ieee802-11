#!/usr/bin/env python3
"""
exp3_make_perturbation_eot.py — alignment-ROBUST targeted δ via EOT
────────────────────────────────────────────────────────────────────────────────
The plain per-frame δ (exp3_make_perturbation.py) fools 92% digitally but dies at a
1-sample timing skew or >45° phase (exp3_loopback_sim.py) — useless for the real
2-radio OTA where the adversary's δ arrives with arbitrary sub-sample delay + phase.

This crafts δ with Expectation-Over-Transformation: each PGD step averages the
gradient over K random draws of
  * fractional time shift  τ ~ U(-shift, +shift) samples   (differentiable FFT shift)
  * phase rotation         φ ~ U(-phase, +phase) deg
so the optimum fools across the whole (τ,φ) cloud instead of one exact alignment.

δ is one CONTIGUOUS vector over the active region (ACTIVE = 14*WIN), so a shift moves
energy across window boundaries — exactly what the hardware misalignment does. The
per-window L2 (PSR) budget is enforced on the UNSHIFTED δ (shift/rotate are unitary,
so realized PSR is preserved).

Writes frame.bin + perturbation_eot.bin (sample-aligned, padded to the loop period)
and prints the robustness sweep vs the plain δ.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from exp3_make_perturbation import load_fp_model, craft, predict_frame
from exp3_train_fingerprint import load_frames
from exp3_fp_model import (torch_iq_to_input, PRE_ROLL, ACTIVE, WIN, FRAME_LEN, FS)

_EPS = 1e-12
NW = ACTIVE // WIN          # 14 non-overlapping windows over the active region
assert NW * WIN == ACTIVE


def frac_shift(x, tau):
    """Differentiable fractional sample shift of complex x[...,N] by tau samples."""
    N = x.shape[-1]
    f = torch.fft.fftfreq(N, device=x.device)
    return torch.fft.ifft(torch.fft.fft(x) * torch.exp(-2j * np.pi * f * tau))


def craft_eot(model, frame, true_label, target_label, psr_db, dev,
              steps=150, step_frac=0.06, n_eot=8, shift=1.5, phase_deg=180.0,
              method='pgd'):
    """EOT attack. method='pgd' -> iterative EOT-PGD; method='fgsm' -> single-step
    EOT-FGSM (one sign-gradient step filled to the per-window PSR budget). target_label=k
    -> TARGETED (descend CE toward k); target_label=None -> UNTARGETED (ascend CE on
    true_label). Returns delta over the FULL frame (complex)."""
    sig = torch.from_numpy(frame.astype(np.complex64)).to(dev)
    act = sig[PRE_ROLL:PRE_ROLL + ACTIVE]                  # [ACTIVE] complex
    sig_w = act.reshape(NW, WIN)
    sig_norm = torch.sqrt((sig_w.real**2 + sig_w.imag**2).sum(1) + _EPS)   # [NW]
    budget = (10.0 ** (psr_db / 20.0)) * sig_norm          # per-window ||δ_w||₂

    d = torch.zeros(ACTIVE, 2, device=dev)
    d.normal_(0, 1e-3); d.requires_grad_(True)
    if target_label is None:                                # untargeted: ascend CE(true)
        y = torch.full((NW,), true_label, dtype=torch.long, device=dev); sign = 1.0
    else:                                                   # targeted: descend CE(target)
        y = torch.full((NW,), target_label, dtype=torch.long, device=dev); sign = -1.0
    ph_max = np.deg2rad(phase_deg)

    def project(d):
        dw = d.detach().reshape(NW, WIN, 2)
        dn = dw.flatten(1).norm(dim=1)
        scale = torch.clamp(budget / dn.clamp_min(_EPS), max=1.0)
        dw *= scale.view(-1, 1, 1)
        return dw.reshape(ACTIVE, 2)

    def eot_grad():
        """EOT-averaged gradient of CE(y) wrt d over random (shift, phase) draws."""
        if d.grad is not None:
            d.grad.zero_()
        dc = torch.complex(d[:, 0], d[:, 1])               # [ACTIVE]
        loss = 0.0
        for _k in range(n_eot):
            tau = float(torch.empty(1).uniform_(-shift, shift))
            phi = float(torch.empty(1).uniform_(-ph_max, ph_max))
            dct = frac_shift(dc, tau) * np.exp(1j * phi)
            comb = (act + dct).reshape(NW, WIN)            # [NW,WIN] complex
            loss = loss + F.cross_entropy(model(torch_iq_to_input(comb)), y)
        (loss / n_eot).backward()
        return d.grad.reshape(NW, WIN, 2)                  # [NW,WIN,2]

    if method == 'fgsm':                                   # single sign-step to the budget
        gw = eot_grad()
        with torch.no_grad():
            direction = sign * torch.sign(gw)              # FGSM (untgt ascend / tgt descend)
            dn = direction.flatten(1).norm(dim=1).clamp_min(_EPS)   # [NW]
            d.copy_((direction * (budget / dn).view(NW, 1, 1)).reshape(ACTIVE, 2))
    else:                                                  # iterative PGD
        for _ in range(steps):
            gw = eot_grad()
            with torch.no_grad():
                gnorm = gw.flatten(1).norm(dim=1).clamp_min(_EPS)   # [NW]
                step = (step_frac * budget / gnorm).view(NW, 1, 1) * gw
                d += sign * step.reshape(ACTIVE, 2)        # untargeted ascend / targeted descend
                d.copy_(project(d))

    delta_full = np.zeros(FRAME_LEN, dtype=np.complex64)
    dd = d.detach().cpu().numpy()
    delta_full[PRE_ROLL:PRE_ROLL + ACTIVE] = dd[:, 0] + 1j * dd[:, 1]
    return delta_full


def hit_under(model, frame, delta, true, tgt, nc, tau=0.0, deg=0.0):
    """target-hit (pred==tgt) for one frame with δ shifted τ samples, rotated deg."""
    d = torch.from_numpy(delta[PRE_ROLL:PRE_ROLL + ACTIVE].copy())
    dct = (frac_shift(d, tau) * np.exp(1j * np.deg2rad(deg))).numpy()
    f = frame.copy()
    f[PRE_ROLL:PRE_ROLL + ACTIVE] += dct.astype(np.complex64)
    pred, _ = predict_frame(model, f, nc)
    return int(pred == tgt), int(pred != true)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=int, default=6)
    p.add_argument('--target', type=int, default=4)
    p.add_argument('--model', default='fingerprint_cnn_ft20260627.pt')
    p.add_argument('--psr', type=float, default=-20.0)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--n-eot', type=int, default=8)
    p.add_argument('--shift', type=float, default=1.5, help='max |τ| samples')
    p.add_argument('--phase', type=float, default=180.0, help='max |φ| deg (180=full)')
    p.add_argument('--frames', type=int, default=10, help='frames to craft+average')
    p.add_argument('--out-frame', default=None)
    p.add_argument('--out-pert', default=None)
    a = p.parse_args()

    dev = 'cpu'                       # tiny model; keeps imported CPU helpers happy
    model, nc, n2i, i2n = load_fp_model(a.model)
    true, tgt = n2i[f'device_{a.device}'], n2i[f'device_{a.target}']
    print(f"{a.model}: device_{a.device}->device_{a.target}  PSR {a.psr}  EOT "
          f"shift±{a.shift} samp / phase±{a.phase}°  n_eot {a.n_eot}  dev {dev}\n")

    frames_all, run = load_frames(a.device)
    idx = np.where(run == 3)[0]
    chosen, eot_ds, plain_ds = [], [], []
    for i in idx:
        if len(chosen) >= a.frames:
            break
        pred, _ = predict_frame(model, frames_all[i], nc)
        if pred != true:
            continue
        de = craft_eot(model, frames_all[i], true, tgt, a.psr, dev,
                       a.steps, n_eot=a.n_eot, shift=a.shift, phase_deg=a.phase)
        dp, _ = craft(model, frames_all[i], true, a.psr, 100, 0.1, tgt)
        chosen.append(frames_all[i]); eot_ds.append(de); plain_ds.append(dp)
        print(f"  crafted frame #{i}  ({len(chosen)}/{a.frames})")
    model_cpu = model                                       # eval helper runs on CPU

    def sweep(deltas, taus, degs):
        out = {}
        for t in taus:
            out[('t', t)] = np.mean([hit_under(model_cpu, f, d, true, tgt, nc, tau=t)[0]
                                     for f, d in zip(chosen, deltas)]) * 100
        for g in degs:
            out[('p', g)] = np.mean([hit_under(model_cpu, f, d, true, tgt, nc, deg=g)[0]
                                     for f, d in zip(chosen, deltas)]) * 100
        return out

    taus = [0, 0.5, 1, 2, 4]
    degs = [0, 45, 90, 135, 180]
    eot, plain = sweep(eot_ds, taus, degs), sweep(plain_ds, taus, degs)

    print(f"\n  →device_{a.target} hit-rate over {len(chosen)} frames "
          f"(PLAIN vs EOT):")
    print(f"  {'TIMING skew (samp)':>20}   " +
          "  ".join(f"{t:>5}" for t in taus))
    print(f"  {'plain':>20}   " + "  ".join(f"{plain[('t',t)]:>5.0f}" for t in taus))
    print(f"  {'EOT':>20}   " + "  ".join(f"{eot[('t',t)]:>5.0f}" for t in taus))
    print(f"  {'PHASE (deg)':>20}   " + "  ".join(f"{g:>5}" for g in degs))
    print(f"  {'plain':>20}   " + "  ".join(f"{plain[('p',g)]:>5.0f}" for g in degs))
    print(f"  {'EOT':>20}   " + "  ".join(f"{eot[('p',g)]:>5.0f}" for g in degs))

    # write OTA files for the most-robust frame (best mean over the (τ,φ) grid)
    def robust_score(d, f):
        s = [hit_under(model_cpu, f, d, true, tgt, nc, tau=t)[0] for t in taus]
        s += [hit_under(model_cpu, f, d, true, tgt, nc, deg=g)[0] for g in degs]
        return np.mean(s)
    bi = int(np.argmax([robust_score(d, f) for d, f in zip(eot_ds, chosen)]))
    target, delta = chosen[bi], eot_ds[bi]

    of = a.out_frame or ("/media/nghoselab/T9/Data/session13/ota_dev6/"
                         "ft20260627_t4/frame_eot.bin")
    op = a.out_pert or ("/media/nghoselab/T9/Data/session13/ota_dev6/"
                        "ft20260627_t4/perturbation_eot.bin")
    L = int(round(50.0e-3 * FS))
    act = slice(PRE_ROLL, PRE_ROLL + ACTIVE)
    rms = np.sqrt(np.mean(np.abs(target[act])**2)) + _EPS
    fb = np.zeros(L, dtype=np.complex64); fb[:FRAME_LEN] = (target / rms).astype(np.complex64)
    pb = np.zeros(L, dtype=np.complex64); pb[:FRAME_LEN] = (delta / rms).astype(np.complex64)
    fb.tofile(of); pb.tofile(op)
    sp = np.mean(np.abs((target/rms)[act])**2); pp = np.mean(np.abs((delta/rms)[act])**2)
    print(f"\n  wrote {of}\n        {op}")
    print(f"  realized PSR {10*np.log10(pp/sp):.2f} dB   (best-robust frame #{bi})")


if __name__ == '__main__':
    main()

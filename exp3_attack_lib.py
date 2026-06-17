#!/usr/bin/env python3
"""
exp3_attack_lib.py — shared harness for the channel-aware OTA perturbation study
────────────────────────────────────────────────────────────────────────────────
Models the receiver superposition from Kim et al. (arXiv:2005.05321):

      r = x  +  alpha * e^{j*phi} * (h ⊛ delta)  +  (noise already in x)

  • x      : the real captured complex window (already contains signal + noise)
  • delta  : adversary perturbation (unit-norm complex direction)
  • alpha  : scale that sets the perturbation power (PSR / PNR budget)
  • phi    : random phase offset adversary→receiver (phase-robustness)
  • h      : adversary→receiver channel (flat complex gain by default; multi-tap
             convolution supported for later channel-aware experiments)

The classifier pipeline is reproduced EXACTLY and differentiably:
      r (complex) -> per-window unit-power normalize -> [I,Q] -> CNN

so adversarial gradients flow back through the same normalization the model
was trained/evaluated with.
"""
import os, glob
import numpy as np
import torch
import torch.nn.functional as F
from exp3_train_fingerprint import CNN1D

DATA_DEFAULT   = '/media/nghoselab/T9/Data/session12/processed/dataset_win1024.npz'
CKPT_DEFAULT   = '/media/nghoselab/T9/Data/session12/processed/fingerprint_cnn.pt'
FRAMES_DEFAULT = '/media/nghoselab/T9/Data/session12/processed/frames'
EPS = 1e-12


# ── model + data ───────────────────────────────────────────────────────
def load_model(ckpt=CKPT_DEFAULT, device='cuda:0'):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    names = ck['names']
    model = CNN1D(n_classes=len(names)).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, names, int(ck['win'])


def load_raw_windows(split='test', data=DATA_DEFAULT, frames_dir=FRAMES_DEFAULT):
    """Reconstruct the EXACT raw complex windows for a split from frame archives.

    Returns z (N,L) complex64 tensor, y (N,) long, meta (N,4) int [dev,run,frame,off].
    """
    z = np.load(data)
    L = int(z['win'])
    yk, mk = f'y_{split}', f'm_{split}'
    y, meta = z[yk], z[mk]
    F_by_dev = {}
    for fp in glob.glob(os.path.join(frames_dir, 'frames_dev*.npz')):
        zz = np.load(fp)
        F_by_dev[int(zz['device'])] = zz['frames']
    out = np.empty((len(y), L), dtype=np.complex64)
    for i, (dev, run, frame, off) in enumerate(meta):
        out[i] = F_by_dev[dev][frame][off:off + L]
    return (torch.from_numpy(out),
            torch.from_numpy(y.astype(np.int64)),
            torch.from_numpy(meta.astype(np.int64)))


# ── differentiable receiver pipeline ───────────────────────────────────
def normalize_to_input(r):
    """Complex (N,L) -> per-window unit-power 2-channel real (N,2,L)."""
    power = (r.real ** 2 + r.imag ** 2).mean(dim=-1, keepdim=True)
    scale = torch.rsqrt(power + EPS)
    return torch.stack([r.real * scale, r.imag * scale], dim=1)


def apply_channel(delta, h=None):
    """delta (L,) or (N,L) complex; h None=identity, scalar, or (taps,) complex FIR."""
    if h is None:
        return delta
    if h.ndim == 0 or (h.ndim == 1 and h.numel() == 1):
        return delta * h
    # multi-tap: causal convolution, keep length
    d = delta if delta.ndim == 2 else delta.unsqueeze(0)
    L = d.shape[-1]
    dr = torch.nn.functional.conv1d(
        torch.stack([d.real, d.imag], 0).unsqueeze(0),  # placeholder; complex conv below
        torch.ones(1, 1, 1))  # not used
    raise NotImplementedError("multi-tap channel: add when needed")


def received(x, delta, alpha, phi=None, h=None):
    """Superpose perturbation onto captured windows.

    x     : (N,L) complex captured windows
    delta : (L,) or (N,L) complex UNIT-NORM direction (mean |.|^2 == 1)
    alpha : scalar or (N,1) real perturbation amplitude (sqrt of perturbation power)
    phi   : None or (N,) real random phase per window
    """
    d = delta if delta.ndim == 2 else delta.unsqueeze(0)
    d = apply_channel(d, h)
    if phi is not None:
        rot = torch.polar(torch.ones_like(phi), phi).unsqueeze(-1)  # (N,1) complex
        d = d * rot
    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(alpha, dtype=x.real.dtype, device=x.device)
    return x + alpha * d


def forward_logits(model, r):
    return model(normalize_to_input(r))


# ── power / ratio helpers ──────────────────────────────────────────────
def unit_norm(delta):
    """Scale a complex direction to mean power 1 (so alpha^2 == perturbation power)."""
    p = (delta.real ** 2 + delta.imag ** 2).mean()
    return delta / torch.sqrt(p + EPS)


def signal_power(x):
    """Mean per-window signal power (N,)."""
    return (x.real ** 2 + x.imag ** 2).mean(dim=-1)


def alpha_for_psr(x, psr_db):
    """Per-window perturbation amplitude for a target PSR (perturbation-to-signal, dB).

    Returns (N,1) so each window's perturbation power = signal_power * 10^(psr/10).
    """
    ps = signal_power(x)
    p_pert = ps * (10.0 ** (psr_db / 10.0))
    return torch.sqrt(p_pert).unsqueeze(-1)


def alpha_for_pnr(pnr_db, noise_power):
    """Scalar perturbation amplitude for a target PNR given a noise power level."""
    p_pert = noise_power * (10.0 ** (pnr_db / 10.0))
    return float(np.sqrt(p_pert))


def estimate_noise_power(frames_dir=FRAMES_DEFAULT, tail=256):
    """Median power of frame tails (post-payload) across devices = receiver noise floor.

    Frames are FRAME_LEN=15360 with active signal up to ~sample 14856, so only the
    last few hundred samples are genuine noise — keep `tail` small to avoid signal
    leakage into the estimate.
    """
    vals = []
    for fp in glob.glob(os.path.join(frames_dir, 'frames_dev*.npz')):
        F = np.load(fp)['frames']
        t = F[:, -tail:]
        vals.append((np.abs(t) ** 2).mean(1))
    v = np.concatenate(vals)
    return float(np.median(v))


# ── evaluation ─────────────────────────────────────────────────────────
@torch.no_grad()
def accuracy(model, x, y, bs=512, device='cuda:0', delta=None, alpha=0.0,
             phi=None, h=None):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        xb = x[i:i+bs].to(device)
        yb = y[i:i+bs].to(device)
        if delta is not None:
            ph = None if phi is None else phi[i:i+bs].to(device)
            al = alpha[i:i+bs].to(device) if torch.is_tensor(alpha) and alpha.ndim else alpha
            rb = received(xb, delta.to(device), al, ph, h)
        else:
            rb = xb
        pred = forward_logits(model, rb).argmax(1)
        correct += (pred == yb).sum().item()
    return correct / len(x)

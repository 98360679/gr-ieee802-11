#!/usr/bin/env python3
"""
exp3_fp_model.py — RF-fingerprint CNN + preprocessing (shared by train & attack)
────────────────────────────────────────────────────────────────────────────────

Single source of truth for:
  * how a complex IQ window becomes the model input  (frame_to_windows / iq_to_input)
  * the CNN architecture                              (FingerprintCNN)

Both the trainer (exp3_train_fingerprint.py) and the attack/perturbation generator
import from here, so the model is attacked under EXACTLY the preprocessing it was
trained with — which is what makes a crafted perturbation faithful.

Conventions (we own these because we train from scratch):
  * fs = 5 MHz, frame_len = 15360, pre_roll = 256  (matches the captured frames)
  * window = 1024 samples, taken over the active region [pre_roll : pre_roll+ACTIVE]
  * per-window UNIT-RMS normalization (removes power/distance, keeps hardware
    fingerprint). The same normalization defines the PSR budget for the attack.
  * model input tensor: [N, 2, 1024]  (channel 0 = I, channel 1 = Q)
"""

import numpy as np
import torch
import torch.nn as nn

FS         = 5_000_000
FRAME_LEN  = 15360
PRE_ROLL   = 256
WIN        = 1024
ACTIVE     = 14336            # 14 * 1024, active span after pre_roll
NUM_CLASSES = 6
DEVICE_NAMES = [f"device_{i}" for i in range(1, 7)]

_EPS = 1e-12


def frame_to_windows(frame, hop=WIN):
    """(FRAME_LEN,) complex64 frame -> (n_win, WIN) complex64 windows.

    Windows tile the active region [PRE_ROLL : PRE_ROLL+ACTIVE]. hop=WIN gives
    non-overlapping inference windows; a smaller hop augments training data.
    """
    frame = np.asarray(frame).ravel()
    lo, hi = PRE_ROLL, PRE_ROLL + ACTIVE
    starts = range(lo, hi - WIN + 1, hop)
    return np.stack([frame[s:s + WIN] for s in starts]).astype(np.complex64)


def iq_to_input(win):
    """(N, WIN) complex windows -> (N, 2, WIN) float32, per-window unit-RMS."""
    win = np.asarray(win)
    if win.ndim == 1:
        win = win[None, :]
    rms = np.sqrt(np.mean(np.abs(win) ** 2, axis=1, keepdims=True)) + _EPS
    w = win / rms
    return np.stack([w.real, w.imag], axis=1).astype(np.float32)


def torch_iq_to_input(win_c):
    """Differentiable version for attack crafting.

    win_c: complex torch tensor [N, WIN]. Returns [N, 2, WIN] float, unit-RMS.
    Kept identical in math to iq_to_input so train/attack agree.
    """
    rms = torch.sqrt(torch.mean(win_c.real ** 2 + win_c.imag ** 2,
                                dim=1, keepdim=True) + _EPS)
    w = win_c / rms
    return torch.stack([w.real, w.imag], dim=1).float()


class FingerprintCNN(nn.Module):
    """1-D CNN over [2,1024] IQ windows -> 6 device classes (~180k params)."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        def block(ci, co, k):
            return nn.Sequential(
                nn.Conv1d(ci, co, k, padding=k // 2),
                nn.BatchNorm1d(co), nn.ReLU(inplace=True),
                nn.MaxPool1d(2))

        self.features = nn.Sequential(
            block(2,   32, 7),     # 1024 -> 512
            block(32,  64, 5),     # 512  -> 256
            block(64,  64, 5),     # 256  -> 128
            block(64, 128, 3),     # 128  -> 64
            block(128, 128, 3),    # 64   -> 32
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def n_params(model):
    return sum(p.numel() for p in model.parameters())

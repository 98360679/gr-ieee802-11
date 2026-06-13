"""
Models for IQ device fingerprinting.

  CNN1D  — real-valued 1-D CNN. Input (B, 2, L): channel 0/1 = real/imag part
           of either the time-domain window or its (shifted) FFT.

  CVNN   — complex-valued 1-D CNN. Input (B, 1, L) complex64 tensor. Uses
           complex convolutions / complex BatchNorm (split-real approx) /
           modReLU, then takes magnitude before the classifier head.

Both expose the same constructor signature (n_classes, in_len).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────── real 1-D CNN ───────────────────────────
class CNN1D(nn.Module):
    def __init__(self, n_classes=4, in_len=512, ch=2, width=64, p_drop=0.3):
        super().__init__()

        def block(ci, co, k, s):
            return nn.Sequential(
                nn.Conv1d(ci, co, k, stride=s, padding=k // 2, bias=False),
                nn.BatchNorm1d(co), nn.ReLU(inplace=True))

        self.features = nn.Sequential(
            block(ch,        width,     7, 2),
            block(width,     width,     5, 2),
            block(width,     width * 2, 5, 2),
            block(width * 2, width * 2, 3, 2),
            block(width * 2, width * 2, 3, 2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 2, width * 2), nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(width * 2, n_classes))

    def forward(self, x):                       # x: (B,2,L) real
        return self.head(self.pool(self.features(x)))


# ─────────────────────────── complex building blocks ────────────────────
class ComplexConv1d(nn.Module):
    """(a+ib)*(W_r+iW_i) = (a*W_r - b*W_i) + i(a*W_i + b*W_r)."""
    def __init__(self, ci, co, k, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv_r = nn.Conv1d(ci, co, k, stride, padding, bias=bias)
        self.conv_i = nn.Conv1d(ci, co, k, stride, padding, bias=bias)

    def forward(self, xr, xi):
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_r(xi) + self.conv_i(xr)
        return yr, yi


class ComplexBN(nn.Module):
    """Lightweight complex BN: normalize real & imag streams independently."""
    def __init__(self, c):
        super().__init__()
        self.bn_r = nn.BatchNorm1d(c)
        self.bn_i = nn.BatchNorm1d(c)

    def forward(self, xr, xi):
        return self.bn_r(xr), self.bn_i(xi)


class ModReLU(nn.Module):
    """modReLU: relu(|z|+b) * z/|z|. Keeps phase, gates by magnitude."""
    def __init__(self, c):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(1, c, 1))

    def forward(self, xr, xi):
        mag = torch.sqrt(xr ** 2 + xi ** 2 + 1e-9)
        scale = F.relu(mag + self.b) / mag
        return xr * scale, xi * scale


class CVNN(nn.Module):
    def __init__(self, n_classes=4, in_len=512, width=32, p_drop=0.3):
        super().__init__()

        def cblock(ci, co, k, s):
            return nn.ModuleList([
                ComplexConv1d(ci, co, k, stride=s, padding=k // 2),
                ComplexBN(co),
                ModReLU(co)])

        self.blocks = nn.ModuleList([
            cblock(1,         width,     7, 2),
            cblock(width,     width,     5, 2),
            cblock(width,     width * 2, 5, 2),
            cblock(width * 2, width * 2, 3, 2),
            cblock(width * 2, width * 2, 3, 2),
        ])
        feat = width * 2
        self.head = nn.Sequential(
            nn.Linear(feat, feat), nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(feat, n_classes))

    def forward(self, x):                        # x: (B,1,L) complex
        xr, xi = x.real, x.imag
        for conv, bn, act in self.blocks:
            xr, xi = conv(xr, xi)
            xr, xi = bn(xr, xi)
            xr, xi = act(xr, xi)
        mag = torch.sqrt(xr ** 2 + xi ** 2 + 1e-9)   # (B,C,L')
        feat = mag.mean(dim=2)                        # global avg pool -> (B,C)
        return self.head(feat)


def build_model(kind, n_classes, in_len):
    if kind == "cnn1d":
        return CNN1D(n_classes=n_classes, in_len=in_len)
    if kind == "cvnn":
        return CVNN(n_classes=n_classes, in_len=in_len)
    raise ValueError(kind)

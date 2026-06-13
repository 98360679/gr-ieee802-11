#!/usr/bin/env python3
"""
Stage 2 — Exploratory / signal-characterization plots.

Produces, into figs/:
  burst_detection.png   — power envelope + detected bursts on a raw capture
  time_domain.png       — example I/Q window per device (time domain)
  constellation.png     — IQ constellation per device
  psd_welch.png         — Welch PSD per device (frequency domain), Fs=5 MHz
  avg_magnitude_fft.png — mean per-window magnitude spectrum per device
  class_balance.png     — #windows per device per split
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
import config as C

COLORS = plt.cm.tab10(np.linspace(0, 1, len(C.DEVICES)))


def fig_burst_detection():
    """Illustrate envelope thresholding on a slice of one raw capture."""
    path = os.path.join(C.DATA_ROOT, C.DEVICES[0], "run_1.bin")
    n = 12_000_000
    x = np.fromfile(path, dtype=np.complex64, count=n)
    W = C.ENV_WIN
    m = (len(x) // W) * W
    p = (x[:m].real ** 2 + x[:m].imag ** 2).reshape(-1, W).mean(1)
    floor = np.median(p); thr = max(floor * C.THR_FLOOR_K, p.mean() * 2)
    t = np.arange(len(p)) * W / C.FS

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, 10 * np.log10(p + 1e-12), lw=0.6, color="0.4", label="power envelope")
    ax.axhline(10 * np.log10(thr), color="r", ls="--", lw=1, label="burst threshold")
    ax.fill_between(t, -120, 0, where=p > thr, color="orange", alpha=0.3, label="detected burst")
    ax.set_xlabel("time (s)"); ax.set_ylabel("power (dB)")
    ax.set_title(f"Burst detection — {C.DEVICES[0]}/run_1 (first {n/C.FS:.1f} s @ {C.FS/1e6:.0f} MHz)")
    ax.set_ylim(10 * np.log10(floor + 1e-12) - 5, 5); ax.legend(loc="upper right")
    fig.tight_layout(); _save(fig, "burst_detection.png")


def _load():
    d = np.load(C.DATASET, allow_pickle=True)
    return d, list(d["label_names"])


def fig_time_domain(d, names):
    fig, ax = plt.subplots(len(names), 1, figsize=(11, 9), sharex=True)
    for k, nm in enumerate(names):
        Xk = d["X_train"][d["y_train"] == k]
        w = Xk[0]
        ax[k].plot(w.real, lw=0.8, label="I")
        ax[k].plot(w.imag, lw=0.8, label="Q", alpha=0.8)
        ax[k].set_ylabel(nm); ax[k].grid(alpha=.3)
        if k == 0:
            ax[k].legend(loc="upper right", ncol=2)
    ax[-1].set_xlabel("sample")
    fig.suptitle(f"Time-domain I/Q (one normalized window, L={C.WIN_LEN})")
    fig.tight_layout(); _save(fig, "time_domain.png")


def fig_constellation(d, names):
    fig, ax = plt.subplots(1, len(names), figsize=(15, 4))
    for k, nm in enumerate(names):
        Xk = d["X_train"][d["y_train"] == k][:200]   # 200 windows
        z = Xk.reshape(-1)
        ax[k].plot(z.real, z.imag, ".", ms=0.4, alpha=0.15, color=COLORS[k])
        ax[k].set_title(nm); ax[k].set_aspect("equal")
        ax[k].set_xlim(-4, 4); ax[k].set_ylim(-4, 4)
        ax[k].set_xlabel("I");
        if k == 0: ax[k].set_ylabel("Q")
        ax[k].grid(alpha=.3)
    fig.suptitle("IQ constellation (normalized windows)")
    fig.tight_layout(); _save(fig, "constellation.png")


def fig_psd(d, names):
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, nm in enumerate(names):
        Xk = d["X_train"][d["y_train"] == k][:400]
        z = Xk.reshape(-1)
        f, Pxx = welch(z, fs=C.FS, nperseg=512, return_onesided=False)
        f = np.fft.fftshift(f); Pxx = np.fft.fftshift(Pxx)
        ax.plot(f / 1e6, 10 * np.log10(Pxx + 1e-12), lw=1, color=COLORS[k], label=nm)
    ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title(f"Welch PSD per device (Fs={C.FS/1e6:.0f} MHz)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); _save(fig, "psd_welch.png")


def fig_avg_fft(d, names):
    fig, ax = plt.subplots(figsize=(10, 5))
    freqs = np.fft.fftshift(np.fft.fftfreq(C.WIN_LEN, 1 / C.FS)) / 1e6
    for k, nm in enumerate(names):
        Xk = d["X_train"][d["y_train"] == k][:2000]
        mag = np.abs(np.fft.fftshift(np.fft.fft(Xk, axis=1), axes=1))
        mavg = mag.mean(0)
        ax.plot(freqs, 20 * np.log10(mavg + 1e-9), lw=1, color=COLORS[k], label=nm)
    ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("magnitude (dB)")
    ax.set_title("Mean per-window magnitude spectrum (frequency-domain feature)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); _save(fig, "avg_magnitude_fft.png")


def fig_class_balance(d, names):
    splits = ["train", "val", "test"]
    counts = {s: [int((d[f"y_{s}"] == k).sum()) for k in range(len(names))] for s in splits}
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, s in enumerate(splits):
        ax.bar(x + (i - 1) * w, counts[s], w, label=s)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("# windows"); ax.set_title("Class balance per split")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); _save(fig, "class_balance.png")


def _save(fig, name):
    p = os.path.join(C.FIG_DIR, name)
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")


def main():
    fig_burst_detection()
    d, names = _load()
    fig_time_domain(d, names)
    fig_constellation(d, names)
    fig_psd(d, names)
    fig_avg_fft(d, names)
    fig_class_balance(d, names)


if __name__ == "__main__":
    main()

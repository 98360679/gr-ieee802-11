#!/usr/bin/env python3
"""
convert_iq2ch.py
================
Convert existing complex64 .npz traces → 2-channel float32 (I, Q).

Input  npz:  X shape (N, 288)    dtype complex64,  y shape (N,) int  labels 1-7
Output npz:  X shape (N, 288, 2) dtype float32,    y shape (N,) int  labels 0-6

Usage:
    python3 convert_iq2ch.py \
        --input  data/sarp_fft_7devices_signal_only.npz \
        --output data/sarp_iq2ch.npz
"""
import argparse
import numpy as np

def convert(input_path, output_path):
    print(f"Loading {input_path} ...")
    d    = np.load(input_path)
    X, y = d['X'], d['y'].astype(np.int64)

    print(f"  Input  X: {X.shape}  dtype={X.dtype}")
    print(f"  Input  y: {y.shape}  labels={sorted(set(y.tolist()))}")

    # Ensure complex64
    X = X.astype(np.complex64)
    if X.ndim == 2:
        pass  # (N, 288)
    elif X.ndim == 3:
        X = X[:, :, 0]  # (N, 288, 1) → (N, 288)

    # Split into I and Q channels → (N, 288, 2) float32
    I = np.real(X).astype(np.float32)
    Q = np.imag(X).astype(np.float32)
    X2 = np.stack([I, Q], axis=-1)   # (N, 288, 2)

    # RMS normalise per trace (over both channels jointly)
    rms = np.sqrt(np.mean(I**2 + Q**2, axis=1, keepdims=True))  # (N, 1)
    rms = np.maximum(rms, 1e-10)
    X2 = X2 / rms[:, :, np.newaxis]   # broadcast: (N,288,2) / (N,1,1)

    # Convert labels to 0-indexed
    if y.min() == 1:
        y = y - 1

    print(f"  Output X: {X2.shape}  dtype={X2.dtype}")
    print(f"  Output y: {y.shape}   labels={sorted(set(y.tolist()))}")
    print(f"  I range: [{X2[:,:,0].min():.3f}, {X2[:,:,0].max():.3f}]")
    print(f"  Q range: [{X2[:,:,1].min():.3f}, {X2[:,:,1].max():.3f}]")

    np.savez_compressed(output_path, X=X2, y=y)
    print(f"Saved → {output_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    convert(args.input, args.output)

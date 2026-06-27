#!/usr/bin/env python3
"""
Process Raw IQ Collection → SARP Training Data (.npz)
=====================================================
Converts the after-FFT bin files from collect_rx.py into the .npz format
expected by the SARP CVNN training script.

The SARP model expects:
  X: (N, 288) complex64 — FFT traces (288 = 4.5 OFDM symbols × 64 bins)
  y: (N,) int — device labels (1-7)

Processing pipeline:
  1. Load rx_af_deviceN.bin (64-point FFT vectors, complex64)
  2. Reshape into sliding windows of 288 samples (trace_len)
  3. Power-filter to keep only high-energy traces (active frames)
  4. RMS-normalize each trace
  5. Subsample to target count per device
  6. Save as .npz with X and y arrays

Usage:
    python3 process_raw_to_npz.py \
        --input-dir data/raw_iq/ \
        --output data/sarp_fft_7devices_5k.npz \
        --n-devices 7 \
        --samples-per-device 5000 \
        --trace-len 288
"""

import numpy as np
import os
import sys
import argparse
import glob


def load_af_bin(af_path):
    """
    Load after-FFT binary file.
    Format: sequence of 64-point complex64 vectors (from GNU Radio FFT block).
    Returns: (N, 64) complex64 array
    """
    data = np.fromfile(af_path, dtype=np.complex64)
    n_vectors = len(data) // 64
    if n_vectors == 0:
        print(f"  [WARN] Empty file: {af_path}")
        return np.empty((0, 64), dtype=np.complex64)
    data = data[:n_vectors * 64].reshape(n_vectors, 64)
    return data


def extract_traces(af_data, trace_len=288, stride=288):
    """
    Extract SARP traces from after-FFT data.
    Each trace is trace_len consecutive FFT bins (flattened).

    af_data: (N_vectors, 64) complex64
    Returns: (N_traces, trace_len) complex64
    """
    # Flatten the FFT vectors into a continuous stream
    stream = af_data.flatten()
    n_samples = len(stream)

    n_traces = (n_samples - trace_len) // stride + 1
    if n_traces <= 0:
        return np.empty((0, trace_len), dtype=np.complex64)

    traces = np.zeros((n_traces, trace_len), dtype=np.complex64)
    for i in range(n_traces):
        start = i * stride
        traces[i] = stream[start:start + trace_len]

    return traces


def power_filter(traces, percentile=25):
    """
    Filter out low-power traces (noise/silence).
    Keep traces above the given power percentile.
    """
    if len(traces) == 0:
        return traces

    powers = np.mean(np.abs(traces)**2, axis=1)
    threshold = np.percentile(powers, percentile)
    mask = powers > threshold
    filtered = traces[mask]

    print(f"    Power filter: {len(traces)} → {len(filtered)} traces "
          f"(threshold={threshold:.6e}, kept {100*len(filtered)/len(traces):.0f}%)")
    return filtered


def rms_normalize(traces):
    """RMS-normalize each trace (matching SARP training preprocessing)."""
    rms = np.sqrt(np.mean(np.abs(traces)**2, axis=1, keepdims=True))
    rms = np.maximum(rms, 1e-10)
    return traces / rms


def main():
    parser = argparse.ArgumentParser(
        description='Process raw IQ collection into SARP training data')
    parser.add_argument('--input-dir', required=True,
                        help='Base directory with device_N/ subdirs')
    parser.add_argument('--output', required=True,
                        help='Output .npz file path')
    parser.add_argument('--n-devices', type=int, default=7,
                        help='Number of devices')
    parser.add_argument('--samples-per-device', type=int, default=5000,
                        help='Target traces per device')
    parser.add_argument('--trace-len', type=int, default=288,
                        help='Trace length (default: 288)')
    parser.add_argument('--stride', type=int, default=288,
                        help='Stride between traces (default: 288)')
    parser.add_argument('--power-percentile', type=float, default=25,
                        help='Power filter percentile (default: 25)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for subsampling')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Skip RMS normalization (preserves power differences between devices)')
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    all_X = []
    all_y = []

    print(f"\n{'='*60}")
    print(f"  Processing {args.n_devices} devices → {args.output}")
    print(f"  Trace length: {args.trace_len}, Stride: {args.stride}")
    print(f"  Target: {args.samples_per_device} traces/device")
    print(f"{'='*60}\n")

    for dev_id in range(1, args.n_devices + 1):
        dev_dir = os.path.join(args.input_dir, f"device_{dev_id}")

        # Find after-FFT file
        af_pattern = os.path.join(dev_dir, f"rx_af_device{dev_id}.bin")
        if not os.path.exists(af_pattern):
            # Try alternate patterns
            alternatives = glob.glob(os.path.join(dev_dir, "rx_af*.bin"))
            if alternatives:
                af_pattern = alternatives[0]
            else:
                print(f"  [SKIP] Device {dev_id}: no after-FFT file in {dev_dir}/")
                continue

        print(f"  Device {dev_id}: {af_pattern}")

        # Load
        af_data = load_af_bin(af_pattern)
        print(f"    Loaded {len(af_data)} FFT vectors ({len(af_data)*64} samples)")

        if len(af_data) == 0:
            print(f"    [SKIP] No data")
            continue

        # Extract traces
        traces = extract_traces(af_data, args.trace_len, args.stride)
        print(f"    Extracted {len(traces)} traces")

        if len(traces) == 0:
            print(f"    [SKIP] No traces extracted")
            continue

        # Power filter
        traces = power_filter(traces, args.power_percentile)

        if len(traces) == 0:
            print(f"    [SKIP] All traces filtered out")
            continue

        # RMS normalize (skip if --no-normalize)
        if not args.no_normalize:
            traces = rms_normalize(traces)
        else:
            print(f"    [Skipping normalization — preserving raw power]")

        # Subsample to target
        if len(traces) > args.samples_per_device:
            idx = rng.choice(len(traces), args.samples_per_device, replace=False)
            idx.sort()
            traces = traces[idx]
            print(f"    Subsampled to {len(traces)} traces")
        elif len(traces) < args.samples_per_device:
            print(f"    [WARN] Only {len(traces)} traces available "
                  f"(target: {args.samples_per_device})")

        # Labels (1-indexed, matching original SARP training data)
        labels = np.full(len(traces), dev_id, dtype=np.int32)

        all_X.append(traces)
        all_y.append(labels)

        print(f"    → {len(traces)} traces, label={dev_id}")
        print()

    if not all_X:
        print("[ERROR] No data processed!")
        sys.exit(1)

    # Combine all devices
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Shuffle
    shuffle_idx = rng.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez(args.output, X=X, y=y)

    print(f"{'='*60}")
    print(f"  SAVED: {args.output}")
    print(f"  X shape: {X.shape} ({X.dtype})")
    print(f"  y shape: {y.shape} ({y.dtype})")
    print(f"  Labels: {sorted(np.unique(y))}")
    print(f"  Samples per class:")
    for label in sorted(np.unique(y)):
        print(f"    Device {label}: {np.sum(y == label)}")
    print(f"  File size: {os.path.getsize(args.output)/1e6:.1f} MB")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

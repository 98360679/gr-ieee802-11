"""
preprocess_ota.py
=================
Preprocess OTA captured af.bin files into training-ready .npz dataset.

Pipeline:
  - Read af.bin (pre-equalization FFT vectors, complex64) from each device
  - Filter out noise vectors using per-vector power threshold
  - Slice active signal into non-overlapping windows of 288 complex samples
  - Balance classes to minimum trace count across all devices
  - Save to .npz with keys: X (complex64), y (int32)

Usage:
  python3 preprocess_ota.py --data-root /media/nghoselab/T9/Data/ota_processed \
                             --output /media/nghoselab/T9/Data/ota_processed/wifi_7devices_ota_new.npz
"""

import argparse
import os
import numpy as np

TRACE_LENGTH  = 288
FFT_VEC_SIZE  = 64    # af.bin stores 64-point FFT vectors (one per OFDM symbol)
POWER_THRESH  = 1.0   # per-vector power threshold — filters ~91% noise floor
N_DEVICES     = 7


def load_device_traces(af_bin_path, trace_length=TRACE_LENGTH,
                       fft_vec_size=FFT_VEC_SIZE, power_thresh=POWER_THRESH):
    """
    Load af.bin, filter noise vectors by power, then slice into
    non-overlapping windows of trace_length complex samples.

    af.bin layout: continuous stream of 64-point FFT vectors (complex64).
    ~91% of vectors are noise floor (power ~0.01); only ~9% are active
    WiFi transmissions. Slicing without filtering feeds mostly noise to
    the model and prevents learning.

    Pipeline:
      1. Reshape flat stream -> (N, 64) vectors
      2. Compute per-vector power = sum(|x|^2)
      3. Keep only vectors where power > threshold
      4. Flatten surviving vectors back to 1D stream
      5. Slice into non-overlapping trace_length windows
    """
    print(f"  Loading {af_bin_path} ...")
    data = np.fromfile(af_bin_path, dtype=np.complex64)
    print(f"  Raw samples: {len(data):,}")

    # Reshape to (N_vecs, 64) — discard remainder
    n_vecs = len(data) // fft_vec_size
    vecs = data[:n_vecs * fft_vec_size].reshape(n_vecs, fft_vec_size)

    # Power filter: keep only active signal vectors
    power = np.sum(np.abs(vecs)**2, axis=1)
    active_mask = power > power_thresh
    n_active = int(active_mask.sum())
    print(f"  Active vectors (power>{power_thresh}): "
          f"{n_active:,} / {n_vecs:,} ({100*n_active/n_vecs:.1f}%)")

    # Flatten active vectors back to 1D stream
    active_data = vecs[active_mask].reshape(-1)

    # Slice into non-overlapping trace_length windows
    n_traces = len(active_data) // trace_length
    active_data = active_data[:n_traces * trace_length]
    traces = active_data.reshape(n_traces, trace_length)
    print(f"  Traces (non-overlapping, len={trace_length}): {n_traces:,}")
    return traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str,
                        default='/media/nghoselab/T9/Data/ota_processed',
                        help='Root folder containing device_1/ ... device_7/')
    parser.add_argument('--output', type=str,
                        default='/media/nghoselab/T9/Data/ota_processed/wifi_7devices_ota_new.npz',
                        help='Output .npz path')
    parser.add_argument('--power-thresh', type=float, default=POWER_THRESH,
                        help='Per-vector power threshold for noise filtering')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for balancing')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    device_data = {}
    counts = []

    # --- Pass 1: load all devices, record counts ---
    for dev in range(1, N_DEVICES + 1):
        af_path = os.path.join(args.data_root, f'device_{dev}', 'af.bin')
        if not os.path.exists(af_path):
            raise FileNotFoundError(f"Missing: {af_path}")
        print(f"\n[Device {dev} -> class {dev-1}]")
        traces = load_device_traces(af_path, power_thresh=args.power_thresh)
        device_data[dev] = traces
        counts.append(len(traces))

    min_count = min(counts)
    print(f"\n{'='*50}")
    print(f"Trace counts per device: {counts}")
    print(f"Balancing to minimum:    {min_count:,} traces per device")
    print(f"Total dataset size:      {min_count * N_DEVICES:,} traces")
    print(f"{'='*50}\n")

    all_traces = []
    all_labels = []

    # --- Pass 2: subsample to min_count and assemble ---
    for dev in range(1, N_DEVICES + 1):
        traces = device_data[dev]
        idx = rng.choice(len(traces), size=min_count, replace=False)
        idx.sort()
        selected = traces[idx]
        label = dev - 1  # 0-indexed: device 1 -> class 0
        labels = np.full(min_count, label, dtype=np.int32)
        all_traces.append(selected)
        all_labels.append(labels)
        print(f"Device {dev} (class {label}): {min_count:,} traces selected")

    X = np.concatenate(all_traces, axis=0)   # (N*7, 288) complex64
    y = np.concatenate(all_labels, axis=0)   # (N*7,) int32

    # Shuffle
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")
    print(f"Label distribution: { {i: int(np.sum(y==i)) for i in range(N_DEVICES)} }")
    print(f"Saving to {args.output} ...")
    np.savez_compressed(args.output, X=X, y=y)
    print("Done.")


if __name__ == '__main__':
    main()

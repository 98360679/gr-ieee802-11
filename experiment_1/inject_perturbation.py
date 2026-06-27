#!/usr/bin/env python3
"""
Step 4 — Inject FGSM Perturbation into Original TX Signal
==========================================================
Takes the original tx.bin (recorded from wifi_tx_perturb.py) and injects
adversarial perturbations DIRECTLY into the data OFDM symbols, keeping
the preamble (STF + LTF) and SIGNAL field intact.

This is superior to rebuilding frames from scratch because:
  - Preambles are preserved → RX can still sync
  - SIGNAL field is preserved → RX knows rate/length
  - Only data subcarriers are perturbed → controlled attack
  - The RF fingerprint from the original USRP hardware is partially preserved

802.11a Frame Structure (at 5 MHz, 64-pt FFT, CP=16):
  ┌─────────┬─────────┬────────┬────────────────────────┐
  │  STF    │  LTF    │ SIGNAL │  DATA (177 OFDM syms)  │
  │ 160 smp │ 160 smp │ 80 smp │ 177 × 80 = 14160 smp  │
  └─────────┴─────────┴────────┴────────────────────────┘
  Total: 14560 samples per frame

Usage:
    python3 inject_perturbation.py \
        --tx-bin data/tx.bin \
        --tx-log tx-log.txt \
        --weights /media/nghoselab/T9/Data/models/sarp_fft_cvnn_sarp_5k/extracted/model.weights.h5 \
        --device-label 6 \
        --epsilons 0.0 0.01 0.05 0.1 0.15 0.2 0.25 0.3 \
        --output-dir data/perturbed/
"""

import numpy as np
import os
import re
import sys
import json
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

# Suppress benign complex64→float32 casting warning from TF gradient machinery
import warnings
warnings.filterwarnings('ignore', message='.*casting.*complex64.*float32.*')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


# ── 802.11a OFDM parameters ──────────────────────────────────────────
FFT_SIZE    = 64
CP_LEN      = 16
OFDM_SYM_LEN = FFT_SIZE + CP_LEN   # 80 samples
N_DATA_SC   = 48
N_PILOT_SC  = 4

# Preamble lengths (samples)
STF_LEN     = 160
LTF_LEN     = 160
SIGNAL_LEN  = 80   # 1 OFDM symbol
PREAMBLE_TOTAL = STF_LEN + LTF_LEN + SIGNAL_LEN   # 400

# Number of data OFDM symbols per frame (BPSK rate 1/2, 500B payload)
N_DATA_OFDM = 177

# Data subcarrier indices (802.11a standard, 0-indexed into 64-pt FFT)
DATA_SC_CENTERED = (
    list(range(-26, -21)) + list(range(-20, -7)) +
    list(range(-6, 0)) + list(range(1, 7)) +
    list(range(8, 21)) + list(range(22, 27))
)
DATA_BINS = [sc % FFT_SIZE for sc in DATA_SC_CENTERED]  # wrap to 0-63

PILOT_SC_CENTERED = [-21, -7, 7, 21]
PILOT_BINS = [sc % FFT_SIZE for sc in PILOT_SC_CENTERED]


def parse_tx_log(tx_log_path):
    """Parse tx-log.txt for frame IDs and sample offsets."""
    frames = []
    pattern = re.compile(r'frame ID:\s*(\d{4})\s*@\s*sample\s*(\d+)')
    with open(tx_log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                frames.append({
                    'id': m.group(1),
                    'sample_offset': int(m.group(2))
                })
    print(f"[Step 4] Parsed {len(frames)} frames from {tx_log_path}")
    return frames


def extract_data_symbols(tx_signal, frame_offset):
    """
    Extract data subcarrier symbols from one frame in tx.bin.
    Returns: (N_DATA_OFDM, N_DATA_SC) complex64 array
    """
    data_start = frame_offset + PREAMBLE_TOTAL
    symbols = np.zeros((N_DATA_OFDM, N_DATA_SC), dtype=np.complex64)

    for i in range(N_DATA_OFDM):
        ofdm_start = data_start + i * OFDM_SYM_LEN
        ofdm_end = ofdm_start + OFDM_SYM_LEN

        if ofdm_end > len(tx_signal):
            print(f"  [WARN] Frame truncated at OFDM symbol {i}")
            break

        # Strip cyclic prefix, take FFT portion
        time_sym = tx_signal[ofdm_start + CP_LEN : ofdm_end]
        freq_sym = np.fft.fft(time_sym)

        # Extract data subcarriers
        for j, b in enumerate(DATA_BINS):
            symbols[i, j] = freq_sym[b]

    return symbols


def inject_perturbed_symbols(tx_signal, frame_offset, perturbed_symbols):
    """
    Replace data subcarrier values in one frame of tx_signal with
    perturbed values. Preamble and pilot subcarriers are untouched.

    perturbed_symbols: (N_DATA_OFDM, N_DATA_SC) complex64
    """
    data_start = frame_offset + PREAMBLE_TOTAL
    modified = tx_signal.copy()

    for i in range(N_DATA_OFDM):
        ofdm_start = data_start + i * OFDM_SYM_LEN
        ofdm_end = ofdm_start + OFDM_SYM_LEN

        if ofdm_end > len(modified):
            break

        # Strip CP, FFT
        time_sym = modified[ofdm_start + CP_LEN : ofdm_end]
        freq_sym = np.fft.fft(time_sym)

        # Replace data subcarriers only
        for j, b in enumerate(DATA_BINS):
            freq_sym[b] = perturbed_symbols[i, j]

        # IFFT back to time domain
        new_time_sym = np.fft.ifft(freq_sym).astype(np.complex64)

        # Write FFT portion back
        modified[ofdm_start + CP_LEN : ofdm_end] = new_time_sym

        # Update cyclic prefix (last CP_LEN samples of the OFDM symbol)
        modified[ofdm_start : ofdm_start + CP_LEN] = new_time_sym[-CP_LEN:]

    return modified


def compute_ber_bpsk(original_syms, perturbed_syms):
    """
    Compute BER by hard-decision BPSK demodulation on the real part.
    Works for any constellation where sign(Re) carries bit information.
    """
    orig_bits = (np.real(original_syms.flatten()) > 0).astype(int)
    pert_bits = (np.real(perturbed_syms.flatten()) > 0).astype(int)
    n_errors = np.sum(orig_bits != pert_bits)
    n_total = len(orig_bits)
    return n_errors / n_total if n_total > 0 else 0.0, n_errors, n_total


def main():
    parser = argparse.ArgumentParser(
        description='Step 4: Inject FGSM perturbation into original tx.bin')
    parser.add_argument('--tx-bin', required=True,
                        help='Original tx.bin from wifi_tx_perturb.py')
    parser.add_argument('--tx-log', required=True,
                        help='TX log file with frame IDs and sample offsets')
    parser.add_argument('--weights', required=True,
                        help='Path to model.weights.h5')
    parser.add_argument('--device-label', type=int, default=6,
                        help='Target device label (default 6, matching USRP device)')
    parser.add_argument('--epsilons', nargs='+', type=float,
                        default=[0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    parser.add_argument('--attack', default='fgsm', choices=['fgsm', 'pgd'])
    parser.add_argument('--pgd-steps', type=int, default=10)
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--output-dir', default='data/perturbed/')
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load original TX signal ──────────────────────────────────────
    print(f"\n[Step 4] Loading {args.tx_bin}...")
    tx_signal = np.fromfile(args.tx_bin, dtype=np.complex64)
    print(f"  Loaded {len(tx_signal)} samples ({len(tx_signal)*8/1e6:.1f} MB)")

    # ── Parse frame log ──────────────────────────────────────────────
    frames = parse_tx_log(args.tx_log)
    if args.max_frames:
        frames = frames[:args.max_frames]
    print(f"  Processing {len(frames)} frames")

    # ── Import attack engine ─────────────────────────────────────────
    from sarp_attack_engine import SARPAttackEngine

    # ── Process each epsilon ─────────────────────────────────────────
    all_results = {}

    for eps in args.epsilons:
        print(f"\n{'='*65}")
        print(f"  EPSILON = {eps:.4f}")
        print(f"{'='*65}")

        t0 = time.time()

        # Create attack engine (skip for eps=0)
        engine = None
        if eps > 0:
            engine = SARPAttackEngine(
                model_path=args.weights,
                device_label=args.device_label,
                epsilon=eps,
                attack=args.attack,
                pgd_steps=args.pgd_steps,
                num_classes=args.num_classes
            )

        # Start with a copy of the original signal
        perturbed_signal = tx_signal.copy()

        frame_bers = []
        frame_perturbation_power = []

        for fi, frame in enumerate(frames):
            fid = frame['id']
            offset = frame['sample_offset']

            # Check if frame fits in signal
            frame_end = offset + PREAMBLE_TOTAL + N_DATA_OFDM * OFDM_SYM_LEN
            if frame_end > len(tx_signal):
                print(f"  [WARN] Frame {fid} extends beyond signal, skipping")
                continue

            # Extract actual constellation symbols from tx.bin
            original_syms = extract_data_symbols(tx_signal, offset)
            flat_original = original_syms.flatten()  # (8496,)

            if eps == 0:
                flat_perturbed = flat_original.copy()
            else:
                # Run FGSM on the actual symbols
                flat_perturbed = engine.perturb(flat_original.copy())

            # Compute pre-channel BER (perturbation-only BER)
            ber, n_err, n_bits = compute_ber_bpsk(flat_original, flat_perturbed)
            frame_bers.append(ber)

            # Compute perturbation power
            delta = flat_perturbed - flat_original
            pert_power = np.mean(np.abs(delta)**2)
            sig_power = np.mean(np.abs(flat_original)**2)
            frame_perturbation_power.append(pert_power / sig_power if sig_power > 0 else 0)

            # Reshape and inject back into the signal
            perturbed_syms_2d = flat_perturbed.reshape(N_DATA_OFDM, N_DATA_SC)
            perturbed_signal = inject_perturbed_symbols(
                perturbed_signal, offset, perturbed_syms_2d)

            if fi < 5 or fi % 50 == 0:
                print(f"  Frame {fid}: BER={ber:.6e}, "
                      f"pert_power_ratio={frame_perturbation_power[-1]:.6e}, "
                      f"max_delta={np.max(np.abs(delta)):.6f}")

        elapsed = time.time() - t0

        # Save perturbed signal
        eps_str = f"{eps:.2f}"
        out_path = os.path.join(args.output_dir, f"tx_perturbed_eps_{eps_str}.bin")
        perturbed_signal.tofile(out_path)

        # Compute aggregate metrics
        avg_ber = np.mean(frame_bers) if frame_bers else 0
        max_ber = np.max(frame_bers) if frame_bers else 0
        avg_pert_power = np.mean(frame_perturbation_power) if frame_perturbation_power else 0

        all_results[eps] = {
            'avg_ber': float(avg_ber),
            'max_ber': float(max_ber),
            'avg_pert_power_ratio': float(avg_pert_power),
            'n_frames': len(frame_bers),
            'output_file': out_path,
            'file_size_mb': os.path.getsize(out_path) / 1e6,
            'time_s': elapsed
        }

        print(f"\n  Results for eps={eps}:")
        print(f"    Avg BER (pre-channel):     {avg_ber:.6e}")
        print(f"    Max BER (pre-channel):     {max_ber:.6e}")
        print(f"    Avg perturbation power:    {avg_pert_power:.6e}")
        print(f"    Frames processed:          {len(frame_bers)}")
        print(f"    Output: {out_path} ({all_results[eps]['file_size_mb']:.1f} MB)")
        print(f"    Time: {elapsed:.1f}s")

        if engine:
            engine.summary()

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  STEP 4 SUMMARY — Perturbation Injection Results")
    print(f"{'='*80}")
    print(f"  {'Eps':>7}  {'AvgBER':>12}  {'MaxBER':>12}  "
          f"{'PertPower':>12}  {'Frames':>6}  {'Time':>6}")
    print(f"  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*6}  {'-'*6}")
    for eps in args.epsilons:
        r = all_results[eps]
        wifi_ok = "OK" if r['avg_ber'] < 1e-5 else "FAIL"
        print(f"  {eps:>7.3f}  {r['avg_ber']:>12.6e}  {r['max_ber']:>12.6e}  "
              f"{r['avg_pert_power_ratio']:>12.6e}  {r['n_frames']:>6d}  "
              f"{r['time_s']:>5.1f}s  [{wifi_ok}]")
    print(f"  WiFi standard BER threshold: < 1e-5")

    # ── Save results JSON ────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, 'step4_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'results': {str(k): v for k, v in all_results.items()},
            'frame_count': len(frames),
            'tx_bin_samples': len(tx_signal),
        }, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # ── Save CSV for plotting ────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'step4_ber_vs_epsilon.csv')
    with open(csv_path, 'w') as f:
        f.write("epsilon,avg_ber,max_ber,avg_pert_power_ratio,n_frames\n")
        for eps in args.epsilons:
            r = all_results[eps]
            f.write(f"{eps},{r['avg_ber']},{r['max_ber']},"
                    f"{r['avg_pert_power_ratio']},{r['n_frames']}\n")
    print(f"  CSV saved to {csv_path}")
    print(f"\n  Perturbed .bin files ready for Step 5 (retransmission)")


if __name__ == '__main__':
    main()

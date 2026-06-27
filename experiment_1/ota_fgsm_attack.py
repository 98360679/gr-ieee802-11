#!/usr/bin/env python3
"""
OTA Adversarial Attack on SARP RF Fingerprinting
=================================================
Uses sarp_attack_engine.py to generate FGSM perturbations on known TX symbols,
then produces perturbed time-domain IQ files for retransmission.

Usage:
    python3 ota_fgsm_attack.py \
        --weights /media/nghoselab/T9/Data/models/sarp_fft_cvnn_sarp_5k/extracted/model.weights.h5 \
        --tx-log tx-log.txt \
        --tx-bin data/tx.bin \
        --output-dir data/perturbed/ \
        --epsilons 0.0 0.01 0.05 0.1 0.15 0.2 0.25 0.3

Pipeline:
    1. Parse tx-log.txt to get frame IDs and sample offsets
    2. For each frame, regenerate the known BPSK constellation symbols
    3. Run FGSM attack at each epsilon level via SARPAttackEngine
    4. Rebuild perturbed time-domain IQ from perturbed symbols
    5. Save perturbed tx_eps_X.XX.bin files for retransmission (Step 5)
"""

import numpy as np
import os
import re
import sys
import argparse


# ── WiFi OFDM parameters ──────────────────────────────────────────────
FFT_SIZE = 64
CP_LEN = 16
N_DATA_SC = 48

# 802.11a data subcarrier indices (centered around DC)
DATA_SC = (
    list(range(-26, -21)) + list(range(-20, -7)) +
    list(range(-6, 0)) + list(range(1, 7)) +
    list(range(8, 21)) + list(range(22, 27))
)
DATA_BINS = [sc + 32 for sc in DATA_SC]  # shifted to 0-63

PILOT_BINS = [11, 25, 39, 53]  # pilot subcarrier positions
PILOT_VALUES = [1, 1, 1, -1]   # BPSK pilot values (standard 802.11)


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
    print(f"Parsed {len(frames)} frames from {tx_log_path}")
    return frames


def payload_to_bits(frame_id_str, pdu_length=500):
    """
    Regenerate the known TX payload as bits.
    Payload = 4-digit frame ID + repeating '0123456789...' pattern
    """
    payload_str = frame_id_str + ''.join(str(i % 10) for i in range(pdu_length - 4))
    payload_bytes = payload_str.encode('ascii')
    bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
    return bits


def bits_to_bpsk(bits):
    """Convert bits to BPSK symbols: 0 -> -1, 1 -> +1"""
    return (2.0 * bits.astype(np.float32) - 1.0).astype(np.complex64)


def get_frame_n_ofdm(pdu_length=500):
    """
    Calculate number of OFDM symbols for a frame.
    802.11 BPSK 1/2: 24 bits per OFDM symbol (48 data SC * 1 bit/SC * 1/2 rate)
    But after convolutional coding: 48 coded bits per symbol, so 24 data bits.
    Total bits = (MAC header 24B + payload 500B + FCS 4B) * 8 = 4224 bits
    At rate 1/2: 4224 * 2 = 8448 coded bits
    OFDM symbols = ceil(8448 / 48) = 176
    Plus 1 signal symbol = 177 total
    """
    return 177


def symbols_to_time_iq(symbols, n_ofdm):
    """
    Convert data constellation symbols back to time-domain IQ.
    symbols: (n_ofdm * 48,) complex64 array of data subcarrier values
    Returns: time-domain IQ samples with CP
    """
    data_syms = symbols.reshape(n_ofdm, N_DATA_SC)

    # Build full 64-subcarrier frequency vectors
    freq_vectors = np.zeros((n_ofdm, FFT_SIZE), dtype=np.complex64)

    # Place data subcarriers
    for i, b in enumerate(DATA_BINS):
        freq_vectors[:, b] = data_syms[:, i]

    # Add pilots
    for b, pv in zip(PILOT_BINS, PILOT_VALUES):
        freq_vectors[:, b] = pv + 0j

    # fftshift: model expects [left|right] but IFFT needs [DC...pos|neg]
    left = freq_vectors[:, :32]
    right = freq_vectors[:, 32:]
    unshifted = np.concatenate([right, left], axis=1)

    # IFFT to time domain
    time_syms = np.fft.ifft(unshifted).astype(np.complex64)

    # Add cyclic prefix
    cp = time_syms[:, -CP_LEN:]
    with_cp = np.concatenate([cp, time_syms], axis=1)

    # Flatten to continuous IQ stream
    time_signal = with_cp.reshape(-1)
    return time_signal


def compute_ber(original_syms, perturbed_syms):
    """
    Compute BER between original and perturbed BPSK symbols.
    For BPSK: bit = 1 if real(sym) > 0, else 0
    """
    orig_bits = (np.real(original_syms) > 0).astype(int)
    pert_bits = (np.real(perturbed_syms) > 0).astype(int)
    errors = np.sum(orig_bits != pert_bits)
    total = len(orig_bits)
    return errors / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='OTA FGSM Attack on SARP')
    parser.add_argument('--weights', required=True,
                        help='Path to model.weights.h5')
    parser.add_argument('--tx-log', default='tx-log.txt',
                        help='TX log file from wifi_tx_perturb.py')
    parser.add_argument('--tx-bin', default='data/tx.bin',
                        help='Original TX IQ binary (for reference)')
    parser.add_argument('--output-dir', default='data/perturbed/',
                        help='Output directory for perturbed .bin files')
    parser.add_argument('--device-label', type=int, default=0,
                        help='Target device label (0-6)')
    parser.add_argument('--epsilons', nargs='+', type=float,
                        default=[0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                        help='Epsilon values for FGSM')
    parser.add_argument('--attack', default='fgsm', choices=['fgsm', 'pgd'])
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Limit number of frames to process')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Parse TX log ──────────────────────────────────────────
    frames = parse_tx_log(args.tx_log)
    if args.max_frames:
        frames = frames[:args.max_frames]

    n_ofdm = get_frame_n_ofdm()
    n_syms_per_frame = n_ofdm * N_DATA_SC  # 177 * 48 = 8496

    # ── Step 2: Import attack engine ──────────────────────────────────
    from sarp_attack_engine import SARPAttackEngine

    # ── Step 3: For each epsilon, process all frames ──────────────────
    results = {}

    for eps in args.epsilons:
        print(f"\n{'='*60}")
        print(f"  EPSILON = {eps}")
        print(f"{'='*60}")

        if eps == 0.0:
            engine = None
        else:
            engine = SARPAttackEngine(
                model_path=args.weights,
                device_label=args.device_label,
                epsilon=eps,
                attack=args.attack,
                num_classes=args.num_classes
            )

        all_perturbed_iq = []
        all_original_iq = []
        frame_bers = []
        frame_ids_processed = []

        for fi, frame in enumerate(frames):
            fid = frame['id']

            # Regenerate known payload bits
            payload_bits = payload_to_bits(fid)

            # Convert to BPSK symbols (enough for n_ofdm * 48)
            bpsk_syms = bits_to_bpsk(payload_bits)

            # Pad or truncate to exact symbol count
            if len(bpsk_syms) < n_syms_per_frame:
                bpsk_syms = np.pad(bpsk_syms, (0, n_syms_per_frame - len(bpsk_syms)),
                                   constant_values=-1.0).astype(np.complex64)
            else:
                bpsk_syms = bpsk_syms[:n_syms_per_frame]

            original_syms = bpsk_syms.copy()

            if eps == 0.0:
                perturbed_syms = bpsk_syms.copy()
            else:
                perturbed_syms = engine.perturb(bpsk_syms)

            # Compute BER for this frame
            ber = compute_ber(original_syms, perturbed_syms)
            frame_bers.append(ber)
            frame_ids_processed.append(fid)

            # Convert to time-domain IQ
            perturbed_iq = symbols_to_time_iq(perturbed_syms, n_ofdm)
            original_iq = symbols_to_time_iq(original_syms, n_ofdm)

            all_perturbed_iq.append(perturbed_iq)
            all_original_iq.append(original_iq)

            if fi < 3 or fi % 50 == 0:
                print(f"  Frame {fid}: BER={ber:.6f}, "
                      f"max_pert={np.max(np.abs(perturbed_syms - original_syms)):.6f}")

        # Concatenate all frames
        perturbed_stream = np.concatenate(all_perturbed_iq).astype(np.complex64)

        # Save perturbed IQ
        eps_str = f"{eps:.2f}"
        out_path = os.path.join(args.output_dir, f"tx_perturbed_eps_{eps_str}.bin")
        perturbed_stream.tofile(out_path)

        # Compute aggregate metrics
        avg_ber = np.mean(frame_bers)
        max_ber = np.max(frame_bers)

        results[eps] = {
            'avg_ber': avg_ber,
            'max_ber': max_ber,
            'n_frames': len(frames),
            'output_file': out_path,
            'file_size_mb': os.path.getsize(out_path) / 1e6
        }

        print(f"\n  Results for eps={eps}:")
        print(f"    Avg BER: {avg_ber:.6f}")
        print(f"    Max BER: {max_ber:.6f}")
        print(f"    Frames: {len(frames)}")
        print(f"    Output: {out_path} ({results[eps]['file_size_mb']:.1f} MB)")

        if engine:
            engine.summary()

    # ── Step 4: Summary table ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY: BER vs Epsilon")
    print(f"{'='*60}")
    print(f"  {'Epsilon':>8}  {'Avg BER':>10}  {'Max BER':>10}  {'File':>30}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*30}")
    for eps in args.epsilons:
        r = results[eps]
        print(f"  {eps:>8.3f}  {r['avg_ber']:>10.6f}  {r['max_ber']:>10.6f}  {r['output_file']:>30}")

    # Save results to file
    results_path = os.path.join(args.output_dir, 'attack_results.txt')
    with open(results_path, 'w') as f:
        f.write("epsilon,avg_ber,max_ber,n_frames,output_file\n")
        for eps in args.epsilons:
            r = results[eps]
            f.write(f"{eps},{r['avg_ber']},{r['max_ber']},{r['n_frames']},{r['output_file']}\n")
    print(f"\nResults saved to {results_path}")
    print("Perturbed .bin files ready for retransmission (Step 5)")


if __name__ == '__main__':
    main()

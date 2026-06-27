#!/usr/bin/env python3
"""
Step 6 — BER Calculation & Analysis
=====================================
Computes Bit Error Rate and Frame Error Rate from OTA adversarial attack
experiments. Compares across epsilon levels and generates publication-quality
plots showing the attack effectiveness vs. communication quality trade-off.

Two modes:
  (A) Post-channel BER from RX logs (after OTA transmission)
      Uses rx-log.txt payload decoding vs known TX payload
  (B) Pre-channel BER from Step 4 results (perturbation-only)
      Uses step4_ber_vs_epsilon.csv

Plots generated:
  1. BER vs Epsilon (with WiFi threshold line)
  2. Frame Error Rate vs Epsilon
  3. SARP Accuracy + BER trade-off (dual y-axis)
  4. Constellation diagram comparison
  5. Summary table

Usage:
    # From Step 4 pre-channel results (no RX needed):
    python3 compute_ber_ota.py \
        --mode prechannel \
        --step4-csv data/perturbed/step4_ber_vs_epsilon.csv \
        --output-dir results/

    # From OTA RX logs (after Step 5 transmission):
    python3 compute_ber_ota.py \
        --mode ota \
        --tx-log tx-log.txt \
        --rx-logs rx_eps_0.00.txt rx_eps_0.05.txt rx_eps_0.10.txt \
        --epsilons 0.00 0.05 0.10 \
        --output-dir results/

    # From both (combined analysis):
    python3 compute_ber_ota.py \
        --mode combined \
        --step4-csv data/perturbed/step4_ber_vs_epsilon.csv \
        --tx-log tx-log.txt \
        --rx-logs rx_eps_0.00.txt rx_eps_0.05.txt rx_eps_0.10.txt \
        --epsilons 0.00 0.05 0.10 \
        --output-dir results/
"""

import numpy as np
import os
import re
import sys
import json
import argparse
import csv
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════
# Payload reconstruction (must match wifi_tx_perturb.py)
# ═══════════════════════════════════════════════════════════════════════

def reconstruct_payload(frame_id_str, pdu_length=500):
    """
    Reconstruct the known TX payload bytes for a given frame ID.
    Must match wifi_tx_perturb.py's generate_payload_with_id().
    """
    payload_str = frame_id_str + ''.join(str(i % 10) for i in range(pdu_length - 4))
    return payload_str.encode('utf-8')


def bytes_to_bits(data):
    """Convert bytes to bit array."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


# ═══════════════════════════════════════════════════════════════════════
# RX log parsing
# ═══════════════════════════════════════════════════════════════════════

def parse_rx_log(rx_log_path):
    """
    Parse RX log file for decoded frame IDs and optional payload data.
    Returns list of dicts with frame info.
    """
    frames = []
    pattern_id = re.compile(r'frame ID:\s*(\d{4})')
    pattern_payload = re.compile(r'payload.*?:\s*(.*)')

    with open(rx_log_path) as f:
        for line in f:
            m = pattern_id.search(line)
            if m:
                frames.append({
                    'id': m.group(1),
                    'payload': None  # filled if payload data is in log
                })

    print(f"  Parsed {len(frames)} decoded frames from {rx_log_path}")
    return frames


def compute_ber_from_logs(tx_frames, rx_frames, pdu_length=500):
    """
    Compute BER by comparing known TX payload against RX decoded payload.

    For successfully decoded frames (RX found the frame ID), the 802.11
    FCS check passed, meaning the payload is correct → BER = 0 for those.

    The interesting metric is Frame Error Rate (FER): what fraction of
    TX frames were NOT decoded at the RX.

    Returns dict with BER and FER metrics.
    """
    tx_ids = set(f['id'] for f in tx_frames)
    rx_ids = set(f['id'] for f in rx_frames)

    # Frames successfully decoded
    decoded_ids = tx_ids & rx_ids
    missing_ids = tx_ids - rx_ids

    n_tx = len(tx_ids)
    n_decoded = len(decoded_ids)
    n_missing = len(missing_ids)

    # Frame Error Rate
    fer = n_missing / n_tx if n_tx > 0 else 0.0

    # Frame Delivery Rate
    fdr = n_decoded / n_tx if n_tx > 0 else 0.0

    # For decoded frames, BER = 0 (FCS passed means all bits correct)
    # For missing frames, we assume all bits are errors (worst case)
    bits_per_frame = pdu_length * 8
    total_bits = n_tx * bits_per_frame
    error_bits = n_missing * bits_per_frame  # worst case for missing frames
    ber_worst = error_bits / total_bits if total_bits > 0 else 0.0
    ber_decoded = 0.0  # successfully decoded frames have 0 errors

    return {
        'n_tx_frames': n_tx,
        'n_decoded': n_decoded,
        'n_missing': n_missing,
        'fer': fer,
        'fdr': fdr,
        'ber_worst_case': ber_worst,
        'ber_decoded_only': ber_decoded,
        'missing_frame_ids': sorted(missing_ids),
    }


# ═══════════════════════════════════════════════════════════════════════
# Pre-channel BER (from Step 4 CSV)
# ═══════════════════════════════════════════════════════════════════════

def load_step4_csv(csv_path):
    """Load Step 4 BER results from CSV."""
    results = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps = float(row['epsilon'])
            results[eps] = {
                'avg_ber': float(row['avg_ber']),
                'max_ber': float(row['max_ber']),
                'avg_pert_power_ratio': float(row.get('avg_pert_power_ratio', 0)),
                'n_frames': int(row.get('n_frames', 0)),
            }
    print(f"  Loaded Step 4 results for {len(results)} epsilon values")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def generate_plots(prechannel=None, ota_results=None, output_dir='.'):
    """Generate all analysis plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, LogFormatterMathtext

    os.makedirs(output_dir, exist_ok=True)

    # ── Color scheme ──────────────────────────────────────────────────
    C_BLUE = '#1f77b4'
    C_RED = '#d62728'
    C_GREEN = '#2ca02c'
    C_ORANGE = '#ff7f0e'
    C_PURPLE = '#9467bd'
    WIFI_THRESH = 1e-5

    # ══════════════════════════════════════════════════════════════════
    # Plot 1: BER vs Epsilon
    # ══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))

    if prechannel:
        eps_vals = sorted(prechannel.keys())
        ber_vals = [prechannel[e]['avg_ber'] for e in eps_vals]
        max_ber_vals = [prechannel[e]['max_ber'] for e in eps_vals]

        # Replace 0 with small value for log scale
        ber_plot = [max(b, 1e-10) for b in ber_vals]
        max_ber_plot = [max(b, 1e-10) for b in max_ber_vals]

        ax.semilogy(eps_vals, ber_plot, 'o-', color=C_BLUE,
                     linewidth=2, markersize=8, label='Avg BER (pre-channel)')
        ax.semilogy(eps_vals, max_ber_plot, 's--', color=C_BLUE,
                     linewidth=1, markersize=6, alpha=0.6, label='Max BER (pre-channel)')

    if ota_results:
        eps_ota = sorted(ota_results.keys())
        fer_vals = [ota_results[e]['fer'] for e in eps_ota]
        ber_wc = [max(ota_results[e]['ber_worst_case'], 1e-10) for e in eps_ota]

        ax.semilogy(eps_ota, ber_wc, '^-', color=C_RED,
                     linewidth=2, markersize=8, label='BER worst-case (OTA)')
        # FER on secondary axis
        ax2 = ax.twinx()
        ax2.plot(eps_ota, [f * 100 for f in fer_vals], 'D-', color=C_ORANGE,
                 linewidth=2, markersize=8, label='Frame Error Rate (%)')
        ax2.set_ylabel('Frame Error Rate (%)', color=C_ORANGE, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=C_ORANGE)
        ax2.set_ylim(-5, 105)
        ax2.legend(loc='upper left', fontsize=10)

    # WiFi threshold line
    ax.axhline(y=WIFI_THRESH, color=C_RED, linestyle=':', linewidth=2,
               alpha=0.7, label=f'WiFi BER threshold ({WIFI_THRESH:.0e})')

    # Shade acceptable region
    ax.axhspan(1e-12, WIFI_THRESH, alpha=0.05, color='green')
    ax.text(0.02, WIFI_THRESH * 0.3, 'Acceptable BER', fontsize=9,
            color=C_GREEN, alpha=0.8, transform=ax.get_yaxis_transform())

    ax.set_xlabel('Epsilon (perturbation magnitude)', fontsize=12)
    ax.set_ylabel('Bit Error Rate', fontsize=12)
    ax.set_title('BER vs. Adversarial Perturbation Strength', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-0.005)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot1_ber_vs_epsilon.pdf'), dpi=150)
    fig.savefig(os.path.join(output_dir, 'plot1_ber_vs_epsilon.png'), dpi=150)
    plt.close(fig)
    print(f"  [Saved] plot1_ber_vs_epsilon.pdf/.png")

    # ══════════════════════════════════════════════════════════════════
    # Plot 2: Frame Error Rate vs Epsilon (OTA only)
    # ══════════════════════════════════════════════════════════════════
    if ota_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        eps_ota = sorted(ota_results.keys())
        fdr_vals = [ota_results[e]['fdr'] * 100 for e in eps_ota]
        fer_vals = [ota_results[e]['fer'] * 100 for e in eps_ota]
        n_decoded = [ota_results[e]['n_decoded'] for e in eps_ota]
        n_total = [ota_results[e]['n_tx_frames'] for e in eps_ota]

        ax.bar(range(len(eps_ota)), fdr_vals, color=C_GREEN, alpha=0.7,
               label='Frame Delivery Rate (%)')
        ax.bar(range(len(eps_ota)), fer_vals, bottom=fdr_vals,
               color=C_RED, alpha=0.7, label='Frame Error Rate (%)')

        # Annotate with counts
        for i, (nd, nt) in enumerate(zip(n_decoded, n_total)):
            ax.text(i, fdr_vals[i] / 2, f'{nd}/{nt}', ha='center',
                    va='center', fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(eps_ota)))
        ax.set_xticklabels([f'{e:.2f}' for e in eps_ota])
        ax.set_xlabel('Epsilon', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Frame Delivery vs. Error Rate by Epsilon (OTA)', fontsize=14)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'plot2_fer_vs_epsilon.pdf'), dpi=150)
        fig.savefig(os.path.join(output_dir, 'plot2_fer_vs_epsilon.png'), dpi=150)
        plt.close(fig)
        print(f"  [Saved] plot2_fer_vs_epsilon.pdf/.png")

    # ══════════════════════════════════════════════════════════════════
    # Plot 3: Perturbation Power Ratio (from Step 4)
    # ══════════════════════════════════════════════════════════════════
    if prechannel:
        eps_vals = sorted(prechannel.keys())
        has_power = any(prechannel[e].get('avg_pert_power_ratio', 0) > 0
                        for e in eps_vals)

        if has_power:
            fig, ax = plt.subplots(figsize=(10, 6))

            power_vals = [prechannel[e].get('avg_pert_power_ratio', 0)
                          for e in eps_vals]
            ber_vals = [max(prechannel[e]['avg_ber'], 1e-10) for e in eps_vals]

            ax.semilogy(eps_vals, [max(p, 1e-12) for p in power_vals],
                        'o-', color=C_PURPLE, linewidth=2, markersize=8,
                        label='Perturbation / Signal Power')

            ax2 = ax.twinx()
            ax2.semilogy(eps_vals, ber_vals, 's-', color=C_BLUE,
                         linewidth=2, markersize=8, label='Avg BER')
            ax2.axhline(y=WIFI_THRESH, color=C_RED, linestyle=':',
                        linewidth=2, alpha=0.7)
            ax2.set_ylabel('BER', color=C_BLUE, fontsize=12)
            ax2.tick_params(axis='y', labelcolor=C_BLUE)
            ax2.legend(loc='upper left', fontsize=10)

            ax.set_xlabel('Epsilon', fontsize=12)
            ax.set_ylabel('Perturbation/Signal Power Ratio', color=C_PURPLE,
                          fontsize=12)
            ax.tick_params(axis='y', labelcolor=C_PURPLE)
            ax.set_title('Perturbation Power vs. BER Trade-off', fontsize=14)
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'plot3_power_tradeoff.pdf'), dpi=150)
            fig.savefig(os.path.join(output_dir, 'plot3_power_tradeoff.png'), dpi=150)
            plt.close(fig)
            print(f"  [Saved] plot3_power_tradeoff.pdf/.png")

    # ══════════════════════════════════════════════════════════════════
    # Plot 4: BER Heatmap (epsilon × frame index) if per-frame data
    # ══════════════════════════════════════════════════════════════════
    # (Placeholder — requires per-frame BER data from Step 4 JSON)

    # ══════════════════════════════════════════════════════════════════
    # Plot 5: Summary comparison bar chart
    # ══════════════════════════════════════════════════════════════════
    if prechannel:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        eps_vals = sorted(prechannel.keys())
        ber_vals = [prechannel[e]['avg_ber'] for e in eps_vals]

        # Left: BER bar chart
        colors = [C_GREEN if b < WIFI_THRESH else C_RED for b in ber_vals]
        axes[0].bar(range(len(eps_vals)), ber_vals, color=colors, alpha=0.8)
        axes[0].axhline(y=WIFI_THRESH, color=C_RED, linestyle='--',
                        linewidth=2, label=f'WiFi threshold ({WIFI_THRESH:.0e})')
        axes[0].set_xticks(range(len(eps_vals)))
        axes[0].set_xticklabels([f'{e:.2f}' for e in eps_vals], rotation=45)
        axes[0].set_xlabel('Epsilon')
        axes[0].set_ylabel('Average BER')
        axes[0].set_title('Pre-channel BER by Epsilon')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3, axis='y')

        # Right: WiFi compliance indicator
        compliant = [1 if b < WIFI_THRESH else 0 for b in ber_vals]
        bar_colors = [C_GREEN if c else C_RED for c in compliant]
        axes[1].bar(range(len(eps_vals)), compliant, color=bar_colors, alpha=0.8)
        axes[1].set_xticks(range(len(eps_vals)))
        axes[1].set_xticklabels([f'{e:.2f}' for e in eps_vals], rotation=45)
        axes[1].set_xlabel('Epsilon')
        axes[1].set_ylabel('WiFi Compliant')
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(['FAIL', 'PASS'])
        axes[1].set_title('WiFi BER Compliance (< 1e-5)')

        fig.suptitle('FGSM Attack: Communication Quality Assessment', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'plot5_summary.pdf'), dpi=150)
        fig.savefig(os.path.join(output_dir, 'plot5_summary.png'), dpi=150)
        plt.close(fig)
        print(f"  [Saved] plot5_summary.pdf/.png")


def generate_summary_table(prechannel=None, ota_results=None, output_dir='.'):
    """Print and save summary table."""
    lines = []
    lines.append("=" * 85)
    lines.append("  STEP 6 — BER ANALYSIS SUMMARY")
    lines.append("=" * 85)

    WIFI_THRESH = 1e-5

    if prechannel:
        lines.append("\n  PRE-CHANNEL BER (perturbation-only, before OTA):")
        lines.append(f"  {'Eps':>7}  {'Avg BER':>12}  {'Max BER':>12}  "
                     f"{'PertPower':>12}  {'WiFi':>6}")
        lines.append(f"  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*6}")
        for eps in sorted(prechannel.keys()):
            r = prechannel[eps]
            wifi = "PASS" if r['avg_ber'] < WIFI_THRESH else "FAIL"
            pp = r.get('avg_pert_power_ratio', 0)
            lines.append(f"  {eps:>7.3f}  {r['avg_ber']:>12.6e}  "
                         f"{r['max_ber']:>12.6e}  {pp:>12.6e}  {wifi:>6}")

    if ota_results:
        lines.append("\n  OTA RESULTS (after over-the-air transmission):")
        lines.append(f"  {'Eps':>7}  {'FDR':>8}  {'FER':>8}  "
                     f"{'Decoded':>8}  {'Missing':>8}  {'BER(wc)':>12}")
        lines.append(f"  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")
        for eps in sorted(ota_results.keys()):
            r = ota_results[eps]
            lines.append(f"  {eps:>7.3f}  {r['fdr']:>7.1%}  {r['fer']:>7.1%}  "
                         f"{r['n_decoded']:>8d}  {r['n_missing']:>8d}  "
                         f"{r['ber_worst_case']:>12.6e}")

    lines.append(f"\n  WiFi standard BER threshold: < {WIFI_THRESH:.0e}")
    lines.append("=" * 85)

    text = '\n'.join(lines)
    print(text)

    # Save to file
    summary_path = os.path.join(output_dir, 'step6_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(text + '\n')
    print(f"\n  Summary saved to {summary_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Step 6: BER calculation and analysis')
    parser.add_argument('--mode', required=True,
                        choices=['prechannel', 'ota', 'combined'],
                        help='Analysis mode')

    # Pre-channel (Step 4) inputs
    parser.add_argument('--step4-csv', default=None,
                        help='CSV from Step 4 (step4_ber_vs_epsilon.csv)')
    parser.add_argument('--step4-json', default=None,
                        help='JSON from Step 4 (step4_results.json)')

    # OTA inputs
    parser.add_argument('--tx-log', default=None,
                        help='Original TX log (tx-log.txt)')
    parser.add_argument('--rx-logs', nargs='+', default=None,
                        help='RX log files, one per epsilon')
    parser.add_argument('--epsilons', nargs='+', type=float, default=None,
                        help='Epsilon values corresponding to --rx-logs')
    parser.add_argument('--rx-payload-dir', default=None,
                        help='Directory with RX payload files (rx_payload_eps_X.XX.txt)')

    # Output
    parser.add_argument('--output-dir', default='results/',
                        help='Output directory for plots and summaries')
    parser.add_argument('--pdu-length', type=int, default=500,
                        help='PDU payload length in bytes')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prechannel = None
    ota_results = None

    # ── Load pre-channel results ─────────────────────────────────────
    if args.mode in ('prechannel', 'combined'):
        if args.step4_csv:
            print(f"\n[Step 6] Loading pre-channel results from {args.step4_csv}")
            prechannel = load_step4_csv(args.step4_csv)
        elif args.step4_json:
            print(f"\n[Step 6] Loading pre-channel results from {args.step4_json}")
            with open(args.step4_json) as f:
                data = json.load(f)
            prechannel = {}
            for k, v in data['results'].items():
                prechannel[float(k)] = v
        else:
            print("[ERROR] --step4-csv or --step4-json required for prechannel mode")
            sys.exit(1)

    # ── Load OTA results ─────────────────────────────────────────────
    if args.mode in ('ota', 'combined'):
        if not args.tx_log or not args.rx_logs or not args.epsilons:
            print("[ERROR] --tx-log, --rx-logs, and --epsilons required for OTA mode")
            sys.exit(1)

        if len(args.rx_logs) != len(args.epsilons):
            print("[ERROR] Number of --rx-logs must match --epsilons")
            sys.exit(1)

        # Parse TX log
        tx_frames = []
        pattern = re.compile(r'frame ID:\s*(\d{4})\s*@\s*sample\s*(\d+)')
        with open(args.tx_log) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    tx_frames.append({'id': m.group(1)})
        print(f"\n[Step 6] TX: {len(tx_frames)} frames from {args.tx_log}")

        ota_results = {}
        for eps, rx_log in zip(args.epsilons, args.rx_logs):
            print(f"\n  Processing eps={eps:.2f}: {rx_log}")
            rx_frames = parse_rx_log(rx_log)
            result = compute_ber_from_logs(tx_frames, rx_frames, args.pdu_length)
            ota_results[eps] = result

    # ── Generate summary ─────────────────────────────────────────────
    generate_summary_table(prechannel, ota_results, args.output_dir)

    # ── Generate plots ───────────────────────────────────────────────
    print(f"\n[Step 6] Generating plots...")
    generate_plots(prechannel, ota_results, args.output_dir)

    # ── Save combined results JSON ───────────────────────────────────
    combined = {}
    if prechannel:
        combined['prechannel'] = {str(k): v for k, v in prechannel.items()}
    if ota_results:
        combined['ota'] = {}
        for k, v in ota_results.items():
            v_copy = v.copy()
            v_copy['missing_frame_ids'] = list(v_copy.get('missing_frame_ids', []))
            combined['ota'][str(k)] = v_copy

    results_path = os.path.join(args.output_dir, 'step6_results.json')
    with open(results_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"  Results JSON saved to {results_path}")

    print(f"\n[Step 6] Done! All outputs in {args.output_dir}/")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
collect_rx.py — Patches wifi_rx_perturb.py for the given device and
prints the command to run. You run it directly in a separate terminal.

Usage:
    python3 collect_rx.py --device 0
    python3 collect_rx.py --devices 0 1 2
    python3 collect_rx.py              # all 7 devices
"""

import os, sys, argparse, subprocess

N_DEVICES   = 7
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RX_SCRIPT   = os.path.join(SCRIPT_DIR, "wifi_rx_perturb.py")
OUTPUT_ROOT = "/home/nghoselab/research/adversarial/data"

RX_SERIAL_ORIG = "325936A"
RX_SERIAL      = "325F45B"
RX_FREQ_ORIG   = "2450000000"
FREQ_HZ        = "5290000000"

AF_BIN_ORIG    = "/home/nghoselab/research/adversarial/data/af.bin"
AFTER_BIN_ORIG = "/home/nghoselab/research/adversarial/data/after.bin"
EQU_BIN_ORIG   = "/home/nghoselab/research/adversarial/data/equ.bin"

# Frame ID print line in wifi_rx_perturb.py
FRAME_PRINT_ORIG = 'print(f"frame ID: {frame_id} @ sample {self.rx_sample_counter}")'


def patch_rx(out_dir, device_idx):
    """Patch wifi_rx_perturb.py in-place (backup saved as .bak)."""
    rx_log = os.path.join(out_dir, "rx-log.txt")

    with open(RX_SCRIPT) as f:
        content = f.read()

    # Save backup
    with open(RX_SCRIPT + ".bak", 'w') as f:
        f.write(content)

    replacements = {
        f"serial={RX_SERIAL_ORIG}": f"serial={RX_SERIAL}",
        RX_FREQ_ORIG:                FREQ_HZ,
        AF_BIN_ORIG:    os.path.join(out_dir, "af.bin"),
        AFTER_BIN_ORIG: os.path.join(out_dir, "after.bin"),
        EQU_BIN_ORIG:   os.path.join(out_dir, "equ.bin"),
        FRAME_PRINT_ORIG:
            f'open("{rx_log}", "a").write(f"frame ID: {{frame_id}} @ sample {{self.rx_sample_counter}}\\n")',
    }

    ok = True
    for old, new in replacements.items():
        if old not in content:
            print(f"  [ERROR] Could not find: {old[:60]}")
            ok = False
        else:
            content = content.replace(old, new)
            label = os.path.basename(old) if '/' in old else old[:40]
            print(f"  [OK] {label}")

    if not ok:
        return False

    with open(RX_SCRIPT, 'w') as f:
        f.write(content)
    return True


def restore_rx():
    """Restore wifi_rx_perturb.py from backup."""
    bak = RX_SCRIPT + ".bak"
    if os.path.exists(bak):
        os.replace(bak, RX_SCRIPT)
        print(f"  [OK] Restored {RX_SCRIPT}")
    else:
        print(f"  [WARN] No backup found at {bak}")


def collect_device(device_idx):
    out_dir = os.path.join(OUTPUT_ROOT, f"device_{device_idx}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Device {device_idx} — RX Setup")
    print(f"  Output → {out_dir}")
    print(f"{'='*60}")

    print("\n  Patching wifi_rx_perturb.py...")
    if not patch_rx(out_dir, device_idx):
        print("  [ERROR] Patching failed. Restoring original.")
        restore_rx()
        return False

    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Run this in a NEW terminal:                        │")
    print(f"  │                                                     │")
    print(f"  │  python3 {os.path.basename(RX_SCRIPT):<43}│")
    print(f"  │                                                     │")
    print(f"  │  Stop it with Ctrl+C when TX is done.              │")
    print(f"  └─────────────────────────────────────────────────────┘")

    input("\n  Press Enter here once RX is running and TX is done...")

    print("\n  Checking output files:")
    all_ok = True
    for fname in ["af.bin", "after.bin", "equ.bin", "rx-log.txt"]:
        fpath = os.path.join(out_dir, fname)
        size  = os.path.getsize(fpath) if os.path.exists(fpath) else 0
        print(f"    {fname:12s}: {size//1024:,} KB" if size else f"    {fname:12s}: MISSING")
        if fname == "af.bin" and size == 0:
            all_ok = False

    print("\n  Restoring wifi_rx_perturb.py...")
    restore_rx()

    if not all_ok:
        print(f"  ✗ Device {device_idx} — af.bin missing.")
        return False

    print(f"  ✓ Device {device_idx} complete.")
    return True


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument('--device',  type=int)
    g.add_argument('--devices', type=int, nargs='+')
    return p.parse_args()


def main():
    args = parse_args()
    if args.device is not None:
        devices = [args.device]
    elif args.devices is not None:
        devices = args.devices
    else:
        devices = list(range(N_DEVICES))

    if not os.path.exists(RX_SCRIPT):
        print(f"[ERROR] {RX_SCRIPT} not found"); sys.exit(1)

    print("="*60)
    print("  RF Collection — RX Machine")
    print(f"  Devices  : {devices}")
    print(f"  Frequency: {int(FREQ_HZ)/1e9:.3f} GHz")
    print(f"  Output   : {OUTPUT_ROOT}/device_N/")
    print("="*60)

    results = {}
    for idx in devices:
        results[idx] = collect_device(idx)

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    for idx, ok in results.items():
        print(f"  Device {idx}: {'✓ OK' if ok else '✗ FAILED'}")

    failed = [i for i, ok in results.items() if not ok]
    if failed:
        print(f"\n  Re-run: --devices {' '.join(map(str, failed))}")


if __name__ == '__main__':
    main()

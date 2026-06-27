#!/usr/bin/env python3
"""
save_device.py — Move captured files into device_N/ folder.

Run on RX machine after each capture:
    python3 save_device.py --device 0

Run on TX machine after each capture:
    python3 save_device.py --device 0
"""
import os, shutil, argparse

# RX machine files
RX_DATA_DIR = "/home/nghoselab/research/adversarial/data"
RX_FILES    = ["af.bin", "after.bin", "equ.bin"]

# TX machine files
TX_DATA_DIR = "/home/misty/research/adversarial/data"
TX_FILES    = ["tx.bin"]

def move_files(data_dir, files, device_idx):
    out_dir = os.path.join(data_dir, f"device_{device_idx}")
    os.makedirs(out_dir, exist_ok=True)
    for fname in files:
        src = os.path.join(data_dir, fname)
        dst = os.path.join(out_dir, fname)
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"  {fname} → device_{device_idx}/ ({os.path.getsize(dst)//1024:,} KB)")
        else:
            print(f"  {fname}: NOT FOUND")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=int, required=True)
    args = p.parse_args()

    # Auto-detect which machine we're on
    if os.path.exists(RX_DATA_DIR):
        print(f"RX machine — saving to {RX_DATA_DIR}/device_{args.device}/")
        move_files(RX_DATA_DIR, RX_FILES, args.device)
    elif os.path.exists(TX_DATA_DIR):
        print(f"TX machine — saving to {TX_DATA_DIR}/device_{args.device}/")
        move_files(TX_DATA_DIR, TX_FILES, args.device)
    else:
        print("[ERROR] Could not find data directory on this machine.")

if __name__ == '__main__':
    main()

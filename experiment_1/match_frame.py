import os
import re
import numpy as np

TX_LOG_PATH = "/home/nghoselab/research/adversarial/data/tx-log.txt"
RX_LOG_PATH = "/home/nghoselab/research/adversarial/data/rx-log.txt"
TX_BIN_PATH = "/home/nghoselab/research/adversarial/data/tx.bin"
RX_BIN_PATH = "/home/nghoselab/research/adversarial/data/af.bin"
TX_MATCHED_BIN = "data/tx_matched.bin"
RX_MATCHED_BIN = "data/rx_matched.bin"

def parse_log(filename):
    pattern = re.compile(r'frame ID[: ]+(\d+)\s*@\s*sample\s*(\d+)', re.IGNORECASE)
    entries = {}
    with open(filename, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                fid = match.group(1).zfill(4)
                idx = int(match.group(2))
                entries[fid] = idx
    if not entries:
        print(f"❌ Could not find frame entries in: {filename}")
    return entries

def load_complex_bin(filename):
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return np.array([], dtype=np.complex64)
    return np.fromfile(filename, dtype=np.complex64)

def extract_frame_samples(log, samples, frame_ids, label):
    frames = []
    for i, fid in enumerate(frame_ids):
        start = log[fid]
        try:
            next_fid = frame_ids[i + 1]
            end = log[next_fid]
        except IndexError:
            end = len(samples)

        if start >= len(samples):
            print(f"⚠️ Skipping {label} frame {fid}: start index out of bounds")
            continue

        if end > len(samples):
            print(f"⚠️ Truncated {label} frame {fid}: only {len(samples) - start} samples available")
            end = len(samples)

        frame = samples[start:end]
        frames.append(frame)
    return frames

def save_concat_bin(filename, frames):
    all_samples = np.concatenate(frames).astype(np.complex64)
    all_samples.tofile(filename)
    print(f"✅ Saved {len(all_samples)} samples to {filename}")

def main():
    print("🔍 Parsing logs...")
    tx_log = parse_log(TX_LOG_PATH)
    rx_log = parse_log(RX_LOG_PATH)

    matched_ids = sorted(set(tx_log.keys()) & set(rx_log.keys()))
    if not matched_ids:
        print("❌ No matched frame IDs found.")
        return

    print(f"✅ Matched frame IDs ({len(matched_ids)}): {matched_ids}")

    print("📥 Loading .bin files...")
    tx_samples = load_complex_bin(TX_BIN_PATH)
    rx_samples = load_complex_bin(RX_BIN_PATH)
    print(f"✅ Loaded {len(tx_samples)} TX samples")
    print(f"✅ Loaded {len(rx_samples)} RX samples")

    print("\n✂️ Extracting matched TX frames...")
    tx_matched = extract_frame_samples(tx_log, tx_samples, matched_ids, "TX")

    print("✂️ Extracting matched RX frames...")
    rx_matched = extract_frame_samples(rx_log, rx_samples, matched_ids, "RX")

    print("\n💾 Saving matched frames to new .bin files...")
    save_concat_bin(TX_MATCHED_BIN, tx_matched)
    save_concat_bin(RX_MATCHED_BIN, rx_matched)

if __name__ == "__main__":
    main()


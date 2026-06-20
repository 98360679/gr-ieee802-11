#!/usr/bin/env python3
"""
exp3_ber_eval.py — true bit-error-rate of an OTA capture (perturbed or clean)
─────────────────────────────────────────────────────────────────────────────

The fingerprint eval (exp3_rx_eval.py) tells us whether the adversarial
perturbation fools DEVICE IDENTIFICATION. This tool answers the orthogonal
question for the same capture: does the perturbation also damage the LEGITIMATE
LINK — i.e. what is the bit-error rate of the WiFi payload underneath?

Why a custom harness (and not exp3_wifi_rx.py):
  decode_mac DROPS every frame whose CRC fails (decode_mac.cc: "checksum wrong
  -- dropping"), so the stock 'out' port only ever yields CRC-OK frames — you
  can measure packet success from it, but never the bit errors on the broken
  frames. decode_mac's print_output() however dumps the decoded MAC bytes for
  EVERY completed frame, BEFORE the CRC check, whenever the block is built with
  debug=True. We run the identical sync/equalize/decode chain with debug=True,
  capture that per-frame byte dump, and compare against the known transmitted
  payload bit-for-bit.

Ground truth (exp3_wifi_tx.py):
  MSDU = 'A'*20 + <4-digit frame_id> + 'A'*476   (500 bytes, BPSK 1/2). The
  decoded 802.11 MAC frame is [24B MAC header][500B MSDU][4B FCS]; the MSDU
  starts at offset 24, and "our" frames are dst 42:42:.. / src 23:23:.. .

  The constant-fill MSDU decodes (through this chain's descrambler) to a single
  constant byte at every non-id position, so we don't hard-code that byte: we
  take the REFERENCE payload to be the per-position majority byte (mode) across
  all of our frames, and count deviations from it as bit errors. Positions that
  vary frame-to-frame (the 4-digit id, and any digit that actually changes in
  this capture) have a weak mode and are excluded — so BER is measured only over
  positions that are genuinely constant in the ground truth.

Run:
  python3 exp3_ber_eval.py --file /media/.../perturbed/capture_1.bin
  python3 exp3_ber_eval.py --file <clean.bin> --tag clean --json out.json
"""

import os
import re
import sys
import json
import argparse
import tempfile
import contextlib

sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

import numpy as np
from gnuradio import gr, blocks, fft
from gnuradio.fft import window
import ieee802_11


# ── frame geometry ─────────────────────────────────────────────────────
PDU_LENGTH = 500         # MSDU length in bytes
MAC_HEADER = 24          # bytes before the MSDU in the decoded 802.11 frame
# NB: the constant-fill MSDU decodes to a constant 0x78 ('x') on this rig (NOT
# 0x41 'A' as the TX scripts' source suggests) — but we never hardcode it; the
# reference is the empirical per-position mode (see analyse()).

# RX chain constants — copied verbatim from exp3_wifi_rx.py (the known-good chain)
SAMP_RATE    = 5e6
FREQ         = 2.45e9
WINDOW_SIZE  = 48
SYNC_LENGTH  = 320
CHAN_EST     = 0         # LS
EQ_BANDWIDTH = 200000

_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.int64)


class ber_rx(gr.top_block):
    """Stock gr-ieee802-11 receive chain on a file, decode_mac in debug mode."""
    def __init__(self, path):
        gr.top_block.__init__(self, "Exp3 BER RX", catch_exceptions=True)
        window_size, sync_length = WINDOW_SIZE, SYNC_LENGTH

        src = blocks.file_source(gr.sizeof_gr_complex, path, False)

        self.sync_short = ieee802_11.sync_short(0.56, 2, False, False)
        self.sync_long = ieee802_11.sync_long(sync_length, False, False)
        self.equalizer = ieee802_11.frame_equalizer(
            ieee802_11.Equalizer(CHAN_EST), FREQ, EQ_BANDWIDTH, False, False)
        # debug=True -> print_output() dumps EVERY completed frame, pre-CRC-drop.
        self.decode = ieee802_11.decode_mac(False, True)

        self.fftb = fft.fft_vcc(64, True, window.rectangular(64), True, 1)
        self.s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, 64)
        self.mult = blocks.multiply_vcc(1)
        self.mavg_c = blocks.moving_average_cc(window_size, 1, 4000, 1)
        self.mavg_f = blocks.moving_average_ff(window_size + 16, 1, 4000, 1)
        self.div = blocks.divide_ff(1)
        self.delay16 = blocks.delay(gr.sizeof_gr_complex, 16)
        self.delay_sync = blocks.delay(gr.sizeof_gr_complex, sync_length)
        self.conj = blocks.conjugate_cc()
        self.mag2 = blocks.complex_to_mag_squared(1)
        self.mag = blocks.complex_to_mag(1)

        self.connect((self.mag, 0), (self.div, 0))
        self.connect((self.mag2, 0), (self.mavg_f, 0))
        self.connect((self.conj, 0), (self.mult, 1))
        self.connect((self.delay_sync, 0), (self.sync_long, 1))
        self.connect((self.delay16, 0), (self.conj, 0))
        self.connect((self.delay16, 0), (self.sync_short, 0))
        self.connect((self.div, 0), (self.sync_short, 2))
        self.connect((self.mavg_f, 0), (self.div, 1))
        self.connect((self.mavg_c, 0), (self.mag, 0))
        self.connect((self.mavg_c, 0), (self.sync_short, 1))
        self.connect((self.mult, 0), (self.mavg_c, 0))
        self.connect((self.s2v, 0), (self.fftb, 0))
        self.connect((self.fftb, 0), (self.equalizer, 0))
        self.connect((self.equalizer, 0), (self.decode, 0))
        self.connect((self.sync_long, 0), (self.s2v, 0))
        self.connect((self.sync_short, 0), (self.delay_sync, 0))
        self.connect((self.sync_short, 0), (self.sync_long, 0))
        self.connect((src, 0), (self.mag2, 0))
        self.connect((src, 0), (self.delay16, 0))
        self.connect((src, 0), (self.mult, 0))


# ── parse decode_mac's debug byte-dump ─────────────────────────────────
# Format emitted by decode_mac_impl::print_output() (debug=True):
#   "psdu size<N>"
#   <N hex bytes, space/newline separated>
#   <ascii rendering>
_PSDU_RE = re.compile(r'psdu size(\d+)')
_HEXTOK_RE = re.compile(r'\b[0-9a-f]{2}\b')


def parse_frames(dump_text):
    """Yield decoded MAC frames (bytes) from a decode_mac debug dump."""
    lines = dump_text.splitlines()
    i = 0
    while i < len(lines):
        m = _PSDU_RE.search(lines[i])
        if not m:
            i += 1
            continue
        n = int(m.group(1))
        # collect hex tokens from following lines until we have n bytes
        toks = []
        j = i + 1
        while j < len(lines) and len(toks) < n:
            row = _HEXTOK_RE.findall(lines[j])
            if not row:                 # blank or ascii line -> hex block ended
                if toks:
                    break
                j += 1
                continue
            toks.extend(row)
            j += 1
        if len(toks) >= n:
            yield bytes(int(t, 16) for t in toks[:n])
        i = j


def _is_ours(mac):
    """Our frames: dst addr1 = 42:42:.., src addr2 = 23:23:.. (tolerate bit errors)."""
    if len(mac) < MAC_HEADER + PDU_LENGTH:
        return False
    hdr = np.frombuffer(mac[:MAC_HEADER], dtype=np.uint8)
    score = int((hdr[4:10] == 0x42).sum()) + int((hdr[10:16] == 0x23).sum())
    return score >= 8          # >=8 of the 12 address bytes intact


def analyse(frames, stable_frac=0.5):
    """BER/FER vs an empirical per-position reference (mode over our frames)."""
    payloads = [np.frombuffer(m[MAC_HEADER:MAC_HEADER + PDU_LENGTH], dtype=np.uint8)
                for m in frames if _is_ours(m)]
    n_aligned = len(payloads)
    if n_aligned == 0:
        return {'frames_dumped': len(frames), 'frames_ours_aligned': 0}

    P = np.stack(payloads)                          # (F, 500)
    F = P.shape[0]
    # per-position majority byte (reference) and how many frames agree with it
    ref = np.empty(PDU_LENGTH, dtype=np.uint8)
    ref_count = np.empty(PDU_LENGTH, dtype=np.int64)
    for k in range(PDU_LENGTH):
        vals, cnts = np.unique(P[:, k], return_counts=True)
        j = int(cnts.argmax())
        ref[k] = vals[j]
        ref_count[k] = cnts[j]
    # constant-in-ground-truth positions: one value dominates. Varying id digits
    # (≈uniform over 10 values) fall below the threshold and are excluded.
    stable = ref_count >= max(2, int(stable_frac * F))
    stable_idx = np.where(stable)[0]

    diff = np.bitwise_xor(P[:, stable_idx], ref[stable_idx])   # (F, n_stable)
    biterr_per_frame = _POPCOUNT[diff].sum(axis=1)
    byteerr_per_frame = (diff != 0).sum(axis=1)

    total_bits = int(stable_idx.size * 8 * F)
    total_biterr = int(biterr_per_frame.sum())
    n_clean = int((biterr_per_frame == 0).sum())

    return {
        'frames_dumped': len(frames),
        'frames_ours_aligned': n_aligned,
        'frames_error_free': n_clean,
        'stable_positions': int(stable_idx.size),
        'excluded_positions': int(PDU_LENGTH - stable_idx.size),
        'reference_fill_byte': int(np.bincount(ref[stable_idx]).argmax()),
        'known_bits_compared': total_bits,
        'bit_errors': total_biterr,
        'ber': (total_biterr / total_bits) if total_bits else float('nan'),
        'fer': (1 - n_clean / n_aligned) if n_aligned else float('nan'),
        'worst_frame_byte_errors': int(byteerr_per_frame.max()),
        'mean_frame_byte_errors': float(byteerr_per_frame.mean()),
    }


def main():
    p = argparse.ArgumentParser(description="True BER of an OTA WiFi capture")
    p.add_argument('--file', required=True, help='complex64 capture')
    p.add_argument('--tag', default='capture')
    p.add_argument('--json', default=None, help='write metrics JSON here')
    p.add_argument('--save-dump', default=None, help='keep the raw decode dump')
    a = p.parse_args()

    nsamp = os.path.getsize(a.file) // 8
    print(f"[{a.tag}] {a.file}")
    print(f"  {nsamp} samples ({nsamp/SAMP_RATE*1e3:.1f} ms @ {SAMP_RATE/1e6:.1f} MHz)")
    print("  running gr-ieee802-11 decode (debug dump) ... this streams the whole file")

    dump_path = a.save_dump or tempfile.mktemp(suffix='_ber_dump.txt')
    # decode_mac's print_output goes to C-level std::cout -> redirect fd 1 to a file
    sys.stdout.flush()
    saved = os.dup(1)
    with open(dump_path, 'w') as f:
        os.dup2(f.fileno(), 1)
        try:
            tb = ber_rx(a.file)
            tb.run()                    # file_source(repeat=False) -> ends at EOF
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)

    with open(dump_path) as f:
        dump = f.read()
    frames = list(parse_frames(dump))
    res = analyse(frames)
    res['tag'] = a.tag
    res['file'] = a.file
    res['n_samples'] = nsamp

    print("\n" + "=" * 60)
    print(f"BER eval — {a.tag}")
    print("=" * 60)
    print(f"  frames decoded (dumped) : {res['frames_dumped']}")
    print(f"  ours, payload-aligned   : {res['frames_ours_aligned']}")
    if not res['frames_ours_aligned']:
        print("  (no frames matched our MAC header — nothing to score)")
        print("=" * 60)
    else:
        print(f"  error-free frames       : {res['frames_error_free']}"
              f"  ({100*res['frames_error_free']/res['frames_ours_aligned']:.1f}%)")
        print(f"  reference fill byte     : 0x{res['reference_fill_byte']:02x}  "
              f"(empirical mode)")
        print(f"  constant positions used : {res['stable_positions']}/500  "
              f"(excluded {res['excluded_positions']} varying, incl. frame id)")
        print(f"  known bits compared     : {res['known_bits_compared']}")
        print(f"  bit errors              : {res['bit_errors']}")
        print(f"  >>> BER                 : {res['ber']:.3e}")
        print(f"  >>> FER (frame err rate): {res['fer']:.3f}")
        print(f"  worst/mean byte errors  : {res['worst_frame_byte_errors']} / "
              f"{res['mean_frame_byte_errors']:.2f}  per frame")
        print("=" * 60)

    if a.json:
        with open(a.json, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"  wrote {a.json}")
    if not a.save_dump and os.path.exists(dump_path):
        os.remove(dump_path)


if __name__ == '__main__':
    main()

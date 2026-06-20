#!/usr/bin/env python3
"""
exp3_make_frame.py — generate the canonical WiFi frame waveform (frame.bin)
───────────────────────────────────────────────────────────────────────────

This is deliverable #1 for the per-frame channel-aware adversarial attack.

The attacker (on the other machine) crafts an FGSM/PGD perturbation against ONE
fixed frame waveform. For a single precomputed perturbation to stay aligned
frame-after-frame on the air, that frame must be byte-for-byte identical every
time it is transmitted. So we generate it ONCE, deterministically, here — no
USRP required — and hand the resulting frame.bin to the attack pipeline.

Pipeline (same chain as exp3_wifi_tx.py, but headless + file sink, no radio):
  vector_source -> wifi_phy_hier -> *MULT_CONST -> packet_pad2 -> file_sink

Output layout of frame.bin (complex64 @ SAMP_RATE):
  [ pad_front zeros | preamble+payload (frame_len) | pad_tail zeros ]  == L samples

The script prints the exact contract the perturbation.bin must satisfy:
  - total length  L           (samples)   <- perturbation.bin MUST be this long
  - active offset pad_front   (samples)   <- where the real frame starts
  - active length frame_len   (samples)
  - sample rate   SAMP_RATE
  - frame RMS                             <- for choosing the perturbation/PSR scale

Run:
  python3 exp3_make_frame.py                     # writes ./frame.bin, 50 ms period
  python3 exp3_make_frame.py --period-ms 300     # match the 300 ms TX cadence
  python3 exp3_make_frame.py --out /dev/shm/frame.bin
"""

import sys
import os
import time
import argparse

sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

import numpy as np
import pmt
from gnuradio import gr, blocks
import ieee802_11
import foo
from wifi_phy_hier import wifi_phy_hier


# ─── Configuration (keep in lock-step with exp3_wifi_tx.py) ────────────
SAMP_RATE  = 5e6
MULT_CONST = 0.7      # MUST match clean/perturbed TX for a valid OTA comparison
PDU_LENGTH = 500      # payload bytes per frame
ENCODING   = 0        # 0 = BPSK 1/2
TARGET_PEAK = 0.8     # scale the saved frame so peak |x| <= this. The raw
                      # wifi_phy_hier*0.7 output peaks ~2.3 (7 dB OFDM PAPR),
                      # which HARD-CLIPS the fc32 DAC (clip at 1.0) and destroys
                      # the OFDM decode. Keep headroom under 1.0.

# Capture-time padding around the single frame. pad_front gives the perturbation
# a clean guard before the preamble; pad_tail is just enough to flush the burst.
CAP_PAD_FRONT = 500
CAP_PAD_TAIL  = 2000

# FIXED payload — deterministic, never varies (unlike the live TX which embeds a
# rolling frame_id). One frozen frame == one valid precomputed perturbation.
FIXED_PAYLOAD = ('EXP3' + 'A' * (PDU_LENGTH - 4)).encode('utf-8')


class make_frame(gr.top_block):
    def __init__(self, raw_path):
        gr.top_block.__init__(self, "Exp3 Make Frame", catch_exceptions=True)

        self.wifi_phy_hier_0 = wifi_phy_hier(
            bandwidth=SAMP_RATE,
            chan_est=ieee802_11.LS,
            encoding=ieee802_11.Encoding(ENCODING),
            frequency=2.45e9,            # informational; no RF here
            sensitivity=0.56,
        )
        self.ieee802_11_mac_0 = ieee802_11.mac(
            [0x23] * 6, [0x42] * 6, [0xff] * 6)

        self.foo_packet_pad2_0 = foo.packet_pad2(
            False, False, 0.0, CAP_PAD_FRONT, CAP_PAD_TAIL)
        self.foo_packet_pad2_0.set_min_output_buffer(200000)

        self.blocks_vector_source_x_0 = blocks.vector_source_c(
            [1 + 0j, 0.998 + 0.063j, 0.992 + 0.125j, 0.982 + 0.187j],
            False, 1, [])
        self.blocks_multiply_const_0 = blocks.multiply_const_cc(MULT_CONST)
        # Must hold a whole tagged packet so packet_pad2 gets it in one work().
        self.blocks_multiply_const_0.set_min_output_buffer(200000)

        self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, raw_path, False)
        self.file_sink.set_unbuffered(False)

        self.msg_connect((self.ieee802_11_mac_0, 'phy out'),
                         (self.wifi_phy_hier_0, 'mac_in'))
        self.connect((self.blocks_vector_source_x_0, 0), (self.wifi_phy_hier_0, 0))
        self.connect((self.wifi_phy_hier_0, 0), (self.blocks_multiply_const_0, 0))
        self.connect((self.blocks_multiply_const_0, 0), (self.foo_packet_pad2_0, 0))
        self.connect((self.foo_packet_pad2_0, 0), (self.file_sink, 0))

    def send_one_frame(self):
        pdu = pmt.cons(pmt.PMT_NIL,
                       pmt.init_u8vector(len(FIXED_PAYLOAD),
                                         bytearray(FIXED_PAYLOAD)))
        self.ieee802_11_mac_0.to_basic_block()._post(pmt.intern('app in'), pdu)


def main():
    p = argparse.ArgumentParser(description="Generate the canonical frame.bin")
    p.add_argument('--out', default='./frame.bin', help='output frame.bin path')
    p.add_argument('--period-ms', type=float, default=50.0,
                   help='loop period on air (ms). frame.bin is zero-padded to '
                        'this length so the OTA loop repeats at this cadence.')
    p.add_argument('--raw', default=None,
                   help='optional: also keep the untrimmed capture for debugging')
    args = p.parse_args()

    raw_path = args.raw or (args.out + '.raw')

    print("=" * 64)
    print("Exp3 — generate canonical frame.bin (no USRP)")
    print(f"  encoding={ENCODING}  pdu_len={PDU_LENGTH}B  mult={MULT_CONST}  "
          f"fs={SAMP_RATE/1e6:.1f}MHz")
    print("=" * 64)

    tb = make_frame(raw_path)
    tb.start()
    time.sleep(0.5)            # let the flowgraph spin up
    tb.send_one_frame()
    time.sleep(1.0)            # let the single burst propagate + flush
    tb.stop()
    tb.wait()

    burst = np.fromfile(raw_path, dtype=np.complex64)
    if burst.size == 0:
        print("ERROR: capture is empty — the frame did not propagate.")
        sys.exit(1)

    # Locate the real frame inside the zero padding (pads are exact zeros).
    nz = np.nonzero(np.abs(burst) > 0.0)[0]
    first, last = int(nz[0]), int(nz[-1])
    frame_len = last - first + 1
    pad_front = first                     # samples of zeros before the frame

    # Re-base so the frame sits at a known offset, then zero-pad to the period.
    period_len = int(round(args.period_ms * 1e-3 * SAMP_RATE))
    if period_len < frame_len + pad_front:
        period_len = frame_len + pad_front + CAP_PAD_TAIL
        print(f"  (period too short for frame; bumped to {period_len} samples)")

    out = np.zeros(period_len, dtype=np.complex64)
    # Keep the frame at its natural pad_front offset within the period.
    out[pad_front:pad_front + frame_len] = burst[first:last + 1]

    # Scale so the peak stays safely under the fc32 DAC clip (1.0). Without this
    # the OFDM peak (~2.3) clips ~40% of the active samples and the frame won't
    # decode. perturbation.bin must be crafted against THIS scaled frame.
    raw_peak = float(np.max(np.abs(out)))
    if raw_peak > 0:
        out *= np.float32(TARGET_PEAK / raw_peak)
        print(f"  scaled frame: raw peak {raw_peak:.3f} -> {TARGET_PEAK} "
              f"(k={TARGET_PEAK/raw_peak:.5f})")

    out.tofile(args.out)

    rms = float(np.sqrt(np.mean(np.abs(out[pad_front:pad_front + frame_len]) ** 2)))
    peak = float(np.max(np.abs(out)))

    if args.raw is None:
        os.remove(raw_path)

    print("\nWrote", args.out)
    print("-" * 64)
    print("perturbation.bin CONTRACT (hand these to the attack pipeline):")
    print(f"  dtype          : complex64 (np.complex64 / fc32)")
    print(f"  sample_rate    : {SAMP_RATE:.0f}  ({SAMP_RATE/1e6:.1f} MHz)")
    print(f"  total length L : {period_len}  samples   <- MUST equal len(perturbation.bin)")
    print(f"  active offset  : {pad_front}  samples    (frame starts here)")
    print(f"  active length  : {frame_len}  samples")
    print(f"  frame RMS      : {rms:.6f}   peak |x|: {peak:.6f}")
    print("-" * 64)
    print("Notes for the attacker:")
    print("  * perturbation.bin must be EXACTLY L samples, complex64, @ 5 MHz.")
    print("  * align it sample-for-sample with frame.bin (same indexing); put")
    print("    energy only over the active region unless intentionally otherwise.")
    print("  * normalize to a known norm (e.g. unit RMS over the active region);")
    print("    on-air strength is then set by epsilon in the TX graph.")
    print("=" * 64)


if __name__ == '__main__':
    main()

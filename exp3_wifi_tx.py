#!/usr/bin/env python3
"""
exp3_wifi_tx.py — Experiment 3 legitimate WiFi transmitter (HEADLESS)
─────────────────────────────────────────────────────────────────────

Clean WiFi TX for the Experiment 3 testbed, written from scratch with all
the N-series lessons baked in:

  • N-series addressed by IP (addr=), not USB serial
  • set_time_now() instead of set_time_unknown_pps() — no external PPS needed
  • Explicit antenna selection (SBX: "TX/RX")
  • 2.45 GHz (in range for both SBX and XCVR2450 daughterboards)
  • NO Qt GUI at all — runs fully headless (avoids the libLLVM crash)
  • Network send-buffer note printed at startup

Pipeline (matches the working gr-ieee802-11 TX):
  vector_source → wifi_phy_hier → multiply_const(0.7) → packet_pad2 → usrp_sink
  Frames are injected as PDUs via the IEEE 802.11 MAC on a timer.

Run:
  python3 exp3_wifi_tx.py                 # 30 s default
  python3 exp3_wifi_tx.py --duration 60   # 60 s
  python3 exp3_wifi_tx.py --freq 5.29e9 --antenna J1   # XCVR2450 at 5.29 GHz

To stop early: Ctrl-C.
"""

import sys
import os
import time
import signal
import argparse
import itertools

sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

import pmt
from gnuradio import gr, blocks, uhd, network
from gnuradio.fft import window
import foo
import ieee802_11
from wifi_phy_hier import wifi_phy_hier


# ─── Configuration constants ──────────────────────────────────────────
# Change these here rather than hunting through the code.
TX_ADDR        = "192.168.10.5"   # N-series transmitter IP
CENTER_FREQ    = 2.45e9           # 2.45 GHz (SBX + XCVR2450 both cover this)
SAMP_RATE      = 5e6              # 5 MHz, matches the pipeline
TX_ANTENNA     = "TX/RX"          # SBX antenna name. Use "J1" for XCVR2450.
NORM_GAIN      = 0.75             # normalized TX gain (0..1)
MULT_CONST     = 0.7              # MUST match clean/perturbed for valid OTA (memory: 0.6 vs 0.7 bug)
PDU_LENGTH     = 500              # payload bytes per frame
FRAME_INTERVAL = 300             # ms between frames
ENCODING       = 0               # 0 = BPSK 1/2
LO_OFFSET      = 0
OUT_BUF_SIZE   = 96000


class exp3_wifi_tx(gr.top_block):
    def __init__(self, args):
        gr.top_block.__init__(self, "Exp3 WiFi TX", catch_exceptions=True)

        self.args = args
        self.freq = args.freq
        self.samp_rate = SAMP_RATE
        self.frame_id_gen = itertools.count()

        # ── PHY hierarchical block (clean WiFi OFDM chain) ────────────
        self.wifi_phy_hier_0 = wifi_phy_hier(
            bandwidth=self.samp_rate,
            chan_est=ieee802_11.LS,
            encoding=ieee802_11.Encoding(ENCODING),
            frequency=self.freq,
            sensitivity=0.56,
        )

        # ── USRP sink (N-series) ──────────────────────────────────────
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            args.device,
            uhd.stream_args(cpu_format="fc32", args='', channels=[0]),
            "packet_len",
        )
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        # Internal-clock time set — no external PPS (avoids the N-series crash)
        self.uhd_usrp_sink_0.set_time_now(uhd.time_spec(0.0))
        self.uhd_usrp_sink_0.set_center_freq(
            uhd.tune_request(self.freq,
                             rf_freq=self.freq - LO_OFFSET,
                             rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_sink_0.set_antenna(args.antenna, 0)
        self.uhd_usrp_sink_0.set_normalized_gain(args.gain, 0)

        # ── Readback: confirm what the radio ACTUALLY accepted ────────
        # If a setting silently failed/clamped, the radio "runs" but may
        # not radiate what you think. Print actuals so you can verify.
        try:
            act_freq = self.uhd_usrp_sink_0.get_center_freq(0)
            act_ant = self.uhd_usrp_sink_0.get_antenna(0)
            act_gain = self.uhd_usrp_sink_0.get_normalized_gain(0)
            act_rate = self.uhd_usrp_sink_0.get_samp_rate()
            avail_ant = self.uhd_usrp_sink_0.get_antennas(0)
            print("  --- radio readback (what the USRP actually set) ---")
            print(f"    center freq : {act_freq/1e9:.4f} GHz  (asked {self.freq/1e9:.4f})")
            print(f"    antenna     : {act_ant}  (asked {args.antenna})")
            print(f"    avail ants  : {list(avail_ant)}")
            print(f"    norm gain   : {act_gain:.3f}  (asked {args.gain})")
            print(f"    samp rate   : {act_rate/1e6:.3f} MHz")
            if abs(act_freq - self.freq) > 1e3:
                print("    !! WARNING: actual freq differs from requested — "
                      "board may not cover this frequency.")
            if act_ant != args.antenna:
                print("    !! WARNING: actual antenna differs from requested — "
                      "the requested antenna name may be invalid for this board.")
            print("  ---------------------------------------------------")
        except Exception as e:
            print(f"  (readback failed: {e})")

        # ── MAC + framing ─────────────────────────────────────────────
        self.network_socket_pdu_0 = network.socket_pdu(
            'TCP_SERVER', '', '52001', 10000, False)
        self.ieee802_11_mac_0 = ieee802_11.mac(
            [0x23] * 6, [0x42] * 6, [0xff] * 6)

        self.foo_packet_pad2_0 = foo.packet_pad2(False, True, 0.01, 4000, 1000)
        self.foo_packet_pad2_0.set_min_output_buffer(OUT_BUF_SIZE)

        self.blocks_vector_source_x_0 = blocks.vector_source_c(
            [1 + 0j, 0.998 + 0.063j, 0.992 + 0.125j, 0.982 + 0.187j],
            False, 1, [])
        self.blocks_multiply_const_0 = blocks.multiply_const_cc(MULT_CONST)
        self.blocks_multiply_const_0.set_min_output_buffer(100000)

        # ── Connections ───────────────────────────────────────────────
        self.msg_connect((self.network_socket_pdu_0, 'pdus'),
                         (self.ieee802_11_mac_0, 'app in'))
        self.msg_connect((self.ieee802_11_mac_0, 'phy out'),
                         (self.wifi_phy_hier_0, 'mac_in'))
        self.connect((self.blocks_vector_source_x_0, 0),
                     (self.wifi_phy_hier_0, 0))
        self.connect((self.wifi_phy_hier_0, 0),
                     (self.blocks_multiply_const_0, 0))
        self.connect((self.blocks_multiply_const_0, 0),
                     (self.foo_packet_pad2_0, 0))
        self.connect((self.foo_packet_pad2_0, 0),
                     (self.uhd_usrp_sink_0, 0))

        # ── TX self-tap: save the exact samples sent to the USRP ──────
        # This lets us verify the flowgraph IS producing WiFi signal,
        # independent of whether RF makes it over the air. If this file
        # has high power/activity, the TX flowgraph works and any failure
        # is in the RF/air/decode path, NOT in signal generation.
        if getattr(args, 'tap', None):
            self.blocks_tx_tap = blocks.file_sink(
                gr.sizeof_gr_complex, args.tap, False)
            self.blocks_tx_tap.set_unbuffered(False)
            self.connect((self.foo_packet_pad2_0, 0), (self.blocks_tx_tap, 0))
            print(f"  TX self-tap: saving transmitted samples to {args.tap}")

    # ── Frame generation ──────────────────────────────────────────────
    def send_frame(self):
        frame_id = next(self.frame_id_gen)
        frame_id_str = f"{frame_id:04d}"
        print(f"frame ID: {frame_id_str}")
        payload = 'A' * 20 + frame_id_str + 'A' * (PDU_LENGTH - 24)
        payload_bytes = payload.encode('utf-8')
        pdu = pmt.cons(pmt.PMT_NIL,
                       pmt.init_u8vector(len(payload_bytes),
                                         bytearray(payload_bytes)))
        self.ieee802_11_mac_0.to_basic_block()._post(pmt.intern('app in'), pdu)


def main():
    p = argparse.ArgumentParser(description="Experiment 3 WiFi TX (headless)")
    p.add_argument('--addr', default=None, help='N-series TX IP (used if --device not given)')
    p.add_argument('--device', required=True,
                   help='full UHD device args, e.g. "addr=192.168.10.5" or '
                        '"serial=3259373" for a B205mini. Overrides --addr.')
    p.add_argument('--freq', type=float, default=CENTER_FREQ, help='center freq Hz')
    p.add_argument('--antenna', default=TX_ANTENNA,
                   help='TX antenna: "TX/RX" (SBX/B205mini) or "J1" (XCVR2450)')
    p.add_argument('--duration', type=int, default=30, help='seconds to transmit')
    p.add_argument('--gain', type=float, default=NORM_GAIN,
                   help='normalized TX gain 0..1 (try 1.0 to bridge a weak link)')
    p.add_argument('--tap', default=None,
                   help='save transmitted samples to this file (verifies the '
                        'flowgraph produces signal regardless of RF)')
    args = p.parse_args()

    # Resolve device args: --device wins, else addr=
    if args.device is None and args.addr is None:
        p.error("Must specify --device or --addr")
    if args.device is None:
        args.device = f"addr={args.addr}"

    print("=" * 60)
    print("Experiment 3 — WiFi TX (headless)")
    print("=" * 60)
    print(f"  Device:    {args.device}")
    print(f"  Frequency: {args.freq/1e9:.3f} GHz")
    print(f"  Antenna:   {args.antenna}")
    print(f"  Samp rate: {SAMP_RATE/1e6:.1f} MHz")
    print(f"  Mult const:{MULT_CONST}   Norm gain: {args.gain}")
    print(f"  Duration:  {args.duration} s")
    print("=" * 60)
    print("If you see 'send buffer could not be resized', run on this host:")
    print("  sudo sysctl -w net.core.wmem_max=2500000")
    print("=" * 60)

    tb = exp3_wifi_tx(args)
    tb.start()

    # Frame-sending timer (simple loop, no Qt)
    stop_time = time.time() + args.duration
    interval_s = FRAME_INTERVAL / 1000.0

    def sig_handler(sig=None, frame=None):
        tb.stop(); tb.wait(); sys.exit(0)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    try:
        while time.time() < stop_time:
            tb.send_frame()
            time.sleep(interval_s)
    except KeyboardInterrupt:
        pass

    print("\nDuration reached. Stopping...")
    tb.stop()
    tb.wait()
    print("Done.")


if __name__ == '__main__':
    main()

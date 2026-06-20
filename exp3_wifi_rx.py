#!/usr/bin/env python3
"""
exp3_wifi_rx.py — Experiment 3 receiver (HEADLESS) — v3
─────────────────────────────────────────────────────────

This version replicates the stock gr-ieee802-11 wifi_rx.py receive chain
EXACTLY (the known-good chain), because the raw I/Q proved signal IS
arriving at healthy power (mean 0.056, 76% active) — the earlier failures
were a decode-chain wiring bug, not an RF problem.

Wiring copied verbatim from stock examples/wifi_rx.py:
  src → complex_to_mag_squared → mavg_ff(window+16) → divide(port1)
  src → delay(16) → conjugate → multiply(port1)
  src → multiply(port0)
  multiply → mavg_cc(window) → complex_to_mag → divide(port0)
  mavg_cc → sync_short(port1)
  divide → sync_short(port2)
  delay(16) → sync_short(port0)
  sync_short → delay(sync_length) → sync_long(port1)
  sync_short → sync_long(port0)
  sync_long → stream_to_vector → fft → frame_equalizer → decode_mac

Also taps raw I/Q to raw_iq.bin, and flags YOUR frames (0x41 'A' fill).

Run:
  python3 exp3_wifi_rx.py --device "addr=192.168.10.4" --antenna J2 --freq 2.45e9 --gain 0.75
  python3 exp3_wifi_rx.py --device "serial=3256204" --freq 2.45e9 --antenna J2
"""

import sys
import os
import time
import signal
import argparse

sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

import pmt
from gnuradio import gr, blocks, uhd, fft
from gnuradio.fft import window
import ieee802_11


# Configuration constants
#RX_ADDR     = "192.168.10.4"
CENTER_FREQ = 2.45e9
SAMP_RATE   = 5e6
RX_ANTENNA  = "J2"
NORM_GAIN   = 0.75
LO_OFFSET   = 0
WINDOW_SIZE = 48          # stock GRC 'window_size' variable
SYNC_LENGTH = 320         # stock GRC 'sync_length' variable
CHAN_EST    = 0           # 0 = LS (Equalizer.LS)
EQ_BANDWIDTH = 200000     # stock wifi_rx passes 200000 to the equalizer bandwidth arg


class exp3_wifi_rx(gr.top_block):
    def __init__(self, args):
        gr.top_block.__init__(self, "Exp3 WiFi RX", catch_exceptions=True)

        self.freq = args.freq
        self.samp_rate = SAMP_RATE
        raw_iq_path = os.path.join(args.out, "raw_iq.bin")
        window_size = WINDOW_SIZE
        sync_length = SYNC_LENGTH

        # USRP source
        self.uhd_usrp_source_0 = uhd.usrp_source(
            args.device,
            uhd.stream_args(cpu_format="fc32", args='', channels=[0]),
        )
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_time_now(uhd.time_spec(0.0))
        self.uhd_usrp_source_0.set_center_freq(
            uhd.tune_request(self.freq,
                             rf_freq=self.freq - LO_OFFSET,
                             rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_source_0.set_antenna(args.antenna, 0)
        self.uhd_usrp_source_0.set_normalized_gain(args.gain, 0)

        # Raw I/Q tap
        self.blocks_file_sink_raw = blocks.file_sink(
            gr.sizeof_gr_complex, raw_iq_path, False)
        self.blocks_file_sink_raw.set_unbuffered(False)

        # ── Receive chain blocks (names mirror stock wifi_rx.py) ──────
        self.ieee802_11_sync_short_0 = ieee802_11.sync_short(0.56, 2, False, False)
        self.ieee802_11_sync_long_0 = ieee802_11.sync_long(sync_length, False, False)
        self.ieee802_11_frame_equalizer_0 = ieee802_11.frame_equalizer(
            ieee802_11.Equalizer(CHAN_EST), self.freq, EQ_BANDWIDTH, False, False)
        self.ieee802_11_decode_mac_0 = ieee802_11.decode_mac(True, False)

        self.fft_vxx_0 = fft.fft_vcc(64, True, window.rectangular(64), True, 1)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(
            gr.sizeof_gr_complex, 64)

        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_moving_average_xx_1 = blocks.moving_average_cc(window_size, 1, 4000, 1)
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(window_size + 16, 1, 4000, 1)
        self.blocks_divide_xx_0 = blocks.divide_ff(1)
        self.blocks_delay_0_0 = blocks.delay(gr.sizeof_gr_complex * 1, 16)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex * 1, sync_length)
        self.blocks_conjugate_cc_0 = blocks.conjugate_cc()
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(1)

        # Frame logger (replaces parse_mac/GUI sinks)
        self.frame_logger = _FrameLogger(verbose=getattr(args, 'verbose', False))

        # ── Connections — copied verbatim from stock wifi_rx.py ───────
        src = self.uhd_usrp_source_0

        # raw I/Q tap
        self.connect((src, 0), (self.blocks_file_sink_raw, 0))

        self.connect((self.blocks_complex_to_mag_0, 0), (self.blocks_divide_xx_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_conjugate_cc_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_delay_0, 0), (self.ieee802_11_sync_long_0, 1))
        self.connect((self.blocks_delay_0_0, 0), (self.blocks_conjugate_cc_0, 0))
        self.connect((self.blocks_delay_0_0, 0), (self.ieee802_11_sync_short_0, 0))
        self.connect((self.blocks_divide_xx_0, 0), (self.ieee802_11_sync_short_0, 2))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_divide_xx_0, 1))
        self.connect((self.blocks_moving_average_xx_1, 0), (self.blocks_complex_to_mag_0, 0))
        self.connect((self.blocks_moving_average_xx_1, 0), (self.ieee802_11_sync_short_0, 1))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_moving_average_xx_1, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.ieee802_11_frame_equalizer_0, 0))
        self.connect((self.ieee802_11_frame_equalizer_0, 0), (self.ieee802_11_decode_mac_0, 0))
        self.connect((self.ieee802_11_sync_long_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.ieee802_11_sync_short_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.ieee802_11_sync_short_0, 0), (self.ieee802_11_sync_long_0, 0))
        self.connect((src, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((src, 0), (self.blocks_delay_0_0, 0))
        self.connect((src, 0), (self.blocks_multiply_xx_0, 0))

        # decoded frames → logger
        self.msg_connect((self.ieee802_11_decode_mac_0, 'out'),
                         (self.frame_logger, 'in'))


class _FrameLogger(gr.basic_block):
    # Your TX builds the MAC SDU as: 'A'*20 + frame_id(4 chars) + 'A'*476  (500 B)
    PDU_LENGTH = 500
    FILL = 0x41  # 'A'

    def __init__(self, verbose=False):
        gr.basic_block.__init__(self, name="frame_logger", in_sig=[], out_sig=[])
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self.handle)
        self.total = 0
        self.yours = 0
        self.verbose = verbose
        self.payload_ok = 0       # frames whose payload matched exactly
        self.payload_bad = 0      # yours but payload mismatch (bit errors)
        self.seen_ids = set()

    def expected_payload(self, fid_str):
        return (b'A' * 20 + fid_str.encode('ascii') +
                b'A' * (self.PDU_LENGTH - 24))

    def handle(self, msg):
        self.total += 1
        try:
            vec = pmt.cdr(msg)
            data = bytes(pmt.u8vector_elements(vec))
        except Exception:
            return

        a_count = data.count(self.FILL)
        has_your_dst = bytes([0x42] * 6) in data
        has_your_src = bytes([0x23] * 6) in data
        is_yours = (a_count >= 50) or has_your_dst or has_your_src

        if not is_yours:
            print(f"[ambient] total={self.total} yours={self.yours} len={len(data)}")
            return

        self.yours += 1

        # ── Parse the 802.11 MAC header (first 24 bytes) ──────────────
        # bytes: [0:2]=frame control, [2:4]=duration,
        #        [4:10]=addr1(dst), [10:16]=addr2(src), [16:22]=addr3(bss), [22:24]=seq
        addr1 = data[4:10].hex(':') if len(data) >= 10 else '??'
        addr2 = data[10:16].hex(':') if len(data) >= 16 else '??'

        # ── Locate the SDU payload and the embedded frame id ──────────
        idx = data.find(b'A' * 20)
        fid = '????'
        payload = b''
        if idx >= 0 and len(data) >= idx + 24:
            fid = data[idx + 20:idx + 24].decode('ascii', 'replace')
            # the SDU runs from idx to idx + PDU_LENGTH (clamped to frame)
            payload = data[idx: idx + self.PDU_LENGTH]

        # ── Byte-exact verification against what the TX sent ──────────
        verdict = "?"
        if fid.isdigit():
            exp = self.expected_payload(fid)
            got = payload
            # compare on the overlapping length actually received
            n = min(len(exp), len(got))
            if n > 0:
                mismatches = sum(1 for i in range(n) if exp[i] != got[i])
                if mismatches == 0 and n >= self.PDU_LENGTH - 8:
                    verdict = "EXACT MATCH"
                    self.payload_ok += 1
                else:
                    verdict = f"{mismatches} byte mismatches / {n}"
                    self.payload_bad += 1
            dup = " (DUP id)" if fid in self.seen_ids else ""
            self.seen_ids.add(fid)
        else:
            dup = ""

        print(f"[YOURS] #{self.yours} id={fid} len={len(data)} "
              f"A={a_count} dst={addr1} src={addr2} -> payload: {verdict}{dup}")

        if self.verbose:
            # show a readable snippet: first 32 payload bytes + the id region
            head = payload[:32]
            print(f"         payload[0:32] ascii: {head.decode('ascii','replace')!r}")
            print(f"         payload[0:32] hex  : {head.hex(' ')}")


def main():
    p = argparse.ArgumentParser(description="Experiment 3 WiFi RX (headless v3, stock chain)")
    p.add_argument('--addr', default=None)
    p.add_argument('--device', required=True,
                   help='UHD device args, e.g. "addr=192.168.10.4" or "serial=3256204"')
    p.add_argument('--freq', type=float, default=CENTER_FREQ)
    p.add_argument('--antenna', default=RX_ANTENNA, help='"J1" or "J2" (USRP2); use "RX2" for B2xx')
    p.add_argument('--duration', type=int, default=30)
    p.add_argument('--gain', type=float, default=NORM_GAIN)
    p.add_argument('--out', default='./capture')
    p.add_argument('--verbose', action='store_true',
                   help='print payload ascii/hex snippet for each of your frames')
    args = p.parse_args()

    if args.device is None and args.addr is None:
        p.error("Must specify either --addr or --device")
        sys.exit(1)
    if args.device is None:
        args.device = f"addr={args.addr}"
    os.makedirs(args.out, exist_ok=True)

    print("=" * 60)
    print("Experiment 3 - WiFi RX (headless v3 - stock decode chain)")
    print("=" * 60)
    print(f"  Device:    {args.device}")
    print(f"  Frequency: {args.freq/1e9:.3f} GHz")
    print(f"  Antenna:   {args.antenna}    Gain: {args.gain}")
    print(f"  Duration:  {args.duration} s    Saving: {args.out}/raw_iq.bin")
    print("=" * 60)
    print("Watch for [YOURS] - frames from exp3_wifi_tx (MAC 23:23/42:42).")
    print("=" * 60)

    tb = exp3_wifi_rx(args)

    def sig_handler(sig=None, frame=None):
        tb.stop(); tb.wait(); sys.exit(0)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    print(f"[RX] Capturing for {args.duration}s ...")
    time.sleep(args.duration)
    tb.stop()
    tb.wait()

    lg = tb.frame_logger
    print("\n" + "=" * 60)
    print(f"Decoded {lg.total} frames total; {lg.yours} were YOURS.")
    print(f"Payload check:  EXACT match: {lg.payload_ok}   "
          f"with bit errors: {lg.payload_bad}")
    if lg.yours > 0:
        uniq = len([f for f in lg.seen_ids if f.isdigit()])
        print(f"Unique frame IDs received: {uniq}")
        if lg.payload_bad == 0 and lg.payload_ok > 0:
            print("SUCCESS - every received payload matched the TX byte-for-byte.")
        elif lg.payload_ok > 0:
            print("Link works; some frames had bit errors (see mismatch counts).")
        else:
            print("Frames detected but payload verification couldn't confirm bytes.")
    else:
        print("0 of yours (raw I/Q still saved).")
    print(f"Raw I/Q: {os.path.join(args.out, 'raw_iq.bin')}")
    print("=" * 60)


if __name__ == '__main__':
    main()

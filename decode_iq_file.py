#!/usr/bin/env python3
"""
decode_iq_file.py — offline 802.11 decoder for a captured I/Q file.

Runs the EXACT gr-ieee802-11 receive chain from exp3_wifi_rx.py, but with a
blocks.file_source in place of the USRP source. Used to prove that a TX-side
tap (exp3_wifi_tx.py --tap) contains decodable WiFi frames — a one-radio
loopback that needs no RF.

Run:
  python3 decode_iq_file.py tx_tap.bin --freq 2.45e9
"""
import sys, os, argparse
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

from gnuradio import gr, blocks, fft
from gnuradio.fft import window
import ieee802_11

# reuse the frame logger + constants from the receiver
from exp3_wifi_rx import _FrameLogger, WINDOW_SIZE, SYNC_LENGTH, CHAN_EST, EQ_BANDWIDTH


class decode_iq_file(gr.top_block):
    def __init__(self, path, freq, verbose=False):
        gr.top_block.__init__(self, "Decode IQ File", catch_exceptions=True)
        window_size = WINDOW_SIZE
        sync_length = SYNC_LENGTH

        # File source (non-repeat) replaces the USRP source
        src = blocks.file_source(gr.sizeof_gr_complex, path, False)
        self.head = src  # keep ref

        self.sync_short = ieee802_11.sync_short(0.56, 2, False, False)
        self.sync_long = ieee802_11.sync_long(sync_length, False, False)
        self.equalizer = ieee802_11.frame_equalizer(
            ieee802_11.Equalizer(CHAN_EST), freq, EQ_BANDWIDTH, False, False)
        self.decode_mac = ieee802_11.decode_mac(True, False)

        fft_v = fft.fft_vcc(64, True, window.rectangular(64), True, 1)
        s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, 64)
        mult = blocks.multiply_vcc(1)
        mavg_c = blocks.moving_average_cc(window_size, 1, 4000, 1)
        mavg_f = blocks.moving_average_ff(window_size + 16, 1, 4000, 1)
        div = blocks.divide_ff(1)
        delay_a = blocks.delay(gr.sizeof_gr_complex * 1, 16)
        delay_b = blocks.delay(gr.sizeof_gr_complex * 1, sync_length)
        conj = blocks.conjugate_cc()
        c2ms = blocks.complex_to_mag_squared(1)
        c2m = blocks.complex_to_mag(1)

        self.frame_logger = _FrameLogger(verbose=verbose)

        # ── identical wiring to exp3_wifi_rx.py ───────────────────────
        self.connect((c2m, 0), (div, 0))
        self.connect((c2ms, 0), (mavg_f, 0))
        self.connect((conj, 0), (mult, 1))
        self.connect((delay_b, 0), (self.sync_long, 1))
        self.connect((delay_a, 0), (conj, 0))
        self.connect((delay_a, 0), (self.sync_short, 0))
        self.connect((div, 0), (self.sync_short, 2))
        self.connect((mavg_f, 0), (div, 1))
        self.connect((mavg_c, 0), (c2m, 0))
        self.connect((mavg_c, 0), (self.sync_short, 1))
        self.connect((mult, 0), (mavg_c, 0))
        self.connect((s2v, 0), (fft_v, 0))
        self.connect((fft_v, 0), (self.equalizer, 0))
        self.connect((self.equalizer, 0), (self.decode_mac, 0))
        self.connect((self.sync_long, 0), (s2v, 0))
        self.connect((self.sync_short, 0), (delay_b, 0))
        self.connect((self.sync_short, 0), (self.sync_long, 0))
        self.connect((src, 0), (c2ms, 0))
        self.connect((src, 0), (delay_a, 0))
        self.connect((src, 0), (mult, 0))

        self.msg_connect((self.decode_mac, 'out'), (self.frame_logger, 'in'))


def main():
    p = argparse.ArgumentParser(description="Offline 802.11 decode of an I/Q file")
    p.add_argument('path')
    p.add_argument('--freq', type=float, default=2.45e9)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    print(f"Decoding {args.path} ...")
    tb = decode_iq_file(args.path, args.freq, args.verbose)
    tb.run()   # runs to EOF (file_source, non-repeat)

    lg = tb.frame_logger
    print("\n" + "=" * 60)
    print(f"Decoded {lg.total} frames total; {lg.yours} were YOURS.")
    print(f"Payload check:  EXACT match: {lg.payload_ok}   "
          f"with bit errors: {lg.payload_bad}")
    if lg.yours and lg.payload_ok and not lg.payload_bad:
        print("SUCCESS — TX tap decodes to byte-exact WiFi frames (loopback OK).")
    elif lg.yours:
        print("Frames decoded; some payloads imperfect (see counts).")
    else:
        print("No YOURS frames decoded from this file.")
    print("=" * 60)


if __name__ == '__main__':
    main()

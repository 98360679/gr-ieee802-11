#!/usr/bin/env python3
"""
exp3_decode_file.py — OFFLINE decode of a complex64 IQ file through the stock
gr-ieee802-11 receive chain (NO radio, NO channel).

Purpose: settle whether a waveform (e.g. /dev/shm/frame.bin) contains a VALID,
decodable 802.11 frame. If it decodes here, the frame is good and any OTA
failure is an RF/channel issue; if it does NOT decode here, the frame itself
is corrupt (generation/capture problem).

Reuses the EXACT chain + payload check from exp3_wifi_rx.py (single source of
truth) — only the signal source is swapped to a file_source.

Run:
  .venv/bin/python exp3_decode_file.py --file /dev/shm/frame.bin --loops 40
"""

import argparse
from gnuradio import gr, blocks, fft
from gnuradio.fft import window
import ieee802_11

import pmt
from exp3_wifi_rx import (_FrameLogger, SAMP_RATE, WINDOW_SIZE, SYNC_LENGTH,
                          CHAN_EST, EQ_BANDWIDTH)


class _RawDump(gr.basic_block):
    """Dump raw decoded MAC bytes from decode_mac, bypassing any parser."""
    def __init__(self, max_dump=3):
        gr.basic_block.__init__(self, "raw_dump", in_sig=None, out_sig=None)
        self.message_port_register_in(pmt.intern('in'))
        self.set_msg_handler(pmt.intern('in'), self.handle)
        self.n = 0
        self.max_dump = max_dump
        self.frames = 0
        self.a_counts = []

    def handle(self, msg):
        self.frames += 1
        car = pmt.car(msg); cdr = pmt.cdr(msg)
        if not pmt.is_u8vector(cdr):
            return
        data = bytes(pmt.u8vector_elements(cdr))
        a_count = data.count(0x41)          # 'A'
        self.a_counts.append(a_count)
        if self.n < self.max_dump:
            self.n += 1
            print(f"  [frame] {len(data)} bytes, 0x41('A') count={a_count}")
            print(f"    hex[0:40] : {data[:40].hex(' ')}")
            # payload region: after 24-byte MAC header
            pl = data[24:24+32]
            print(f"    payload[0:32] ascii: {pl.decode('ascii','replace')!r}")


class decode_file(gr.top_block):
    def __init__(self, path, freq, loops, verbose):
        gr.top_block.__init__(self, "Exp3 offline file decode")
        window_size = WINDOW_SIZE
        sync_length = SYNC_LENGTH

        # count samples in the file, cap total at loops*len so run() terminates
        import os
        nsamp = os.path.getsize(path) // 8
        total = nsamp * max(1, loops)

        self.src = blocks.file_source(gr.sizeof_gr_complex, path, True)  # repeat
        self.head = blocks.head(gr.sizeof_gr_complex, total)

        # ── stock gr-ieee802-11 RX chain (verbatim from exp3_wifi_rx) ──
        self.sync_short = ieee802_11.sync_short(0.56, 2, False, False)
        self.sync_long = ieee802_11.sync_long(sync_length, False, False)
        self.frame_eq = ieee802_11.frame_equalizer(
            ieee802_11.Equalizer(CHAN_EST), freq, EQ_BANDWIDTH, False, False)
        self.decode_mac = ieee802_11.decode_mac(True, False)
        self.fft_vxx = fft.fft_vcc(64, True, window.rectangular(64), True, 1)
        self.s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, 64)
        self.mult = blocks.multiply_vcc(1)
        self.mavg_c = blocks.moving_average_cc(window_size, 1, 4000, 1)
        self.mavg_f = blocks.moving_average_ff(window_size + 16, 1, 4000, 1)
        self.divide = blocks.divide_ff(1)
        self.delay_long = blocks.delay(gr.sizeof_gr_complex, sync_length)
        self.delay16 = blocks.delay(gr.sizeof_gr_complex, 16)
        self.conj = blocks.conjugate_cc()
        self.mag2 = blocks.complex_to_mag_squared(1)
        self.mag = blocks.complex_to_mag(1)
        self.logger = _RawDump(max_dump=3)

        self.connect((self.src, 0), (self.head, 0))
        src = self.head
        self.connect((self.mag, 0), (self.divide, 0))
        self.connect((self.mag2, 0), (self.mavg_f, 0))
        self.connect((self.conj, 0), (self.mult, 1))
        self.connect((self.delay_long, 0), (self.sync_long, 1))
        self.connect((self.delay16, 0), (self.conj, 0))
        self.connect((self.delay16, 0), (self.sync_short, 0))
        self.connect((self.divide, 0), (self.sync_short, 2))
        self.connect((self.mavg_f, 0), (self.divide, 1))
        self.connect((self.mavg_c, 0), (self.mag, 0))
        self.connect((self.mavg_c, 0), (self.sync_short, 1))
        self.connect((self.mult, 0), (self.mavg_c, 0))
        self.connect((self.s2v, 0), (self.fft_vxx, 0))
        self.connect((self.fft_vxx, 0), (self.frame_eq, 0))
        self.connect((self.frame_eq, 0), (self.decode_mac, 0))
        self.connect((self.sync_long, 0), (self.s2v, 0))
        self.connect((self.sync_short, 0), (self.delay_long, 0))
        self.connect((self.sync_short, 0), (self.sync_long, 0))
        self.connect((src, 0), (self.mag2, 0))
        self.connect((src, 0), (self.delay16, 0))
        self.connect((src, 0), (self.mult, 0))
        self.msg_connect((self.decode_mac, 'out'), (self.logger, 'in'))


def main():
    p = argparse.ArgumentParser(description="Offline file decode (gr-ieee802-11)")
    p.add_argument('--file', default='/dev/shm/frame.bin')
    p.add_argument('--freq', type=float, default=2.452e9)
    p.add_argument('--loops', type=int, default=40,
                   help='how many times to loop the file through the decoder')
    p.add_argument('--verbose', action='store_true')
    a = p.parse_args()

    print(f"Offline decode: {a.file}  (loops={a.loops})")
    tb = decode_file(a.file, a.freq, a.loops, a.verbose)
    tb.run()
    lg = tb.logger
    print("-" * 56)
    print(f"frames decoded (decode_mac emitted) : {lg.frames}")
    if lg.a_counts:
        import statistics
        print(f"  0x41('A') bytes per frame: min={min(lg.a_counts)} "
              f"max={max(lg.a_counts)} median={int(statistics.median(lg.a_counts))}")
        # FIXED_PAYLOAD is 'EXP3' + 'A'*496 -> ~496 'A' bytes if payload is clean
        if max(lg.a_counts) >= 480:
            print("VERDICT: VALID — payload decodes to the EXP3 'A'-fill offline. "
                  "Any OTA failure is RF/channel, not the waveform.")
        elif max(lg.a_counts) > 0:
            print("VERDICT: partial — header+some payload decode, but 'A'-fill is "
                  "incomplete offline (waveform/generation issue).")
        else:
            print("VERDICT: payload has ZERO 'A' bytes offline — the data field is "
                  "corrupt even with no channel (generation problem).")
    else:
        print("VERDICT: nothing decoded.")


if __name__ == '__main__':
    main()

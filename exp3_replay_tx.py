#!/usr/bin/env python3
"""Stream a captured WiFi baseband file CONTINUOUSLY out the N210.

Rationale: the N210 radiates continuous streams fine (proven by the CW tone)
but drops short tagged BURSTS over GigE. This bypasses burst mode entirely:
file_source (repeat) -> usrp_sink with NO length tag = continuous streaming.
The file already contains real OFDM frames (preambles intact), so a mini RX
should decode them.

Use the tap from a TX run as the input file:
  python3 exp3_replay_tx.py --addr 192.168.10.5 --freq 3.3e9 \
        --antenna TX/RX --gain 1.0 --file /tmp/tx_samples.bin --duration 90
"""
import sys, time, signal, argparse
from gnuradio import gr, blocks, uhd

SAMP_RATE = 5e6


class replay_tx(gr.top_block):
    def __init__(self, a):
        gr.top_block.__init__(self, "Exp3 Replay TX", catch_exceptions=True)
        # Continuous USRP sink: NOTE no "packet_len" tag -> no burst mode.
        self.usrp = uhd.usrp_sink(
            f"addr={a.addr}",
            uhd.stream_args(cpu_format="fc32", args='', channels=[0]),
        )
        self.usrp.set_samp_rate(SAMP_RATE)
        self.usrp.set_time_now(uhd.time_spec(0.0))
        self.usrp.set_center_freq(uhd.tune_request(a.freq), 0)
        self.usrp.set_antenna(a.antenna, 0)
        self.usrp.set_normalized_gain(a.gain, 0)

        # Loop the captured baseband forever -> continuous stream.
        self.src = blocks.file_source(gr.sizeof_gr_complex, a.file, True)
        self.src.set_min_output_buffer(100000)
        self.connect(self.src, self.usrp)

        # readback
        print("  --- radio readback ---")
        print(f"    freq : {self.usrp.get_center_freq(0)/1e9:.4f} GHz")
        print(f"    ant  : {self.usrp.get_antenna(0)}  avail {self.usrp.get_antennas(0)}")
        print(f"    rate : {self.usrp.get_samp_rate()/1e6:.3f} MHz")
        print("  ----------------------")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--addr', default='192.168.10.5')
    p.add_argument('--freq', type=float, default=3.3e9)
    p.add_argument('--antenna', default='TX/RX')
    p.add_argument('--gain', type=float, default=1.0)
    p.add_argument('--file', default='/tmp/tx_samples.bin')
    p.add_argument('--duration', type=int, default=90)
    a = p.parse_args()

    print("=" * 50)
    print("Exp3 Replay TX (continuous, no burst)")
    print(f"  file={a.file}  freq={a.freq/1e9:.3f}GHz  ant={a.antenna}  gain={a.gain}")
    print("=" * 50)

    tb = replay_tx(a)
    def stop(*_): tb.stop(); tb.wait(); sys.exit(0)
    signal.signal(signal.SIGINT, stop)
    tb.start()
    print(f"[TX] continuous replay for {a.duration}s ...")
    time.sleep(a.duration)
    tb.stop(); tb.wait()
    print("Done.")


if __name__ == '__main__':
    main()

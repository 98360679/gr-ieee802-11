#!/usr/bin/env python3
"""
live_power.py — real-time RX signal-power meter (no decode, no file).

Streams from the USRP and prints mean power several times per second so you can
SEE whether energy is arriving while you key the transmitter. Use this to
coordinate the TX and to tell "RF is arriving" from "nothing on the air".

Implementation note: the power is computed by built-in GNU Radio blocks and
read from the main thread via probe_signal_f. (An earlier version used a custom
Python sink block in the stream, which segfaulted at 5 MS/s in GR 3.10.)

Run:
  python3 live_power.py --device "addr=192.168.10.4"            # J2, 2.45 GHz
  python3 live_power.py --device "addr=192.168.10.4" --gain 0.4 # lower gain if saturating
Ctrl-C to stop.
"""
import sys, time, signal, argparse, math
from gnuradio import gr, uhd, blocks

SAMP_RATE = 5e6


class live_power(gr.top_block):
    def __init__(self, args):
        gr.top_block.__init__(self, "Live Power", catch_exceptions=True)
        win = max(1, int(SAMP_RATE * args.report))

        src = uhd.usrp_source(
            args.device, uhd.stream_args(cpu_format="fc32", args='', channels=[0]))
        src.set_samp_rate(SAMP_RATE)
        src.set_time_now(uhd.time_spec(0.0))
        src.set_center_freq(uhd.tune_request(args.freq), 0)
        src.set_antenna(args.antenna, 0)
        src.set_normalized_gain(args.gain, 0)

        # mean power over the report window, sampled by the main thread
        mag2 = blocks.complex_to_mag_squared(1)
        avg = blocks.moving_average_ff(win, 1.0 / win, 4000, 1)
        self.probe_mean = blocks.probe_signal_f()
        self.connect(src, mag2, avg, self.probe_mean)

        print(f"  device={args.device}  freq={args.freq/1e9:.3f} GHz  "
              f"ant={src.get_antenna(0)}  gain={src.get_normalized_gain(0):.3f}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Live RX power meter")
    p.add_argument('--device', required=True, help='UHD args, e.g. "addr=192.168.10.4"')
    p.add_argument('--freq', type=float, default=2.45e9)
    p.add_argument('--antenna', default="J2")
    p.add_argument('--gain', type=float, default=0.75)
    p.add_argument('--report', type=float, default=0.2, help='seconds between prints')
    p.add_argument('--duration', type=int, default=0, help='0 = run until Ctrl-C')
    args = p.parse_args()

    tb = live_power(args)

    def stop(*_):
        tb.stop(); tb.wait(); print("\nstopped."); sys.exit(0)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print("Watching RX power — key the TX and watch the bar move. Ctrl-C to stop.", flush=True)
    tb.start()
    t0 = time.time()
    try:
        while True:
            time.sleep(args.report)
            mean_p = tb.probe_mean.level()
            mean_db = 10 * math.log10(mean_p + 1e-12)
            frac = min(1.0, max(0.0, (mean_db + 60) / 60))   # -60 dB empty .. 0 dB full
            bar = "#" * int(frac * 30)
            # normalized power near 1.0 (0 dB) means the front-end is clipping
            sat = "  <<< SATURATING (lower --gain)" if mean_p > 0.5 else ""
            print(f"[{time.time()-t0:6.1f}s] mean {mean_db:7.2f} dB  "
                  f"(P={mean_p:.4f})  |{bar:<30}|{sat}", flush=True)
            if args.duration and time.time() - t0 >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    tb.stop(); tb.wait()


if __name__ == '__main__':
    main()

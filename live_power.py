#!/usr/bin/env python3
"""
live_power.py — real-time RX signal-power meter (no decode, no file).

Streams from the USRP and prints mean/peak power several times per second so
you can SEE whether energy is arriving while you key the transmitter. Use this
to coordinate the TX and to tell "RF is arriving" from "nothing on the air".

Run:
  python3 live_power.py --device "addr=192.168.10.4"            # J2, 2.45 GHz
  python3 live_power.py --device "addr=192.168.10.4" --gain 0.4 # lower gain if saturating
Ctrl-C to stop.
"""
import sys, os, time, signal, argparse, math
import numpy as np
from gnuradio import gr, uhd

SAMP_RATE = 5e6


class PowerMeter(gr.sync_block):
    """Sink that prints mean/peak power every `report_s` seconds."""
    def __init__(self, report_s=0.2, samp_rate=SAMP_RATE):
        gr.sync_block.__init__(self, "power_meter", in_sig=[np.complex64], out_sig=[])
        self.report_n = int(samp_rate * report_s)
        self.acc_sumsq = 0.0
        self.acc_peak = 0.0
        self.acc_n = 0
        self.t0 = None

    def work(self, input_items, output_items):
        x = input_items[0]
        p = np.abs(x) ** 2
        self.acc_sumsq += float(p.sum())
        self.acc_peak = max(self.acc_peak, float(np.sqrt(p.max())) if len(p) else 0.0)
        self.acc_n += len(x)
        if self.acc_n >= self.report_n:
            if self.t0 is None:
                self.t0 = time.time()
            mean_p = self.acc_sumsq / self.acc_n
            mean_db = 10 * math.log10(mean_p + 1e-12)
            # 30-char bar scaled from -60 dB (empty) to 0 dB (full)
            frac = min(1.0, max(0.0, (mean_db + 60) / 60))
            bar = "#" * int(frac * 30)
            sat = "  <<< SATURATING" if self.acc_peak > 0.98 else ""
            print(f"[{time.time()-self.t0:6.1f}s] mean {mean_db:7.2f} dB  "
                  f"peak |x| {self.acc_peak:5.3f}  |{bar:<30}|{sat}", flush=True)
            self.acc_sumsq = 0.0; self.acc_peak = 0.0; self.acc_n = 0
        return len(x)


class live_power(gr.top_block):
    def __init__(self, args):
        gr.top_block.__init__(self, "Live Power", catch_exceptions=True)
        src = uhd.usrp_source(
            args.device, uhd.stream_args(cpu_format="fc32", args='', channels=[0]))
        src.set_samp_rate(SAMP_RATE)
        src.set_time_now(uhd.time_spec(0.0))
        src.set_center_freq(uhd.tune_request(args.freq), 0)
        src.set_antenna(args.antenna, 0)
        src.set_normalized_gain(args.gain, 0)
        print(f"  device={args.device}  freq={args.freq/1e9:.3f} GHz  "
              f"ant={src.get_antenna(0)}  gain={src.get_normalized_gain(0):.3f}", flush=True)
        self.connect(src, PowerMeter(args.report))


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
    if args.duration > 0:
        time.sleep(args.duration); tb.stop(); tb.wait()
    else:
        tb.wait()


if __name__ == '__main__':
    main()

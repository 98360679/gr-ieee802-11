#!/usr/bin/env python3
"""Transmit a CW tone from a USRP for N seconds (non-interactive, no stdin).
Used to test a radio's TX path. Run:
  python3 tone_tx.py --device addr=192.168.10.3 --ant J2 --freq 2.45e9 --offset 500e3 --duration 12
"""
import time, signal, argparse
from gnuradio import gr, uhd, analog

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', required=True)
    p.add_argument('--freq', type=float, default=2.45e9)
    p.add_argument('--offset', type=float, default=500e3, help='tone offset from center')
    p.add_argument('--ant', default='J2')
    p.add_argument('--rate', type=float, default=5e6)
    p.add_argument('--gain', type=float, default=0.8, help='normalized 0..1')
    p.add_argument('--ampl', type=float, default=0.7)
    p.add_argument('--duration', type=int, default=12)
    a = p.parse_args()

    tb = gr.top_block("tone_tx", catch_exceptions=True)
    src = analog.sig_source_c(a.rate, analog.GR_COS_WAVE, a.offset, a.ampl, 0, 0)
    sink = uhd.usrp_sink(a.device, uhd.stream_args(cpu_format="fc32", channels=[0]))
    sink.set_samp_rate(a.rate)
    sink.set_center_freq(a.freq, 0)
    sink.set_antenna(a.ant, 0)
    sink.set_normalized_gain(a.gain, 0)
    tb.connect(src, sink)
    print(f"TX tone: {a.device} ant={sink.get_antenna(0)} "
          f"freq={a.freq/1e9:.4f}GHz +{a.offset/1e3:.0f}kHz "
          f"gain={sink.get_normalized_gain(0):.2f} for {a.duration}s", flush=True)

    signal.signal(signal.SIGTERM, lambda *_: (tb.stop(), tb.wait()))
    tb.start()
    time.sleep(a.duration)
    tb.stop(); tb.wait()
    print("tone done", flush=True)

if __name__ == '__main__':
    main()

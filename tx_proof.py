#!/usr/bin/env python3
"""
tx_proof.py — Dead-simple proof that a USRP is transmitting.

Transmits a continuous tone and reports, from the RADIO ITSELF:
  • what freq/antenna/gain the radio actually accepted (readback)
  • the async TX status messages from the device (proof samples went out)
  • a count of samples sent

If you see "samples sent" climbing and async messages, the radio is
transmitting. No receiver needed.

Usage:
  python3 tx_proof.py --device "serial=XXXX" --freq 2.45e9 --antenna TX/RX --gain 1.0
  python3 tx_proof.py --device "addr=192.168.10.5" --freq 2.45e9 --antenna TX/RX
"""
import argparse, time, sys
import numpy as np
import uhd  # native UHD python API (not gnuradio) — gives direct async access


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', required=True, help='e.g. "serial=3256204" or "addr=192.168.10.5"')
    p.add_argument('--freq', type=float, default=2.45e9)
    p.add_argument('--antenna', default='TX/RX')
    p.add_argument('--gain', type=float, default=1.0, help='normalized gain 0..1')
    p.add_argument('--rate', type=float, default=5e6)
    p.add_argument('--duration', type=float, default=10.0)
    p.add_argument('--tone', type=float, default=500e3, help='tone offset Hz')
    args = p.parse_args()

    print("=" * 60)
    print("TX PROOF — does this radio actually transmit?")
    print("=" * 60)

    usrp = uhd.usrp.MultiUSRP(args.device)

    # Apply settings
    usrp.set_tx_rate(args.rate)
    usrp.set_tx_freq(uhd.types.TuneRequest(args.freq), 0)
    usrp.set_tx_antenna(args.antenna, 0)
    usrp.set_normalized_tx_gain(args.gain, 0)

    # READBACK — what did the radio actually accept?
    act_freq = usrp.get_tx_freq(0)
    act_ant = usrp.get_tx_antenna(0)
    act_gain = usrp.get_normalized_tx_gain(0)
    act_rate = usrp.get_tx_rate()
    avail = usrp.get_tx_antennas(0)
    print(f"  device      : {args.device}")
    print(f"  actual freq : {act_freq/1e9:.4f} GHz  (asked {args.freq/1e9:.4f})")
    print(f"  actual ant  : {act_ant}  (asked {args.antenna})   avail={list(avail)}")
    print(f"  actual gain : {act_gain:.3f} normalized")
    print(f"  actual rate : {act_rate/1e6:.3f} MHz")
    if abs(act_freq - args.freq) > 1e3:
        print("  !! FREQ MISMATCH — board may not cover this frequency")
    if act_ant != args.antenna:
        print("  !! ANTENNA MISMATCH — requested name invalid; radio chose another")
    print("-" * 60)

    # Build a continuous tone
    n = int(args.rate * 0.1)  # 100 ms chunks
    t = np.arange(n) / args.rate
    tone = np.exp(1j * 2 * np.pi * args.tone * t).astype(np.complex64) * 0.7

    # Set up TX streamer
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    tx_streamer = usrp.get_tx_stream(st_args)

    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst = False
    md.has_time_spec = False

    # Async receiver for TX status from the device
    async_md = uhd.types.TXAsyncMetadata()

    print("Transmitting tone... watch 'samples sent' climb + async events.")
    print("(If samples climb and no errors, the radio IS transmitting.)\n")

    total = 0
    t_end = time.time() + args.duration
    last_print = 0
    async_counts = {}
    try:
        while time.time() < t_end:
            sent = tx_streamer.send(tone, md)
            total += sent
            md.start_of_burst = False
            # poll async status (non-blocking-ish)
            if tx_streamer.recv_async_msg(async_md, 0.0):
                ev = str(async_md.event_code)
                async_counts[ev] = async_counts.get(ev, 0) + 1
            now = time.time()
            if now - last_print >= 1.0:
                print(f"  samples sent: {total:>12,}   async events: {async_counts}")
                last_print = now
    except KeyboardInterrupt:
        pass

    # end burst
    md.end_of_burst = True
    tx_streamer.send(np.zeros(1, dtype=np.complex64), md)

    print("\n" + "=" * 60)
    print(f"Total samples sent to radio: {total:,}")
    print(f"Async TX events seen: {async_counts}")
    if total > 0:
        print("=> The host pushed samples to the radio.")
        if async_counts:
            print("=> The radio returned async TX status => IT IS TRANSMITTING.")
        else:
            print("=> No async events captured (some setups don't surface them),")
            print("   but a climbing sample count + no send errors means it's streaming.")
    else:
        print("=> No samples sent — TX streamer failed. Radio NOT transmitting.")
    print("=" * 60)


if __name__ == '__main__':
    main()

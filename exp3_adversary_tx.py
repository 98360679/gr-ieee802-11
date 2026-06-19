#!/usr/bin/env python3
"""
exp3_adversary_tx.py — per-frame adversarial OTA transmitter (HEADLESS)
───────────────────────────────────────────────────────────────────────

Experiment 3, per-frame channel-aware attack, transmit side.

Design rationale (built from the lessons already in this repo):
  * The N-series DROPS short tagged bursts over GigE (see exp3_replay_tx.py),
    so we DO NOT use burst mode. Both channels are CONTINUOUS file loops.
  * For a single precomputed per-frame perturbation to stay aligned with the
    frame forever, frame.bin and perturbation.bin are the SAME length L and are
    looped together. A 2-channel USRP sink pulls both ports sample-synchronously,
    so identical-length loops stay locked sample-for-sample for the whole run.
  * The two motherboards are time-synced (MIMO) and given a common stream
    start_time, so ch0 (frame) and ch1 (perturbation) hit the air simultaneously.

Signal:
  frame.bin (loop) ──────────────► USRP ch0  (legit WiFi, radio @ .3)
  pert.bin  (loop) ─► *epsilon ───► USRP ch1  (adversary, radio @ .4)

frame.bin already has MULT_CONST (0.7) baked in by exp3_make_frame.py, so the
two scripts stay in lock-step for a valid clean-vs-perturbed comparison.

Both files MUST be the same length L (run exp3_make_frame.py to get L and to
generate frame.bin; the attacker produces perturbation.bin of length L).

Run (perturbed):
  python3 exp3_adversary_tx.py --frame /dev/shm/frame.bin \
        --pert /dev/shm/perturbation.bin --epsilon 0.05 --duration 60
Run (clean baseline, ch1 silent):
  python3 exp3_adversary_tx.py --frame /dev/shm/frame.bin --clean --duration 60
"""

import sys
import os
import time
import signal
import argparse

import numpy as np
from gnuradio import gr, blocks, uhd


SAMP_RATE = 5e6
LO_OFFSET = 0.0

# Two-motherboard MIMO rig (matches wifi_adversary_tx.grc).
ADDR_LEGIT = "192.168.10.3"   # ch0 — legitimate WiFi frame
ADDR_ADV   = "192.168.10.4"   # ch1 — adversarial perturbation


class adversary_tx(gr.top_block):
    def __init__(self, a, L):
        gr.top_block.__init__(self, "Exp3 Adversary TX", catch_exceptions=True)
        self.a = a

        # ── 2-channel / 2-mboard USRP sink, CONTINUOUS (no length tag) ──
        dev = f"addr0={ADDR_LEGIT},addr1={ADDR_ADV}"
        self.usrp = uhd.usrp_sink(
            dev,
            uhd.stream_args(cpu_format="fc32", args='', channels=[0, 1]),
            # NOTE: no length-tag name -> continuous streaming, not burst mode.
        )
        self.usrp.set_samp_rate(SAMP_RATE)

        # Per-channel RF. mboard0 = legit (ch0), mboard1 = adversary (ch1).
        for ch in (0, 1):
            self.usrp.set_center_freq(
                uhd.tune_request(a.freq, rf_freq=a.freq - LO_OFFSET,
                                 rf_freq_policy=uhd.tune_request.POLICY_MANUAL), ch)
            self.usrp.set_antenna(a.antenna, ch)
        self.usrp.set_gain(a.tx_gain, 0)    # ch0 legit, ABSOLUTE dB (range 0..35)
        self.usrp.set_gain(a.adv_gain, 1)   # ch1 adversary, ABSOLUTE dB (range 0..35)

        # ── MIMO time sync + common synchronized start ──────────────────
        # mboard0 is the master (internal), mboard1 slaves off the MIMO cable.
        try:
            self.usrp.set_clock_source("internal", 0)
            self.usrp.set_time_source("none", 0)   # mb0 master: 'internal' is NOT a valid
                                                   # time source (only clock); 'none' is.
            self.usrp.set_clock_source("mimo", 1)  # mb1 slave: clock+time off the MIMO cable
            self.usrp.set_time_source("mimo", 1)
        except Exception as e:
            print(f"  (clock/time source setup note: {e})")
        time.sleep(1.0)                       # let the MIMO link settle
        self.usrp.set_time_now(uhd.time_spec(0.0))
        time.sleep(0.2)
        t_start = self.usrp.get_time_now() + uhd.time_spec(3.0)
        self.usrp.set_start_time(t_start)
        print(f"  synchronized start scheduled at t={t_start.get_real_secs():.3f}s")

        # ── ch0: legitimate frame, looped continuously ─────────────────
        self.src_frame = blocks.file_source(gr.sizeof_gr_complex, a.frame, True)
        self.src_frame.set_min_output_buffer(2 * L)
        self.connect((self.src_frame, 0), (self.usrp, 0))

        # ── ch1: perturbation * epsilon, looped continuously ───────────
        if a.clean:
            # Baseline: ch1 silent (same graph topology, zero adversary power).
            self.src_pert = blocks.null_source(gr.sizeof_gr_complex)
            self.connect((self.src_pert, 0), (self.usrp, 1))
            print("  CLEAN baseline: adversary channel (ch1) is silent.")
        else:
            self.src_pert = blocks.file_source(gr.sizeof_gr_complex, a.pert, True)
            self.src_pert.set_min_output_buffer(2 * L)
            self.mult_eps = blocks.multiply_const_cc(a.epsilon)
            self.mult_eps.set_min_output_buffer(2 * L)
            self.connect((self.src_pert, 0), (self.mult_eps, 0))
            self.connect((self.mult_eps, 0), (self.usrp, 1))

        self._readback()

    def _readback(self):
        try:
            print("  --- radio readback ---")
            for ch, tag in ((0, "legit "), (1, "advers")):
                print(f"    ch{ch} {tag}: f={self.usrp.get_center_freq(ch)/1e9:.4f}GHz "
                      f"ant={self.usrp.get_antenna(ch)} "
                      f"gain={self.usrp.get_gain(ch):.1f}dB "
                      f"(norm {self.usrp.get_normalized_gain(ch):.3f})")
            print(f"    rate    : {self.usrp.get_samp_rate()/1e6:.3f} MHz")
            print("  ----------------------")
        except Exception as e:
            print(f"  (readback failed: {e})")


def _check_lengths(a):
    """frame.bin and perturbation.bin must be the same length L (complex64)."""
    fn = os.path.getsize(a.frame) // 8
    if fn == 0:
        sys.exit(f"ERROR: {a.frame} is empty.")
    if a.clean:
        return fn
    if not os.path.exists(a.pert):
        sys.exit(f"ERROR: perturbation file not found: {a.pert}\n"
                 f"       (generate it on the attack machine, length L={fn})")
    pn = os.path.getsize(a.pert) // 8
    if pn != fn:
        sys.exit(f"ERROR: length mismatch — frame L={fn} but perturbation={pn}.\n"
                 f"       They MUST be equal for per-frame alignment.")
    return fn


def main():
    p = argparse.ArgumentParser(description="Exp3 per-frame adversarial TX")
    p.add_argument('--frame', default='/dev/shm/frame.bin',
                   help='legit WiFi frame loop (from exp3_make_frame.py)')
    p.add_argument('--pert', default='/dev/shm/perturbation.bin',
                   help='adversarial perturbation loop (length L == frame)')
    p.add_argument('--epsilon', type=float, default=0.05,
                   help='perturbation amplitude scale (PSR knob)')
    p.add_argument('--clean', action='store_true',
                   help='baseline run: keep ch1 silent (no perturbation)')
    p.add_argument('--freq', type=float, default=2.452e9,
                   help='center freq Hz (match the RX; wifi_tx.grc uses 2.452 GHz / ch9)')
    p.add_argument('--antenna', default='J1', help='TX antenna name')
    p.add_argument('--tx-gain', type=float, default=33.5,
                   help='ch0 legit ABSOLUTE TX gain in dB (realized cap ~33.5 @2.45GHz)')
    p.add_argument('--adv-gain', type=float, default=21.0,
                   help='ch1 adversary ABSOLUTE TX gain in dB (range 0..35)')
    p.add_argument('--duration', type=int, default=60, help='seconds to transmit')
    a = p.parse_args()

    L = _check_lengths(a)

    print("=" * 64)
    print("Experiment 3 — per-frame adversarial TX (headless, 2-mboard MIMO)")
    print("=" * 64)
    print(f"  legit  ch0 @ {ADDR_LEGIT}   frame={a.frame}")
    print(f"  advers ch1 @ {ADDR_ADV}   pert ={'(silent)' if a.clean else a.pert}")
    print(f"  L={L} samples ({L/SAMP_RATE*1e3:.1f} ms loop)   "
          f"freq={a.freq/1e9:.3f}GHz  fs={SAMP_RATE/1e6:.1f}MHz")
    if not a.clean:
        print(f"  epsilon={a.epsilon}   tx_gain={a.tx_gain}dB  adv_gain={a.adv_gain}dB")
    print(f"  duration={a.duration}s")
    print("  (if 'send buffer could not be resized': "
          "sudo sysctl -w net.core.wmem_max=2500000)")
    print("=" * 64)

    tb = adversary_tx(a, L)

    def stop(*_):
        tb.stop(); tb.wait(); sys.exit(0)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    tb.start()
    print(f"[TX] transmitting for {a.duration}s ...")
    time.sleep(a.duration)
    tb.stop(); tb.wait()
    print("Done.")


if __name__ == '__main__':
    main()

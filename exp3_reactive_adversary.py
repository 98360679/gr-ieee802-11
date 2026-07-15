#!/usr/bin/env python3
"""exp3_reactive_adversary.py — live self-synchronizing 2-channel adversary.

The adversary uses ONE radio (usrp2 1198) for BOTH sensing and striking. RX and TX
on the same USRP share the FPGA clock, so a frame detected at RX-time T can have its
delta scheduled at TX-time T+delta, sample-aligned, WITH NO CABLE to the victim.
That replaces the MIMO cable of Exp3 with the intra-radio clock — the realistic way
a reactive attacker self-synchronizes.

Pipeline:
  1. SENSE : RX device_6 (J1) for --sense-secs; set_time_now(0) so samples are timestamped.
  2. DETECT: find device_6's frame phase t0 and period P from the RX buffer.
  3. STRIKE: transmit the padded delta bundle (J2) with tx_time scheduled so its payload
             lands on device_6's predicted payloads; hold for --strike-secs.
  (A separate receiver, e.g. the b200, captures the superposition and the classifier scores.)

Modes:
  --detect-file CAP   : validate the detector on a recorded capture (no radio).
  --live              : run the full sense->detect->strike loop on 1198 (needs 1198 to hear device_6).

Hardware note: requires 1198 to actually receive device_6's frames (RX antenna facing the
victim). As of build time 1198's RX sees only RFI from its adversary-TX position — reposition
the RX antenna before --live.
"""
import sys, time, argparse
import numpy as np
FS = 5_000_000
FRAME_PERIOD = 65661          # device_6 replay slot (15661 active + 50000 gap)
DATA_OFF = 100; SPLIT = 1360  # payload begins at frame_start + DATA_OFF + SPLIT
SERIAL = "1198"; FREQ = 5.29e9


def detect_timing(iq, fs=FS, expect_P=FRAME_PERIOD):
    """Return (t0, P, confidence): device_6 frame phase (sample of first onset) and period.
    Robust to missing frames: locates power bursts, fits them to a regular grid."""
    from exp3_extract_frames import extract_frames_for_file
    import tempfile, os
    tmp = tempfile.mktemp('.bin'); np.asarray(iq, np.complex64).tofile(tmp)
    try:
        fr, starts, *_ = extract_frames_for_file(tmp, floor_pct=20.0, thr_mult=2.0)
    finally:
        os.remove(tmp)
    starts = np.sort(np.asarray(starts, dtype=np.int64))
    if len(starts) < 8:
        return None, None, 0.0
    # period from the modal near-P gap; phase from the grid residual
    gaps = np.diff(starts)
    near = gaps[(gaps > expect_P * 0.7) & (gaps < expect_P * 1.3)]
    P0 = float(np.median(near)) if len(near) else float(expect_P)
    # index each frame by cumulative gap-count (robust to missed frames AND clock drift):
    # +1 per ~1-period gap, +2 per double gap, etc. Then LS-fit start_i ~ a + b*idx_i.
    idx = np.concatenate([[0], np.cumsum(np.round(gaps / P0))]).astype(np.int64)
    A = np.vstack([np.ones_like(idx), idx]).astype(np.float64).T
    (a, b), *_ = np.linalg.lstsq(A, starts.astype(np.float64), rcond=None)
    P = int(round(b)); t0 = int(round(a)) % P
    resid = starts - (a + b * idx)
    conf = float(np.mean(np.abs(resid) < P / 16))      # within +/-P/16 (~4100 samp) of the grid
    return t0, P, conf


def run_live(a):
    from gnuradio import gr, blocks, uhd
    # ---- SENSE ----
    class rx(gr.top_block):
        def __init__(s, out):
            gr.top_block.__init__(s)
            s.u = uhd.usrp_source(f"serial={SERIAL}", uhd.stream_args(cpu_format="fc32", channels=[0]))
            s.u.set_samp_rate(FS); s.u.set_time_now(uhd.time_spec(0.0))
            s.u.set_center_freq(uhd.tune_request(FREQ, rf_freq=FREQ,
                                rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
            s.u.set_antenna(a.rx_ant, 0); s.u.set_normalized_gain(a.rx_gain, 0)
            s.k = blocks.file_sink(gr.sizeof_gr_complex, out, False); s.connect(s.u, s.k)
    sense = '/tmp/reactive_sense.bin'
    tb = rx(sense); tb.start(); time.sleep(a.sense_secs); tb.stop(); tb.wait()
    iq = np.fromfile(sense, dtype=np.complex64)
    t0, P, conf = detect_timing(iq)
    print(f"[detect] t0={t0} P={P} confidence={conf:.2f} (rms {np.sqrt(np.mean(np.abs(iq)**2)):.5f})", flush=True)
    if conf < 0.5:
        print("[abort] cannot resolve device_6 timing (1198 RX not hearing frames). Fix RX antenna.", flush=True)
        return
    # ---- STRIKE: schedule delta so its slot-start aligns to a future device_6 frame ----
    now = a.sense_secs + 1.0                                   # FPGA time has advanced past sensing
    k = int(np.ceil((now * FS - t0) / P)) + 5                  # first future frame with margin
    t_start = (t0 + k * P) / FS                                # TX-time to begin the delta bundle
    class tx(gr.top_block):
        def __init__(s, f):
            gr.top_block.__init__(s)
            s.u = uhd.usrp_sink(f"serial={SERIAL}", uhd.stream_args(cpu_format="fc32", channels=[0]))
            s.u.set_samp_rate(FS)
            s.u.set_center_freq(uhd.tune_request(FREQ, rf_freq=FREQ,
                                rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
            s.u.set_antenna(a.tx_ant, 0); s.u.set_normalized_gain(a.tx_gain, 0)
            s.u.set_start_time(uhd.time_spec(t_start))          # aligned start (shared FPGA clock)
            s.src = blocks.file_source(gr.sizeof_gr_complex, f, True); s.connect(s.src, s.u)
    print(f"[strike] delta bundle starts at TX-time {t_start:.4f}s (frame k={k}), holding {a.strike_secs}s", flush=True)
    tb = tx(a.delta); tb.start(); time.sleep(a.strike_secs); tb.stop(); tb.wait()
    print("[done] strike complete", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--detect-file', help='validate detector on a capture, no radio')
    p.add_argument('--live', action='store_true')
    p.add_argument('--delta', default='/home/nghoselab/captures/session14/attack/exp3_2ch_0714/ch1__scratch_pgd_untgt__m6.bin')
    p.add_argument('--rx-ant', default='J1'); p.add_argument('--tx-ant', default='J2')
    p.add_argument('--rx-gain', type=float, default=0.6); p.add_argument('--tx-gain', type=float, default=0.3)
    p.add_argument('--sense-secs', type=float, default=3.0); p.add_argument('--strike-secs', type=float, default=12.0)
    a = p.parse_args()
    if a.detect_file:
        iq = np.fromfile(a.detect_file, dtype=np.complex64)
        t0, P, conf = detect_timing(iq)
        print(f"detect: t0={t0}  period={P}  confidence={conf:.2f}")
        if t0 is not None:
            pay = (t0 + DATA_OFF + SPLIT) % (P or 1)
            print(f"predicted payload offset within frame = {pay} samp; a reactive strike fires delta here each period")
    elif a.live:
        run_live(a)
    else:
        print("specify --detect-file CAP or --live")


if __name__ == '__main__':
    main()

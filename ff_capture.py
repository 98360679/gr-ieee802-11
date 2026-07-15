#!/usr/bin/env python3
"""
ff_capture.py — "Fading Fingerprints" data-collection driver.

Cross-day RF-fingerprint study: Exp 1 = same-model devices, Exp 2 = different-model
devices, SAME receiver throughout, repeated Day 1 then Day 2. This driver captures ONE
device's clean data with the same gain-calibrated method proven in the attack work, but
retargeted for this study:

  * FREQUENCY  -> 3.3 GHz (the 2.4-4.4 GHz band every model in the lab can reach; the
                 N-series caps at 4.4 GHz so the old 5.29 GHz is out). Both the device
                 TX and the shared RX are tuned here.
  * CAPTURES   -> ~/captures/fading_fingerprints/day{D}/exp{E}/device_{id}/clean_run_{k}.bin
  * GAIN CAL   -> each device's tx_gain is tuned so the RECEIVED power hits a fixed target
                 (-44.5 dB), equalising SNR across devices and across days. This is the
                 control that makes the cross-day comparison honest: any Day-2 change is
                 drift, not a level difference. Use the SAME target both days.

Design is identical to lab_exp3_enroll_calib.py: the slow device TX (>60 s init) is brought
up ONCE and kept resident; tx_gain is nudged LIVE between 5 s RX probes until the received
level locks; then the RX re-runs to record the 30 s clean_run(s). The shared receiver
(b200 325936A, TX/RX antenna) runs as short subprocess probes via wifi_rx.py.

  # Exp 1 (same model), Day 1, one B205mini:
  python3 ff_capture.py --day 1 --exp 1 --device-id 1 --tx-serial 325426C \
        --tx-antenna TX/RX --run-ids 1,2,3

  # Exp 2 (different model), Day 1, the USRP2 (J1 antenna):
  python3 ff_capture.py --day 1 --exp 2 --device-id 3 --tx-serial 2192 \
        --tx-antenna J1 --run-ids 1,2,3
"""
import os, re, sys, time, signal, argparse, subprocess
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RXLOG = '/tmp/.ff_rx.log'

# The literal base path baked into the stock wifi_tx_updated.py / wifi_rx.py (patched out).
STOCK_BASE = '/home/nghoselab/captures/session14/enrollment/7_14_2026'
DEFAULT_ROOT = '/home/nghoselab/captures/fading_fingerprints'


def kill_stale():
    """free the radios: kill any lingering flowgraph python procs (never a shell)."""
    for pid in subprocess.run(['pgrep', '-f', 'wifi_tx_updated.py|wifi_rx.py'],
                              capture_output=True, text=True).stdout.split():
        try:
            if open('/proc/%s/comm' % pid).read().strip() == 'python3':
                os.kill(int(pid), signal.SIGKILL)
        except Exception:
            pass


def patch_var(py, **kv):
    s = open(py).read()
    for k, v in kv.items():
        s = re.sub(r'(self\.%s = %s = )[0-9.]+' % (k, k), r'\g<1>%s' % v, s)
    open(py, 'w').write(s)


def patch_freq(py, freq):
    """set the default center frequency (must be one of the flowgraph's freq_options)."""
    patch_var(py, freq='%.1f' % freq)


def patch_basedir(py, new_base):
    """retarget every '.../device_%d/...' capture path to new_base. Idempotent and
    re-targetable: matches from the opening quote up to /device_%d/ regardless of the
    current base, so it works whether the file is stock or already patched."""
    s = open(py).read()
    s2 = re.sub(r"'/[^']*/device_%d/", "'" + new_base + "/device_%d/", s)
    open(py, 'w').write(s2)
    return s2 != s


def patch_serial(py, ident):
    kind = 'addr' if re.match(r'^\d+\.\d+\.\d+\.\d+$', ident) else 'serial'
    s = open(py).read()
    assert re.search(r'(?:serial|addr)=[A-Za-z0-9.:_]+', s), "no serial=/addr= in %s" % py
    open(py, 'w').write(re.sub(r'(?:serial|addr)=[A-Za-z0-9.:_]+', '%s=%s' % (kind, ident), s, count=1))
    return '%s=%s' % (kind, ident)


def patch_antenna(py, ant):
    """usrp2 wants J1/J2, b200 wants TX/RX — set the TX antenna to match the device."""
    s = open(py).read()
    s2 = re.sub(r'set_antenna\("[^"]*", 0\)', 'set_antenna("%s", 0)' % ant, s, count=1)
    open(py, 'w').write(s2)
    return ant


def last_signal_db(log):
    m = re.findall(r'signal=\s*(-?\d+\.?\d*)\s*dB', open(log).read()) if os.path.exists(log) else []
    return float(m[-1]) if m else None


def rx_run(rx_py, secs, warm=5.0):
    """run wifi_rx.py for ~secs of measurement (b200 inits ~2s); return signal dB.
    Also (re)writes clean_run — search probes overwrite; the final call is the keeper."""
    p = subprocess.Popen("QT_QPA_PLATFORM=offscreen timeout %d python3 %s" % (int(warm + secs), rx_py),
                         shell=True, stdout=open(RXLOG, 'w'), stderr=subprocess.STDOUT,
                         preexec_fn=os.setsid)
    p.wait()
    return last_signal_db(RXLOG)


def rx_bad_packet():
    return os.path.exists(RXLOG) and any(('bad packet' in l.lower() or 'vrt' in l.lower())
                                         for l in open(RXLOG))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', type=int, required=True, help='study day: 1 or 2')
    ap.add_argument('--exp', type=int, required=True, choices=[1, 2],
                    help='1 = same-model devices, 2 = different-model devices')
    ap.add_argument('--device-id', type=int, required=True)
    ap.add_argument('--run-id', type=int, default=1)
    ap.add_argument('--run-ids', default='1,2,3', help='comma list — several runs in one init')
    ap.add_argument('--tx-serial', required=True, help="serial= (or IP for addr=)")
    ap.add_argument('--tx-antenna', default=None, help="TX antenna: TX/RX (b200) or J1 (usrp2)")
    ap.add_argument('--freq', type=float, default=3.3e9, help='center freq (default 3.3 GHz)')
    ap.add_argument('--out-root', default=DEFAULT_ROOT)
    ap.add_argument('--target-db', type=float, default=-44.5)
    ap.add_argument('--tol', type=float, default=0.5)
    ap.add_argument('--g0', type=float, default=0.6)
    ap.add_argument('--max-gain', type=float, default=1.0, help='hard cap on tx_gain (device limit)')
    ap.add_argument('--fix-gain', type=float, default=None, help='skip search: capture at this exact gain')
    ap.add_argument('--tx-py', default=os.path.join(HERE, 'wifi_tx_updated.py'))
    ap.add_argument('--rx-py', default=os.path.join(HERE, 'wifi_rx.py'))
    ap.add_argument('--init-wait', type=float, default=70.0, help='TX radio init seconds (>60)')
    ap.add_argument('--settle', type=float, default=2.0, help='seconds after a live gain change')
    ap.add_argument('--probe-secs', type=float, default=5.0)
    ap.add_argument('--capture-secs', type=float, default=30.0)
    ap.add_argument('--max-iters', type=int, default=8)
    ap.add_argument('--slope', type=float, default=89.75)
    a = ap.parse_args()

    base = '%s/day%d/exp%d' % (a.out_root, a.day, a.exp)
    dev_dir = '%s/device_%d' % (base, a.device_id)
    os.makedirs(dev_dir, exist_ok=True)
    log_csv = '%s/enroll_gains.csv' % base

    kill_stale(); time.sleep(2)
    # stamp identity, freq, capture directory, and initial gain BEFORE importing the TX flowgraph
    patch_var(a.tx_py, device_id=a.device_id, run_id=a.run_id, tx_gain='%.4f' % a.g0)
    patch_var(a.rx_py, device_id=a.device_id, run_id=a.run_id)
    patch_freq(a.tx_py, a.freq); patch_freq(a.rx_py, a.freq)
    if not patch_basedir(a.tx_py, base): print('[warn] TX base path not patched — check wifi_tx_updated.py')
    if not patch_basedir(a.rx_py, base): print('[warn] RX base path not patched — check wifi_rx.py')
    used = patch_serial(a.tx_py, a.tx_serial)
    ant = patch_antenna(a.tx_py, a.tx_antenna) if a.tx_antenna else 'TX/RX'
    print(f"[ff] Fading Fingerprints  day{a.day} exp{a.exp}  device_{a.device_id} run_{a.run_id}")
    print(f"[ff]   TX {used} ant {ant}  freq {a.freq/1e9:.3f} GHz  target {a.target_db} dB (+-{a.tol})")
    print(f"[ff]   -> {dev_dir}/clean_run_*.bin")

    # bring the TX up RESIDENT (DSP in GR threads); QApplication offscreen for the Qt widgets
    from PyQt5 import Qt
    import importlib
    TXM = importlib.import_module('wifi_tx_updated')
    qapp = Qt.QApplication(sys.argv)
    tb = TXM.wifi_tx_updated()
    tb.start()
    print(f"[ff] TX resident; waiting {a.init_wait:.0f}s for radio init (>60s)...", flush=True)
    time.sleep(a.init_wait)

    g, hist, locked = a.g0, [], None
    try:
        if a.fix_gain is not None:                       # no search: capture at the device's cap
            g = a.fix_gain
            tb.set_tx_gain(g)
            time.sleep(a.settle)
            db = rx_run(a.rx_py, a.probe_secs)
            locked = (g, db)
            print(f"[ff] FIXED gain {g:.4f} -> signal={db if db is None else round(db,2)} dB", flush=True)
        else:
          for it in range(a.max_iters):
            tb.set_tx_gain(g)                       # LIVE gain change, no re-init
            time.sleep(a.settle)
            db = rx_run(a.rx_py, a.probe_secs)
            print(f"  iter {it}: tx_gain={g:.4f} -> signal={db if db is None else round(db,2)} dB", flush=True)
            if db is None:
                g = min(a.max_gain, g + 0.1); continue
            hist.append((g, db))
            err = a.target_db - db
            if abs(err) <= a.tol:
                locked = (g, db); break
            if len(hist) >= 2 and hist[-1][0] != hist[-2][0] and hist[-1][1] != hist[-2][1]:
                slope = (hist[-1][1] - hist[-2][1]) / (hist[-1][0] - hist[-2][0])
                slope = slope if abs(slope) > 5 else a.slope
            else:
                slope = a.slope
            g = min(a.max_gain, max(0.0, g + err / slope))

        if locked is None:
            print(f"[ff] did NOT converge (last g={g:.4f}); not capturing."); return
        g, db = locked
        run_ids = [int(x) for x in a.run_ids.split(',')] if a.run_ids else [a.run_id]
        print(f"[ff] LOCKED device_{a.device_id}: tx_gain={g:.4f}  signal={db:.2f} dB — "
              f"capturing runs {run_ids} @ {a.capture_secs:.0f}s each (one init)", flush=True)
        tb.set_tx_gain(g)
        header = not os.path.exists(log_csv)
        for rid in run_ids:                          # TX stays resident; only the RX re-runs per run
            patch_var(a.rx_py, run_id=rid)
            cap_db = rx_run(a.rx_py, a.capture_secs)
            cr = f"{dev_dir}/clean_run_{rid}.bin"
            sz = os.path.getsize(cr) / 1e6 if os.path.exists(cr) else 0
            ber = f"{dev_dir}/ber_frames_run_{rid}.jsonl"
            nfr = sum(1 for _ in open(ber)) if os.path.exists(ber) else 0
            print(f"RESULT day{a.day} exp{a.exp} device_{a.device_id} run_{rid}: tx_gain={g:.4f}  "
                  f"signal={cap_db if cap_db is None else round(cap_db, 2)} dB  clean_run={sz:.0f} MB  "
                  f"frames={nfr}  bad_packet={'YES' if rx_bad_packet() else 'no'}", flush=True)
            with open(log_csv, 'a') as f:
                if header:
                    f.write("day,exp,device_id,run_id,tx_gain,signal_db,clean_run_MB,frames\n"); header = False
                f.write(f"{a.day},{a.exp},{a.device_id},{rid},{g:.4f},"
                        f"{cap_db if cap_db is not None else db:.2f},{sz:.0f},{nfr}\n")
    finally:
        tb.stop(); tb.wait()


if __name__ == '__main__':
    main()

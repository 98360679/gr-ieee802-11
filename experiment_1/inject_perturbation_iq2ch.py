#!/usr/bin/env python3
"""
inject_perturbation_iq2ch.py  —  OTA Adversarial Attack (IQ2CH model)

SIGNAL CHAIN (from wifi_rx_perturb.py + match_frame.py)
  af.bin  : flat complex64, each value = one 64-pt FFT bin from the GNU Radio
            ieee802_11 sync+FFT chain (post-sync, pre-equaliser).
            Sliced into 288-sample windows → SARP IQ2CH features.
  tx.bin  : interleaved float32 I/Q, raw USRP TX samples.
  rx-log.txt : frame ID → sample offset in af.bin
  tx-log.txt : frame ID → sample offset in tx.bin
  Frame IDs link RX features to TX injection sites.

PIPELINE (per matched frame)
  1. af.bin[rx_offset:rx_offset+288]  — 288 complex64 FFT bins (real features)
  2. Re/Im split + RMS-normalise → (288,2) float32  [matches training exactly]
  3. Batch all frames → FGSM/PGD gradient → delta_feat (N,288,2)
  4. Denormalise: delta_complex = (delta[:,0] + j*delta[:,1]) * rms
  5. Group 288 complex values into FFT frames (4×64 full + 1×32 partial)
  6. IFFT each 64-bin group → 64 time-domain samples delta_time
  7. Add delta_time to tx.bin data OFDM symbols (after preamble)
  8. Regenerate CP from modified symbol tail
  Channel assumption: H ≈ 1 (flat fading, short-range lab).

Usage:
    python3 inject_perturbation_iq2ch.py \
        --tx-bin data/tx.bin --tx-log data/tx-log.txt \
        --rx-bin data/af.bin --rx-log data/rx-log.txt \
        --weights models/sarp_iq2ch/best_model.keras \
        --device-label 6 --mode untargeted \
        --attack both --pgd-steps 40 \
        --epsilons 0.0 0.05 0.1 0.2 0.3 \
        --output-dir data/perturbed_iq2ch/
"""

import os, re, json, argparse, time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings; warnings.filterwarnings('ignore')
import tensorflow as tf
from collections import Counter

PREAMBLE_LEN    = 400
FFT_SIZE        = 64
CP_LEN          = 16
OFDM_SYM_LEN    = FFT_SIZE + CP_LEN       # 80
N_DATA_OFDM     = 177
FINGERPRINT_LEN = 288
N_FFT_FRAMES    = FINGERPRINT_LEN // FFT_SIZE   # 4 full + 32 partial
NUM_CLASSES     = 7

DATA_SC = [sc % FFT_SIZE for sc in
           list(range(-26,-21)) + list(range(-20,-7)) +
           list(range(-6,  0))  + list(range(1,  7)) +
           list(range( 8, 21))  + list(range(22, 27))]  # 48 bins


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_model(path):
    model = tf.keras.models.load_model(path)
    out   = model(tf.zeros((2, FINGERPRINT_LEN, 2)), training=False).numpy()
    assert abs(out.sum() - 2.0) < 0.05, f"Softmax sanity: {out.sum()}"
    print(f"[OK] Model  input={model.input.shape}  output={model.output.shape}")
    return model

def load_tx_bin(path):
    raw = np.fromfile(path, dtype=np.float32)
    if len(raw) % 2: raw = raw[:-1]
    tx = (raw[0::2] + 1j*raw[1::2]).astype(np.complex64)
    print(f"[OK] tx.bin  {len(tx):,} samples")
    return tx

def save_tx_bin(tx, path):
    out = np.empty(len(tx)*2, dtype=np.float32)
    out[0::2] = np.real(tx); out[1::2] = np.imag(tx)
    out.tofile(path)

def load_rx_bin(path):
    rx = np.fromfile(path, dtype=np.complex64)
    print(f"[OK] af.bin  {len(rx):,} FFT bins  "
          f"(= {len(rx)//FFT_SIZE} frames × {FFT_SIZE})")
    return rx

def parse_log(path):
    pat = re.compile(r'frame ID[: ]+(\d+)\s*@\s*sample\s*(\d+)', re.IGNORECASE)
    d = {}
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m: d[m.group(1).zfill(4)] = int(m.group(2))
    print(f"[OK] {os.path.basename(path)}  {len(d)} frames")
    return d


# ── Preprocessing (matches convert_afbin.py + convert_iq2ch.py) ─────────────

def extract_feature(rx_signal, rx_offset):
    bins = rx_signal[rx_offset : rx_offset + FINGERPRINT_LEN]
    if len(bins) < FINGERPRINT_LEN:
        return None, None
    I   = np.real(bins).astype(np.float32)
    Q   = np.imag(bins).astype(np.float32)
    rms = max(float(np.sqrt(np.mean(I**2 + Q**2))), 1e-10)
    return np.stack([I/rms, Q/rms], axis=-1), rms   # (288,2), scalar


# ── Injection: delta_feat → tx.bin ───────────────────────────────────────────

def inject_delta(tx, tx_offset, delta_feat, rms):
    """
    Denormalise delta_feat → complex delta, group into 64-bin FFT frames,
    IFFT each → add to tx.bin data OFDM symbols, regenerate CP.
    """
    dI = delta_feat[:, 0] * rms
    dQ = delta_feat[:, 1] * rms
    d  = (dI + 1j*dQ).astype(np.complex64)   # (288,)

    for sym_i in range(N_FFT_FRAMES + 1):     # 0..4
        b0 = sym_i * FFT_SIZE
        b1 = min(b0 + FFT_SIZE, FINGERPRINT_LEN)
        if b0 >= FINGERPRINT_LEN: break

        df = np.zeros(FFT_SIZE, dtype=np.complex64)
        df[:b1-b0] = d[b0:b1]

        sym_start = tx_offset + PREAMBLE_LEN + sym_i * OFDM_SYM_LEN
        fft_start = sym_start + CP_LEN
        fft_end   = fft_start + FFT_SIZE
        if fft_end > len(tx): break

        new_time = np.fft.ifft(np.fft.fft(tx[fft_start:fft_end]) + df
                               ).astype(np.complex64)
        tx[fft_start:fft_end]            = new_time
        tx[sym_start:sym_start + CP_LEN] = new_time[-CP_LEN:]


# ── Loss functions ────────────────────────────────────────────────────────────

def make_loss_fn(mode, true_label, target_label):
    def _unt(model, x):
        lg = model(x, training=False)
        c  = lg[:, true_label]
        m  = tf.one_hot(tf.fill([tf.shape(lg)[0]], true_label),
                        NUM_CLASSES, on_value=-1e9, off_value=0.0)
        return tf.reduce_mean(c - tf.reduce_max(lg + m, axis=1))
    def _tgt(model, x):
        lg = model(x, training=False)
        t  = lg[:, target_label]
        m  = tf.one_hot(tf.fill([tf.shape(lg)[0]], target_label),
                        NUM_CLASSES, on_value=-1e9, off_value=0.0)
        return tf.reduce_mean(tf.reduce_max(lg + m, axis=1) - t)
    return _unt if mode == 'untargeted' else _tgt


# ── FGSM / PGD ───────────────────────────────────────────────────────────────

def fgsm(model, feat, loss_fn, eps):
    x = tf.Variable(feat)
    with tf.GradientTape() as t: loss = loss_fn(model, x)
    return -eps * np.sign(t.gradient(loss, x).numpy()), float(loss.numpy())

def pgd(model, feat, loss_fn, eps, steps, alpha=None):
    if alpha is None: alpha = 2.5 * eps / max(steps, 1)
    orig = feat.astype(np.float32)
    x    = tf.Variable(orig + np.random.default_rng(42).uniform(
                        -eps, eps, orig.shape).astype(np.float32))
    best_d, best_l = np.zeros_like(orig), np.inf
    for _ in range(steps):
        with tf.GradientTape() as t: loss = loss_fn(model, x)
        new_x = np.clip(x.numpy() - alpha*np.sign(t.gradient(loss, x).numpy()),
                        orig-eps, orig+eps)
        x.assign(new_x)
        lv = float(loss.numpy())
        if lv < best_l: best_l, best_d = lv, new_x - orig
    return best_d.astype(np.float32), best_l


# ── BER ───────────────────────────────────────────────────────────────────────

def measure_ber(tx_orig, tx_pert, tx_offsets):
    n_err = n_tot = 0
    for off in tx_offsets:
        for i in range(N_DATA_OFDM):
            s = off + PREAMBLE_LEN + i*OFDM_SYM_LEN
            e = s + OFDM_SYM_LEN
            if e > min(len(tx_orig), len(tx_pert)): break
            fo = np.fft.fft(tx_orig[s+CP_LEN:e])
            fp = np.fft.fft(tx_pert[s+CP_LEN:e])
            for b in DATA_SC:
                n_err += int((np.real(fo[b])>0) != (np.real(fp[b])>0))
                n_tot += 1
    return (n_err/n_tot if n_tot else 0.0), n_err, n_tot


# ── Core per-epsilon loop ─────────────────────────────────────────────────────

def inject_one(tx, rx, tx_offs, rx_offs, model, loss_fn,
               eps, attack, pgd_steps, true_label, target_label):

    feat_list, rms_list, v_tx = [], [], []
    for to, ro in zip(tx_offs, rx_offs):
        feat, rms = extract_feature(rx, ro)
        if feat is not None:
            feat_list.append(feat); rms_list.append(rms); v_tx.append(to)

    if not feat_list: raise RuntimeError("No valid matched frames.")
    batch = np.stack(feat_list).astype(np.float32)

    # Classify before
    preds_b  = np.argmax(model(tf.constant(batch), training=False).numpy(), 1)
    acc_b    = float(np.mean(preds_b == true_label))

    if eps == 0.0:
        # Print distribution to help debug device label issues
        print(f"  [diag ε=0] pred dist: {dict(sorted(Counter(preds_b.tolist()).items()))}")

    # Compute delta
    if eps == 0.0:
        delta = np.zeros_like(batch)
        loss_val = float(loss_fn(model, tf.constant(batch)).numpy())
    elif attack == 'fgsm':
        delta, loss_val = fgsm(model, batch, loss_fn, eps)
    else:
        delta, loss_val = pgd(model, batch, loss_fn, eps, pgd_steps)

    # Classify after (feature space — upper bound on OTA success)
    preds_a   = np.argmax(
        model(tf.constant(batch + delta), training=False).numpy(), 1)
    acc_a     = float(np.mean(preds_a == true_label))
    targ_rate = (float(np.mean(preds_a == target_label))
                 if target_label is not None else None)

    # Inject into tx copy
    tx_pert = tx.copy()
    for i, to in enumerate(v_tx):
        inject_delta(tx_pert, to, delta[i], rms_list[i])

    ber, n_err, n_tot = measure_ber(tx, tx_pert, v_tx)
    return tx_pert, acc_b, acc_a, targ_rate, ber, n_err, n_tot, loss_val, len(v_tx)


# ── CLI / main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tx-bin',       required=True)
    p.add_argument('--tx-log',       required=True)
    p.add_argument('--rx-bin',       required=True)
    p.add_argument('--rx-log',       required=True)
    p.add_argument('--weights',      required=True)
    p.add_argument('--device-label', type=int, required=True)
    p.add_argument('--mode',         choices=['untargeted','targeted'],
                   default='untargeted')
    p.add_argument('--target-label', type=int, default=None)
    p.add_argument('--epsilons',     type=float, nargs='+',
                   default=[0.0, 0.05, 0.1, 0.2, 0.3])
    p.add_argument('--attack',       choices=['fgsm','pgd','both'],
                   default='both')
    p.add_argument('--pgd-steps',    type=int, default=40)
    p.add_argument('--max-frames',   type=int, default=258)
    p.add_argument('--num-classes',  type=int, default=7)
    p.add_argument('--output-dir',   default='./perturbed_iq2ch')
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == 'targeted':
        if args.target_label is None: raise ValueError('--target-label required')
        if args.target_label == args.device_label:
            raise ValueError('--target-label must differ from --device-label')

    global NUM_CLASSES; NUM_CLASSES = args.num_classes
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir,'config.json'),'w') as f:
        json.dump(vars(args), f, indent=2)

    print('='*74)
    print('inject_perturbation_iq2ch.py — OTA Adversarial Attack (IQ2CH)')
    print('='*74)
    print(f'  Mode         : {args.mode.upper()}'
          +(f' → class {args.target_label}' if args.mode=='targeted' else ''))
    print(f'  Device label : {args.device_label}')
    print(f'  Attack(s)    : {args.attack.upper()}')
    print(f'  Epsilons     : {args.epsilons}')
    print(f'  Signal chain : af.bin (real RX FFT) → gradient → IFFT → tx.bin')

    model  = load_model(args.weights)
    tx     = load_tx_bin(args.tx_bin)
    rx     = load_rx_bin(args.rx_bin)
    tx_log = parse_log(args.tx_log)
    rx_log = parse_log(args.rx_log)

    matched = sorted(set(tx_log) & set(rx_log))[:args.max_frames]
    print(f"[OK] {len(matched)} matched frames")
    tx_offs = [tx_log[f] for f in matched]
    rx_offs = [rx_log[f] for f in matched]

    is_tgt  = (args.mode == 'targeted')
    loss_fn = make_loss_fn(args.mode, args.device_label, args.target_label)
    attacks = ['fgsm','pgd'] if args.attack == 'both' else [args.attack]

    results = []
    hdr = ('Attack   Eps    Frames  AccBefore  AccAfter  AttSucc'
           +('  TargRate' if is_tgt else '')
           +'     BER       N_err  Loss     Time')
    print('\n'+'='*len(hdr)); print(hdr); print('-'*len(hdr))

    for attack in attacks:
        for eps in args.epsilons:
            t0 = time.time()
            tx_pert, acc_b, acc_a, trate, ber, n_err, n_tot, lv, nf = \
                inject_one(tx, rx, tx_offs, rx_offs, model, loss_fn,
                           eps, attack, args.pgd_steps,
                           args.device_label, args.target_label)
            elapsed = time.time()-t0
            tag = f'{attack}_eps{eps:.3f}'.replace('.','p')
            out = os.path.join(args.output_dir, f'perturbed_{tag}.bin')
            save_tx_bin(tx_pert, out)

            line = ('%6s  %6.3f  %6d  %9.4f  %8.4f  %7.4f' %
                    (attack, eps, nf, acc_b, acc_a, 1-acc_a))
            if is_tgt: line += '  %8.4f' % (trate or 0.0)
            line += ('  %10.2e  %6d  %8.4f  %5.1fs' % (ber, n_err, lv, elapsed))
            print(line)

            rec = dict(attack=attack, epsilon=eps, n_frames=nf,
                       acc_before=acc_b, acc_after=acc_a,
                       attack_success=1-acc_a,
                       ber=ber, n_errors=n_err, n_bits=n_tot,
                       loss=lv, time_s=elapsed,
                       output_file=os.path.basename(out))
            if is_tgt: rec['target_rate'] = trate
            results.append(rec)
        print()

    with open(os.path.join(args.output_dir,'injection_results.json'),'w') as f:
        json.dump({'config':vars(args),'results':results}, f, indent=2)

    print('\n'+'='*74)
    print('SUMMARY  (AccAfter = feature-space upper bound; OTA may differ)')
    print('='*74)
    print('%-6s  %7s  %9s  %9s  %9s  %10s' %
          ('Attack','Epsilon','AccBefore','AccAfter','AttSucc','BER')
          +('  %9s'%'TargRate' if is_tgt else ''))
    print('-'*74)
    for r in results:
        line = ('%-6s  %7.4f  %9.4f  %9.4f  %9.4f  %10.2e' %
                (r['attack'],r['epsilon'],r['acc_before'],
                 r['acc_after'],r['attack_success'],r['ber']))
        if is_tgt: line += '  %9.4f' % (r.get('target_rate') or 0.0)
        print(line)
    print(f'\nPerturbed binaries → {args.output_dir}/')
    print('Next: wifi_tx_perturbed.py (unchanged), then compute_ber_ota.py')

if __name__ == '__main__':
    main()

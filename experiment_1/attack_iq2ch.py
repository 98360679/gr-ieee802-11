#!/usr/bin/env python3
"""
attack_iq2ch.py
===============
Feature-domain white-box adversarial attacks on the 2-channel IQ SARP model.

Untargeted: fool model away from true class  (any wrong prediction = success)
Targeted:   force model to predict a specific class (impersonation attack)

BER is computed theoretically for 4 modulations (BPSK/QPSK/16QAM/64QAM)
using the Q-function under uniform-bounded perturbation as additive noise.

Usage:
    python3 attack_iq2ch.py \
        --weights models/sarp_iq2ch/best_model.keras \
        --npz     data/sarp_iq2ch.npz \
        --mode    untargeted \
        --device-label 4 \
        --epsilon 0.0 0.01 0.05 0.1 0.2 0.3 0.5 \
        --pgd-steps 200 --trials 5 \
        --output-dir ./results_untargeted

    python3 attack_iq2ch.py \
        --weights models/sarp_iq2ch/best_model.keras \
        --npz     data/sarp_iq2ch.npz \
        --mode    targeted --target-label 2 \
        --device-label 4 \
        --epsilon 0.0 0.05 0.1 0.2 0.3 0.5 \
        --pgd-steps 200 --trials 5 \
        --output-dir ./results_targeted_dev2
"""

import os, argparse, json, time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erfc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WIFI_BER  = 1e-5
N_CLASSES = 7
ATT_COLORS = {'fgsm': '#e74c3c', 'pgd': '#2980b9'}
MOD_COLORS = {'bpsk': '#1a6faf', 'qpsk': '#2ca02c',
              '16qam': '#d62728', '64qam': '#9467bd'}


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path):
    model = tf.keras.models.load_model(path)
    dummy = tf.zeros((2, 288, 2), dtype=tf.float32)
    out   = model(dummy, training=False).numpy()
    assert abs(out.sum() - 2.0) < 0.01, f"Softmax sanity: sum={out.sum()}"
    print(f"[OK] Model  input={model.input.shape}  output={model.output.shape}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_traces(npz_path, device_label, n=500):
    d = np.load(npz_path)
    X, y = d['X'].astype(np.float32), d['y'].astype(int)
    assert X.ndim == 3 and X.shape[2] == 2, \
        f"Expected (N,288,2), got {X.shape}. Run convert_iq2ch.py first."
    if y.min() == 1:
        y -= 1
    idx = np.where(y == device_label)[0]
    if len(idx) == 0:
        raise ValueError(f"No traces for label {device_label}. "
                         f"Available: {sorted(set(y.tolist()))}")
    traces = X[idx[:n]]
    print(f"[OK] Loaded {len(traces)} traces  class={device_label}  "
          f"shape={traces.shape}")
    return traces


# ─────────────────────────────────────────────────────────────────────────────
# BER — theoretical, Q-function based
# ─────────────────────────────────────────────────────────────────────────────

def theoretical_ber(epsilon, modulation):
    """
    BER under epsilon-bounded uniform perturbation modelled as AWGN.

    Uniform[-ε, ε] has variance σ² = ε²/3.
    Signal is RMS-normalised → unit power.

    Crossover points (BER = 1e-5):
        BPSK  : ε ≈ 0.574
        QPSK  : ε ≈ 0.406
        16QAM : ε ≈ 0.193   ← enters WiFi-relevant range at ε > 0.1
        64QAM : ε ≈ 0.097   ← crosses 1e-5 already at ε = 0.1
    """
    if epsilon <= 1e-12:
        return 0.0
    noise_var = epsilon ** 2 / 3.0
    if modulation == 'bpsk':
        # 1 bit/symbol, Eb = signal_power = 1
        return float(0.5 * erfc(np.sqrt(1.0 / noise_var)))
    elif modulation == 'qpsk':
        # 2 bits/symbol
        return float(0.5 * erfc(np.sqrt(0.5 / noise_var)))
    elif modulation == '16qam':
        # Gray-coded: BER ≈ (3/8) erfc(sqrt(Es / 10 N0))
        return float(0.375 * erfc(np.sqrt(1.0 / (10.0 * noise_var))))
    elif modulation == '64qam':
        # Gray-coded: BER ≈ (7/24) erfc(sqrt(Es / 42 N0))
        return float((7/24) * erfc(np.sqrt(1.0 / (42.0 * noise_var))))
    else:
        raise ValueError(f"Unknown modulation: {modulation}")


MODULATIONS = ('bpsk', 'qpsk', '16qam', '64qam')


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions  (we always MINIMISE these)
#
# Untargeted CW:  loss = E[correct_logit - max_other_logit]
#   At baseline: positive (model confident in correct class)
#   After attack: goes negative (model prefers some other class)
#
# Targeted CW:    loss = E[max_non_target_logit - target_logit]
#   At baseline: positive (target not predicted)
#   After attack: goes negative (target becomes top prediction)
#
# IMPORTANT: both losses are MINIMISED → gradient DESCENT → x -= lr * grad
# ─────────────────────────────────────────────────────────────────────────────

def untargeted_loss(model, x, true_label, num_classes):
    logits  = model(x, training=False)
    correct = logits[:, true_label]
    mask    = tf.one_hot(
        tf.fill([tf.shape(logits)[0]], true_label),
        num_classes, on_value=-1e9, off_value=0.0)
    other   = tf.reduce_max(logits + mask, axis=1)
    return tf.reduce_mean(correct - other)   # minimise → fool


def targeted_loss(model, x, target_label, num_classes):
    logits = model(x, training=False)
    target = logits[:, target_label]
    mask   = tf.one_hot(
        tf.fill([tf.shape(logits)[0]], target_label),
        num_classes, on_value=-1e9, off_value=0.0)
    other  = tf.reduce_max(logits + mask, axis=1)
    return tf.reduce_mean(other - target)    # minimise → steer to target


# ─────────────────────────────────────────────────────────────────────────────
# FGSM  — single gradient step, DESCENT (subtract gradient)
# ─────────────────────────────────────────────────────────────────────────────

def fgsm(model, traces, loss_fn, epsilon):
    if epsilon == 0.0:
        loss = loss_fn(model, tf.constant(traces), )
        return traces.copy(), float(loss.numpy())

    x = tf.Variable(traces, dtype=tf.float32)
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x)
    grad = tape.gradient(loss, x).numpy()

    # Gradient DESCENT: subtract gradient to minimise loss
    perturbed = traces - epsilon * np.sign(grad)
    return perturbed.astype(np.float32), float(loss.numpy())


# ─────────────────────────────────────────────────────────────────────────────
# PGD  — iterative FGSM with projection, DESCENT
# ─────────────────────────────────────────────────────────────────────────────

def pgd(model, traces, loss_fn, epsilon, steps=200, alpha=None):
    if epsilon == 0.0:
        loss = loss_fn(model, tf.constant(traces))
        return traces.copy(), float(loss.numpy())

    if alpha is None:
        alpha = 2.5 * epsilon / steps

    rng  = np.random.default_rng(42)
    init = traces + rng.uniform(-epsilon, epsilon,
                                traces.shape).astype(np.float32)
    x    = tf.Variable(np.clip(init, traces - epsilon, traces + epsilon),
                       dtype=tf.float32)
    orig = traces.astype(np.float32)

    best_x = x.numpy().copy()
    best_l = np.inf        # tracking minimum loss

    for _ in range(steps):
        with tf.GradientTape() as tape:
            loss = loss_fn(model, x)
        grad  = tape.gradient(loss, x).numpy()

        # Gradient DESCENT: subtract gradient
        new_x = x.numpy() - alpha * np.sign(grad)

        # Project back into ε-ball
        new_x = np.clip(new_x, orig - epsilon, orig + epsilon)
        x.assign(new_x)

        lv = float(loss.numpy())
        if lv < best_l:
            best_l = lv
            best_x = new_x.copy()

    return best_x.astype(np.float32), best_l


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(model, traces, label):
    preds = np.argmax(model(tf.constant(traces), training=False).numpy(), axis=1)
    return float(np.mean(preds == label))


def target_rate(model, traces, label):
    preds = np.argmax(model(tf.constant(traces), training=False).numpy(), axis=1)
    return float(np.mean(preds == label))


def pred_distribution(model, traces, num_classes):
    preds = np.argmax(model(tf.constant(traces), training=False).numpy(), axis=1)
    return np.bincount(preds, minlength=num_classes).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def get_loss_fn(mode, true_label, target_label, num_classes):
    if mode == 'untargeted':
        return lambda model, x: untargeted_loss(model, x, true_label, num_classes)
    else:
        return lambda model, x: targeted_loss(model, x, target_label, num_classes)


def run(args, model, traces):
    is_targeted  = (args.mode == 'targeted')
    target_label = args.target_label
    num_cls      = args.num_classes
    loss_fn      = get_loss_fn(args.mode, args.device_label,
                               target_label, num_cls)

    results = {a: {e: [] for e in args.epsilon} for a in ['fgsm', 'pgd']}

    hdr = ('Attack   Eps    Trial  AccBefore  AccAfter  AttackSucc'
           + ('  TargetRate' if is_targeted else '')
           + '  BPSK-BER    16QAM-BER   64QAM-BER    Loss     Time')
    print('\n' + '='*len(hdr))
    print(hdr)
    print('-'*len(hdr))

    for attack_name in ['fgsm', 'pgd']:
        for eps in args.epsilon:
            for trial in range(args.trials):
                rng = np.random.default_rng(
                    trial * 10000 + int(eps * 1000))
                idx   = rng.choice(len(traces),
                                   size=min(args.batch_size, len(traces)),
                                   replace=False)
                batch = traces[idx]

                t0          = time.time()
                acc_before  = accuracy(model, batch, args.device_label)

                if attack_name == 'fgsm':
                    pert, loss = fgsm(model, batch, loss_fn, eps)
                else:
                    pert, loss = pgd(model, batch, loss_fn, eps,
                                     steps=args.pgd_steps)

                acc_after   = accuracy(model, pert, args.device_label)
                att_success = 1.0 - acc_after
                trate       = target_rate(model, pert, target_label) \
                              if is_targeted else None
                pred_dist   = pred_distribution(model, pert, num_cls)
                bers        = {m: theoretical_ber(eps, m) for m in MODULATIONS}
                elapsed     = time.time() - t0

                rec = dict(
                    acc_before=acc_before, acc_after=acc_after,
                    attack_success=att_success,
                    pred_dist=pred_dist, loss=loss, time=elapsed,
                    **{f'ber_{m}': bers[m] for m in MODULATIONS}
                )
                if is_targeted:
                    rec['target_rate'] = trate
                results[attack_name][eps].append(rec)

                line = ('%6s  %6.3f  %5d  %9.4f  %8.4f  %10.4f' %
                        (attack_name, eps, trial+1,
                         acc_before, acc_after, att_success))
                if is_targeted:
                    line += '  %10.4f' % trate
                line += ('  %10.2e  %10.2e  %10.2e  %8.4f  %5.1fs' %
                         (bers['bpsk'], bers['16qam'], bers['64qam'],
                          loss, elapsed))
                print(line)
        print()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results, is_targeted):
    agg = {}
    for att in results:
        agg[att] = {}
        for eps, trials in results[att].items():
            a = dict(
                acc_before    = np.mean([t['acc_before']     for t in trials]),
                acc_after     = np.mean([t['acc_after']      for t in trials]),
                acc_after_std = np.std( [t['acc_after']      for t in trials]),
                attack_success= np.mean([t['attack_success'] for t in trials]),
                loss          = np.mean([t['loss']           for t in trials]),
                **{f'ber_{m}': np.mean([t[f'ber_{m}'] for t in trials])
                   for m in MODULATIONS}
            )
            if is_targeted:
                a['target_rate']     = np.mean([t['target_rate'] for t in trials])
                a['target_rate_std'] = np.std( [t['target_rate'] for t in trials])
            agg[att][eps] = a
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(agg, args):
    is_targeted = (args.mode == 'targeted')
    print('\n' + '='*90)
    print('SUMMARY — Mean over trials')
    print('='*90)
    print('%-6s  %7s  %10s  %10s  %10s  %10s  %10s  %10s  %10s' %
          ('Attack', 'Epsilon', 'AccBefore', 'AccAfter', 'AttSucc',
           'BER-BPSK', 'BER-QPSK', 'BER-16QAM', 'BER-64QAM')
          + ('  %10s' % 'TargetRate' if is_targeted else ''))
    print('-'*90)
    for att in ['fgsm', 'pgd']:
        for eps in sorted(agg[att].keys()):
            a = agg[att][eps]
            print('%-6s  %7.4f  %10.4f  %10.4f  %10.4f  %10.2e  %10.2e  %10.2e  %10.2e' %
                  (att, eps, a['acc_before'], a['acc_after'], a['attack_success'],
                   a['ber_bpsk'], a['ber_qpsk'], a['ber_16qam'], a['ber_64qam'])
                  + ('  %10.4f' % a['target_rate'] if is_targeted else ''))
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def save_plots(agg, args, output_dir):
    eps_list    = sorted(agg['fgsm'].keys())
    is_targeted = (args.mode == 'targeted')
    mode_tag    = (f'Targeted→Dev{args.target_label}') \
                  if is_targeted else 'Untargeted'

    def arr(att, key):
        return np.array([agg[att][e][key] for e in eps_list])

    # ── 1. Attack success per attack type ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for att in ['fgsm', 'pgd']:
        succ = arr(att, 'attack_success')
        ax.plot(eps_list, succ, 'o-', color=ATT_COLORS[att], lw=2,
                label=att.upper())
        if is_targeted:
            ax.plot(eps_list, arr(att, 'target_rate'), 's--',
                    color=ATT_COLORS[att], lw=1.5, alpha=0.7,
                    label=f'{att.upper()} target rate')
    ax.set_xlabel('ε', fontsize=12)
    ax.set_ylabel('Attack Success Rate (1 − Accuracy)', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f'SARP [{mode_tag}] — Attack Success vs ε')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(output_dir, f'attack_success.{ext}'), dpi=150)
    plt.close(fig)

    # ── 2. BER — all four modulations ────────────────────────────────────
    # BER depends only on epsilon, not on the attack type
    fig, ax = plt.subplots(figsize=(8, 5))
    mod_styles = [
        ('bpsk',  'BPSK',  '-',  'o'),
        ('qpsk',  'QPSK',  '--', 's'),
        ('16qam', '16QAM', '-.', '^'),
        ('64qam', '64QAM', ':',  'D'),
    ]
    for mod, label, ls, mk in mod_styles:
        ber_vals = np.array([agg['fgsm'][e][f'ber_{mod}'] for e in eps_list])
        ber_vals = np.clip(ber_vals, 1e-20, None)
        ax.plot(eps_list, ber_vals, ls, color=MOD_COLORS[mod],
                marker=mk, markersize=6, lw=2.2, label=label)
    ax.axhline(WIFI_BER, color='red', ls='--', lw=1.8,
               label='WiFi BER limit (10⁻⁵)')
    # Annotate crossover epsilons
    for label, eps_x, col in [('BPSK ε≈0.57',  0.574, MOD_COLORS['bpsk']),
                               ('QPSK ε≈0.41',  0.406, MOD_COLORS['qpsk']),
                               ('16QAM ε≈0.19', 0.193, MOD_COLORS['16qam']),
                               ('64QAM ε≈0.10', 0.097, MOD_COLORS['64qam'])]:
        if min(eps_list) <= eps_x <= max(eps_list):
            ax.axvline(eps_x, color=col, ls=':', lw=1, alpha=0.6)
    ax.set_yscale('log')
    ax.set_ylim(1e-20, 1.5)
    ax.set_xlabel('Perturbation Bound ε', fontsize=12)
    ax.set_ylabel('Theoretical BER (Q-function)', fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')
    ax.set_title('Theoretical BER vs ε — 802.11 Modulations\n'
                 '(Uniform perturbation modelled as AWGN, σ²=ε²/3)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(output_dir, f'ber_vs_epsilon.{ext}'), dpi=150)
    plt.close(fig)

    # ── 3. Tradeoff: attack success vs BER (per modulation) ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, att in zip(axes, ['fgsm', 'pgd']):
        succ = arr(att, 'attack_success')
        for mod, label, ls, mk in mod_styles:
            ber_v = np.clip(
                np.array([agg[att][e][f'ber_{mod}'] for e in eps_list]),
                1e-20, None)
            sc = ax.scatter(ber_v, succ, c=eps_list, cmap='plasma',
                            s=60, zorder=4, marker=mk)
            ax.plot(ber_v, succ, ls, color=MOD_COLORS[mod],
                    lw=1.8, alpha=0.7, label=label)
        ax.axvline(WIFI_BER, color='red', ls='--', lw=1.5,
                   label='WiFi BER limit')
        ax.set_xlabel('BER (log scale)', fontsize=11)
        ax.set_ylabel('Attack Success Rate', fontsize=11)
        ax.set_xscale('log'); ax.set_xlim(1e-20, 2.0)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'{att.upper()} — Stealthiness Tradeoff [{mode_tag}]')
        ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
        plt.colorbar(sc, ax=ax, label='ε')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(output_dir, f'tradeoff.{ext}'), dpi=150)
    plt.close(fig)

    # ── 4. Before vs after bar chart at key epsilon ───────────────────────
    key_eps = max([e for e in eps_list if e <= 0.3], default=eps_list[-2])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, att in zip(axes, ['fgsm', 'pgd']):
        before = agg[att][0.0]['acc_before'] if 0.0 in agg[att] \
                 else arr(att, 'acc_before')[0]
        after  = agg[att][key_eps]['acc_after']
        bars   = ax.bar(['Before attack', f'After {att.upper()}\nε={key_eps}'],
                        [before, after],
                        color=['steelblue', 'tomato'], alpha=0.85, width=0.4)
        for bar, val in zip(bars, [before, after]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
        if is_targeted:
            trate = agg[att][key_eps].get('target_rate', 0)
            ax.bar(['Target rate'], [trate], color='orange', alpha=0.8,
                   width=0.4)
        ax.axhline(1/args.num_classes, color='grey', ls=':', lw=1,
                   label='Chance level (1/7)')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Accuracy / Rate')
        ax.set_title(f'{att.upper()} — Class {args.device_label} [{mode_tag}]')
        ax.legend(fontsize=9); ax.grid(alpha=0.2, axis='y')
    fig.suptitle(f'Accuracy Before vs After Attack at ε={key_eps}', fontsize=13)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(output_dir, f'before_after_bar.{ext}'), dpi=150)
    plt.close(fig)

    print('  [Saved] attack_success, ber_vs_epsilon, tradeoff, before_after_bar')


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(results, output_dir, is_targeted):
    path = os.path.join(output_dir, 'results.csv')
    cols = ('attack,epsilon,trial,acc_before,acc_after,attack_success,'
            'ber_bpsk,ber_qpsk,ber_16qam,ber_64qam,loss,time_s,pred_dist')
    if is_targeted:
        cols += ',target_rate'
    with open(path, 'w') as f:
        f.write(cols + '\n')
        for att in results:
            for eps, trials in results[att].items():
                for ti, t in enumerate(trials):
                    row = ('%s,%.4f,%d,%.6f,%.6f,%.6f,'
                           '%.6e,%.6e,%.6e,%.6e,%.6f,%.2f,"%s"' %
                           (att, eps, ti,
                            t['acc_before'], t['acc_after'], t['attack_success'],
                            t['ber_bpsk'], t['ber_qpsk'],
                            t['ber_16qam'], t['ber_64qam'],
                            t['loss'], t['time'], t['pred_dist']))
                    if is_targeted:
                        row += ',%.6f' % (t.get('target_rate') or 0)
                    f.write(row + '\n')
    print(f'  [Saved] {path}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights',      required=True)
    p.add_argument('--npz',          required=True)
    p.add_argument('--device-label', type=int, required=True)
    p.add_argument('--num-classes',  type=int, default=7)
    p.add_argument('--mode',
                   choices=['untargeted', 'targeted'], default='untargeted')
    p.add_argument('--target-label', type=int, default=None)
    p.add_argument('--epsilon',      type=float, nargs='+',
                   default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    p.add_argument('--pgd-steps',    type=int,   default=200)
    p.add_argument('--trials',       type=int,   default=5)
    p.add_argument('--n-traces',     type=int,   default=500)
    p.add_argument('--batch-size',   type=int,   default=100)
    p.add_argument('--output-dir',   default='./attack_results')
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == 'targeted' and args.target_label is None:
        raise ValueError('--target-label is required for --mode targeted')
    if args.mode == 'targeted' and args.target_label == args.device_label:
        raise ValueError('--target-label must differ from --device-label')

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print('='*80)
    print('attack_iq2ch.py  —  Feature-Domain Adversarial Attack on SARP')
    print('='*80)
    print(f'  Mode         : {args.mode.upper()}'
          + (f' → target class {args.target_label}' if args.mode == 'targeted'
             else ''))
    print(f'  Victim class : {args.device_label}')
    print(f'  Attacks      : FGSM, PGD-{args.pgd_steps}')
    print(f'  Epsilons     : {args.epsilon}')
    print(f'  Trials       : {args.trials}')

    model  = load_model(args.weights)
    traces = load_traces(args.npz, args.device_label, n=args.n_traces)

    results = run(args, model, traces)
    agg     = aggregate(results, args.mode == 'targeted')

    print('\nGenerating plots ...')
    save_plots(agg, args, args.output_dir)
    save_csv(results, args.output_dir, args.mode == 'targeted')
    print_summary(agg, args)
    print(f'\nAll results → {args.output_dir}/')


if __name__ == '__main__':
    main()

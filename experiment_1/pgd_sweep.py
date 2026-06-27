#!/usr/bin/env python3
import numpy as np, os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings; warnings.filterwarnings('ignore')
import tensorflow as tf
from sarp_attack_engine import SARPModel

model = SARPModel('models/sarp_cvnn_5ghz_v3/extracted/model.weights.h5', num_classes=7)
tx = np.fromfile('data/raw_iq/device_5/tx_device5.bin', dtype=np.complex64)

PREAMBLE = 400; OFDM_SYM = 80; N_OFDM = 177
SARP_FFT = 64; TRACE_LEN = 288
frame_iq = tx[PREAMBLE:PREAMBLE + N_OFDM * OFDM_SYM]
n_chunks = len(frame_iq) // SARP_FFT
n_traces = (n_chunks * SARP_FFT) // TRACE_LEN

print("Fine-grained PGD sweep — stealth regime")
print("200 PGD steps, small epsilon")
header = f"{'Epsilon':>8}  {'Acc':>6}  {'LogitGap':>10}  {'PertPower':>12}  {'BER':>8}  WiFi   Predictions"
print(header)
print("-" * 80)

for eps in [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]:
    alpha = eps / 20  # smaller step size for more precision
    n_steps = 200
    pert_r = np.real(frame_iq).astype(np.float32).copy()
    pert_i = np.imag(frame_iq).astype(np.float32).copy()
    orig_r = pert_r.copy()
    orig_i = pert_i.copy()
    labels = tf.constant(np.full(n_traces, 4, dtype=np.int32))

    for step in range(n_steps):
        iq_r = tf.Variable(pert_r)
        iq_i = tf.Variable(pert_i)
        with tf.GradientTape() as tape:
            iq_c = tf.complex(iq_r, iq_i)
            chunks = tf.reshape(iq_c[:n_chunks*SARP_FFT], [n_chunks, SARP_FFT])
            fft = tf.signal.fft(chunks)
            shifted = tf.concat([fft[:, 32:], fft[:, :32]], axis=1)
            stream = tf.reshape(shifted, [-1])
            tr = tf.reshape(stream[:n_traces*TRACE_LEN], [n_traces, TRACE_LEN, 1])
            logits = model(tr, training=False)
            bs = tf.shape(logits)[0]
            correct = tf.gather_nd(logits, tf.stack([tf.range(bs), labels[:bs]], axis=1))
            mask = tf.one_hot(labels[:bs], 7)
            other = tf.reduce_max(logits - mask*1e9, axis=1)
            loss = tf.reduce_mean(other - correct)
        gr, gi = tape.gradient(loss, [iq_r, iq_i])
        if gr is None:
            break
        pert_r = pert_r + alpha * np.sign(gr.numpy())
        pert_i = pert_i + alpha * np.sign(gi.numpy())
        pert_r = np.clip(pert_r, orig_r - eps, orig_r + eps)
        pert_i = np.clip(pert_i, orig_i - eps, orig_i + eps)

    pert_iq = (pert_r + 1j * pert_i).astype(np.complex64)
    ch = pert_iq[:n_chunks*SARP_FFT].reshape(n_chunks, SARP_FFT)
    ff = np.fft.fft(ch)
    sh = np.concatenate([ff[:, 32:], ff[:, :32]], axis=1)
    st = sh.flatten()
    tr2 = st[:n_traces*TRACE_LEN].reshape(n_traces, TRACE_LEN, 1)
    logits_np = model(tf.constant(tr2), training=False).numpy()
    preds = np.argmax(logits_np, axis=1)
    acc = np.mean(preds == 4)
    gap = np.mean(logits_np[:, 4]) - np.max(np.mean(logits_np[:, [0,1,2,3,5,6]], axis=0))
    pp = np.mean(np.abs(pert_iq - frame_iq)**2) / np.mean(np.abs(frame_iq)**2)

    orig_bits = (np.real(frame_iq) > 0).astype(int)
    pert_bits = (np.real(pert_iq) > 0).astype(int)
    ber = np.mean(orig_bits != pert_bits)

    pred_dist = np.bincount(preds, minlength=7)
    wifi_ok = "OK" if ber < 1e-5 else "FAIL"
    print(f"{eps:>8.3f}  {acc:>5.1%}  {gap:>10.2f}  {pp:>12.2e}  {ber:>8.6f}  [{wifi_ok}]  {pred_dist}")

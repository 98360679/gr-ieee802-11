"""
Standalone model loader for SARP v3.
Imports build_sarp_cvnn directly from train_sarp_cvnn.py (already in your cwd),
extracts weights from the .keras zip, loads them by name.
"""
import sys, os, zipfile, tempfile
import numpy as np
import tensorflow as tf

# ── 1. Import builder directly from the training script ──────────────────────
sys.path.insert(0, '.')
from train_sarp_cvnn import build_sarp_cvnn

# ── 2. Extract weights.h5 from the .keras zip ────────────────────────────────
def extract_weights(keras_path):
    with zipfile.ZipFile(keras_path, 'r') as z:
        wfile = next(n for n in z.namelist() if n.endswith('.weights.h5'))
        tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        tmp.write(z.read(wfile)); tmp.flush(); tmp.close()
    return tmp.name

# ── 3. Build + load ───────────────────────────────────────────────────────────
def load_sarp_model(keras_path, num_classes=7):
    model = build_sarp_cvnn(num_classes=num_classes)
    weights_h5 = extract_weights(keras_path)
    model.load_weights(weights_h5, by_name=True, skip_mismatch=True)
    os.unlink(weights_h5)

    # Sanity: softmax over 7 classes should sum to ~1 per sample
    dummy = tf.zeros((2, 288, 1), dtype=tf.complex64)
    out = model(dummy, training=False).numpy()
    print(f"[OK] Model loaded. Dummy output sum={out.sum():.4f} (expect 2.0)")
    return model

if __name__ == '__main__':
    import argparse
    from collections import Counter

    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='models/sarp_cvnn_5ghz_v3/best_model.keras')
    p.add_argument('--npz',     default='data/sarp_fft_7devices_signal_only.npz')
    args = p.parse_args()

    model = load_sarp_model(args.weights)

    d = np.load(args.npz)
    X, y = d['X'].astype(np.complex64), d['y'].astype(int)
    if y.min() == 1: y -= 1
    if X.ndim == 2:  X = X[:, :, np.newaxis]

    print('\nClass   N    TopPred   TopAcc    Confusion')
    for c in range(7):
        idx = np.where(y == c)[0][:100]
        traces = X[idx].copy()
        rms = np.sqrt(np.mean(np.abs(traces)**2, axis=1, keepdims=True))
        traces /= np.maximum(rms, 1e-10)
        preds = np.argmax(model(traces, training=False).numpy(), axis=1)
        cnt = Counter(preds.tolist())
        top = cnt.most_common(1)[0]
        print('  %d    %d    %d         %.0f%%      %s' %
              (c, len(idx), top[0], 100*top[1]/len(idx), dict(cnt)))

#!/usr/bin/env python3
"""
SARP CVNN Training Script
===========================
Trains a Complex-Valued Neural Network for WiFi device RF fingerprinting.

Architecture:
  Input(288, 1, complex64)
  4× [ComplexConv1D → ComplexBatchNorm → CReLU → ComplexAvgPool1D(2)]
     filters: [32, 64, 128, 256], kernel_size=3, padding=same
     (no AvgPool after last block)
  ComplexGlobalAveragePooling1D
  Concat(real, imag) → 512 float features
  Dense(256, relu) → BatchNorm → Dropout(0.3) → Dense(num_classes, softmax)

Usage:
    python3 train_sarp_cvnn.py \
        --data data/sarp_fft_7devices_5k.npz \
        --output-dir models/sarp_cvnn_5ghz/ \
        --epochs 100 \
        --batch-size 64 \
        --lr 0.001

    # With custom train/val/test split:
    python3 train_sarp_cvnn.py \
        --data data/sarp_fft_7devices_5k.npz \
        --output-dir models/sarp_cvnn_5ghz/ \
        --val-split 0.15 \
        --test-split 0.15
"""

import numpy as np
import os
import sys
import json
import argparse
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks


# ══════════════════════════════════════════════════════════════════════
# Custom Complex-Valued Layers
# ══════════════════════════════════════════════════════════════════════

class ComplexConv1D(layers.Layer):
    """
    Complex-valued 1D convolution.
    (r + j*i) ⊛ (Wrr + j*Wri) = (r⊛Wrr - i⊛Wii) + j*(r⊛Wri + i⊛Wir)
    """
    def __init__(self, filters, kernel_size, strides=1, padding='same',
                 kernel_regularizer=None, **kwargs):
        super().__init__(dtype='complex64', **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        reg = self.kernel_regularizer
        a = dict(strides=self.strides, padding=self.padding,
                 kernel_regularizer=reg)
        self.conv_rr = layers.Conv1D(self.filters, self.kernel_size, **a)
        self.conv_ri = layers.Conv1D(self.filters, self.kernel_size, **a)
        self.conv_ir = layers.Conv1D(self.filters, self.kernel_size, **a)
        self.conv_ii = layers.Conv1D(self.filters, self.kernel_size, **a)
        super().build(input_shape)

    def call(self, x):
        x = tf.cast(x, tf.complex64)
        r = tf.math.real(x)
        i = tf.math.imag(x)
        return tf.complex(
            self.conv_rr(r) - self.conv_ii(i),
            self.conv_ri(r) + self.conv_ir(i)
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'kernel_regularizer': keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
        })
        return config


class ComplexBatchNorm(layers.Layer):
    """Applies separate BatchNorm to real and imaginary parts."""
    def __init__(self, **kwargs):
        super().__init__(dtype='complex64', **kwargs)

    def build(self, input_shape):
        self.bn_real = layers.BatchNormalization()
        self.bn_imag = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, x, training=None):
        x = tf.cast(x, tf.complex64)
        return tf.complex(
            self.bn_real(tf.math.real(x), training=training),
            self.bn_imag(tf.math.imag(x), training=training)
        )


class CReLU(layers.Layer):
    """Applies ReLU independently to real and imaginary parts."""
    def __init__(self, **kwargs):
        super().__init__(dtype='complex64', **kwargs)

    def call(self, x):
        x = tf.cast(x, tf.complex64)
        return tf.complex(
            tf.nn.relu(tf.math.real(x)),
            tf.nn.relu(tf.math.imag(x))
        )


class ComplexAvgPooling1D(layers.Layer):
    """Average pooling applied separately to real and imaginary parts."""
    def __init__(self, pool_size=2, **kwargs):
        super().__init__(dtype='complex64', **kwargs)
        self.pool_size = pool_size
        self.pool = layers.AveragePooling1D(pool_size)

    def call(self, x):
        x = tf.cast(x, tf.complex64)
        return tf.complex(
            self.pool(tf.math.real(x)),
            self.pool(tf.math.imag(x))
        )

    def get_config(self):
        config = super().get_config()
        config.update({'pool_size': self.pool_size})
        return config


# ══════════════════════════════════════════════════════════════════════
# Model Builder
# ══════════════════════════════════════════════════════════════════════

def build_sarp_cvnn(input_length=288, num_classes=7, l2_reg=1e-4,
                    dropout_rate=0.3):
    """
    Build the SARP CVNN model.

    Architecture matches the original training script exactly:
      4 complex conv blocks + global avg pool + 2 dense layers.
    """
    filters_list = [32, 64, 128, 256]
    kernel_size = 3
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    inp = layers.Input(shape=(input_length, 1), dtype='complex64')
    x = inp

    for i, filters in enumerate(filters_list):
        x = ComplexConv1D(filters, kernel_size, padding='same',
                          kernel_regularizer=reg, name=f'complex_conv1d{"" if i==0 else f"_{i}"}')(x)
        x = ComplexBatchNorm(name=f'complex_batch_norm{"" if i==0 else f"_{i}"}')(x)
        x = CReLU(name=f'c_re_lu{"" if i==0 else f"_{i}"}')(x)

        # Average pooling after all blocks except the last
        if i < len(filters_list) - 1:
            x = ComplexAvgPooling1D(pool_size=2,
                                     name=f'complex_avg_pooling1d{"" if i==0 else f"_{i}"}')(x)

    # Complex Global Average Pooling
    x = tf.reduce_mean(x, axis=1)  # (B, C) complex64

    # Split complex → real: concat(real, imag) → (B, 2*C)
    x = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)

    # Dense head
    x = layers.Dense(256, activation='relu', name='dense')(x)
    x = layers.BatchNormalization(name='batch_normalization')(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(num_classes, activation='softmax', name='dense_1')(x)

    model = keras.Model(inputs=inp, outputs=out, name='sarp_cvnn')
    return model


# ══════════════════════════════════════════════════════════════════════
# Data Loading & Preprocessing
# ══════════════════════════════════════════════════════════════════════

def load_data(npz_path, val_split=0.15, test_split=0.15, seed=42,
              no_normalize=False):
    """
    Load .npz and split into train/val/test.
    Labels in .npz are 1-7; we keep them as-is (the model outputs 7 classes,
    indexed 0-6, but we use sparse_categorical_crossentropy which handles
    label values 1-7 if num_classes > max(label)).

    NOTE: If labels are 1-7, we need num_classes=8 OR remap to 0-6.
    We remap to 0-6 for clean indexing.
    """
    print(f"Loading {npz_path}...")
    d = np.load(npz_path)
    X = d['X']
    y = d['y']

    print(f"  Raw: X={X.shape} ({X.dtype}), y={y.shape}")
    print(f"  Labels: {sorted(np.unique(y))}")

    # Remap labels to 0-indexed if needed
    unique_labels = sorted(np.unique(y))
    if min(unique_labels) > 0:
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y_mapped = np.array([label_map[l] for l in y], dtype=np.int32)
        print(f"  Remapped labels: {unique_labels} → {list(range(len(unique_labels)))}")
    else:
        y_mapped = y.astype(np.int32)
        label_map = {i: i for i in unique_labels}

    num_classes = len(unique_labels)

    # Add channel dimension if needed: (N, 288) → (N, 288, 1)
    if X.ndim == 2:
        X = X[:, :, np.newaxis]

    # Ensure complex64
    X = X.astype(np.complex64)

    # RMS normalize each trace (skip if data is already raw/unnormalized intentionally)
    if not no_normalize:
        rms = np.sqrt(np.mean(np.abs(X)**2, axis=1, keepdims=True))
        rms = np.maximum(rms, 1e-10)
        X = X / rms
        print(f"  Applied RMS normalization")
    else:
        print(f"  Skipping normalization (preserving raw power)")
        power = np.mean(np.abs(X)**2, axis=1)
        print(f"  Power range: [{np.min(power):.6e}, {np.max(power):.6e}]")

    # Shuffle
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    X, y_mapped = X[idx], y_mapped[idx]

    # Split
    n = len(X)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    X_train, y_train = X[:n_train], y_mapped[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y_mapped[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y_mapped[n_train+n_val:]

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {num_classes}")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            num_classes, label_map, unique_labels)


# ══════════════════════════════════════════════════════════════════════
# Evaluation & Plotting
# ══════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, num_classes, label_names,
                   output_dir, batch_size=256):
    """Run full evaluation with confusion matrix and per-class accuracy."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Predict in batches
    all_preds = []
    all_probs = []
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        probs = model(batch, training=False).numpy()
        all_probs.append(probs)
        all_preds.append(np.argmax(probs, axis=1))

    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)

    # Overall accuracy
    acc = np.mean(preds == y_test)
    print(f"\n  Test Accuracy: {acc:.4f} ({int(acc*len(y_test))}/{len(y_test)})")

    # Per-class accuracy
    print(f"\n  Per-class accuracy:")
    per_class_acc = {}
    for c in range(num_classes):
        mask = y_test == c
        if np.sum(mask) > 0:
            c_acc = np.mean(preds[mask] == c)
            per_class_acc[c] = c_acc
            print(f"    Device {label_names[c]}: {c_acc:.4f} "
                  f"({int(c_acc*np.sum(mask))}/{np.sum(mask)})")

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true, pred in zip(y_test, preds):
        cm[true, pred] += 1

    # ── Plot 1: Confusion Matrix ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    tick_labels = [f'Dev {label_names[i]}' for i in range(num_classes)]
    ax.set(xticks=range(num_classes), yticks=range(num_classes),
           xticklabels=tick_labels, yticklabels=tick_labels,
           ylabel='True Device', xlabel='Predicted Device',
           title=f'Confusion Matrix (Accuracy: {acc:.1%})')

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    fig.savefig(os.path.join(output_dir, 'confusion_matrix.pdf'), dpi=150)
    plt.close(fig)
    print(f"  [Saved] confusion_matrix.png/.pdf")

    # ── Plot 2: Per-class accuracy bar chart ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = sorted(per_class_acc.keys())
    accs = [per_class_acc[c] for c in classes]
    colors = ['#2ca02c' if a > 0.9 else '#ff7f0e' if a > 0.7 else '#d62728'
              for a in accs]

    bars = ax.bar(range(len(classes)), [a*100 for a in accs],
                  color=colors, alpha=0.85, edgecolor='white')
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{a:.1%}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f'Device {label_names[c]}' for c in classes])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Per-Device Classification Accuracy (Overall: {acc:.1%})')
    ax.set_ylim(0, 110)
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90%')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'per_class_accuracy.png'), dpi=150)
    fig.savefig(os.path.join(output_dir, 'per_class_accuracy.pdf'), dpi=150)
    plt.close(fig)
    print(f"  [Saved] per_class_accuracy.png/.pdf")

    return acc, per_class_acc, cm


def plot_training_history(history, output_dir):
    """Plot training curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle('SARP CVNN Training History', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    fig.savefig(os.path.join(output_dir, 'training_history.pdf'), dpi=150)
    plt.close(fig)
    print(f"  [Saved] training_history.png/.pdf")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train SARP CVNN for RF Fingerprinting')

    # Data
    parser.add_argument('--data', required=True,
                        help='Path to .npz file (X: complex64, y: int labels)')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test-split', type=float, default=0.15,
                        help='Test split ratio (default: 0.15)')

    # Architecture
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (auto-detected if None)')
    parser.add_argument('--l2-reg', type=float, default=1e-4,
                        help='L2 regularization weight (default: 1e-4)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--lr-patience', type=int, default=7,
                        help='LR reduction patience (default: 7)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate (default: 1e-6)')

    # Output
    parser.add_argument('--output-dir', default='models/sarp_cvnn/',
                        help='Output directory for model and plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Skip RMS normalization (use when data has meaningful power differences)')

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SARP CVNN Training")
    print(f"{'='*60}")

    (X_train, y_train, X_val, y_val, X_test, y_test,
     num_classes, label_map, original_labels) = load_data(
        args.data, args.val_split, args.test_split, args.seed,
        no_normalize=args.no_normalize)

    if args.num_classes is not None:
        num_classes = args.num_classes

    input_length = X_train.shape[1]

    # ── Build model ───────────────────────────────────────────────────
    print(f"\nBuilding model...")
    model = build_sarp_cvnn(
        input_length=input_length,
        num_classes=num_classes,
        l2_reg=args.l2_reg,
        dropout_rate=args.dropout,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Print summary
    model.summary()

    total_params = model.count_params()
    print(f"\n  Total parameters: {total_params:,}")

    # ── Callbacks ─────────────────────────────────────────────────────
    cb = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
            mode='max',
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=args.lr_patience,
            min_lr=args.min_lr,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max',
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"lr={args.lr}")
    print(f"  Early stopping: patience={args.patience} (val_accuracy)")
    print(f"  LR reduction: patience={args.lr_patience}, factor=0.5")
    print(f"{'='*60}\n")

    t0 = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        verbose=1,
    )

    train_time = time.time() - t0
    print(f"\n  Training time: {train_time:.1f}s "
          f"({train_time/60:.1f} min)")

    # ── Save model and weights ────────────────────────────────────────
    print(f"\nSaving model...")

    # Save full Keras model
    model.save(os.path.join(args.output_dir, 'cvnn_sarp_model.keras'))
    print(f"  [Saved] cvnn_sarp_model.keras")

    # Save weights only (for sarp_attack_engine.py compatibility)
    weights_dir = os.path.join(args.output_dir, 'extracted')
    os.makedirs(weights_dir, exist_ok=True)
    model.save_weights(os.path.join(weights_dir, 'model.weights.h5'))
    print(f"  [Saved] extracted/model.weights.h5")

    # ── Evaluate ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Evaluation on Test Set")
    print(f"{'='*60}")

    # Map back to original label names for display
    inv_map = {v: k for k, v in label_map.items()}
    label_names = {i: str(inv_map[i]) for i in range(num_classes)}

    acc, per_class_acc, cm = evaluate_model(
        model, X_test, y_test, num_classes, label_names,
        args.output_dir)

    # ── Plot training history ─────────────────────────────────────────
    plot_training_history(history.history, args.output_dir)

    # ── Save training config and results ──────────────────────────────
    results = {
        'args': vars(args),
        'num_classes': num_classes,
        'input_length': input_length,
        'label_map': {str(k): int(v) for k, v in label_map.items()},
        'original_labels': [int(l) for l in original_labels],
        'total_params': int(total_params),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'test_accuracy': float(acc),
        'per_class_accuracy': {str(k): float(v) for k, v in per_class_acc.items()},
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'best_val_loss': float(min(history.history['val_loss'])),
        'epochs_trained': len(history.history['loss']),
        'training_time_s': float(train_time),
    }

    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  [Saved] training_results.json")

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Test accuracy:     {acc:.4f} ({acc:.1%})")
    print(f"  Best val accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"  Epochs trained:    {len(history.history['loss'])}")
    print(f"  Training time:     {train_time:.1f}s")
    print(f"  Model saved to:    {args.output_dir}/")
    print(f"")
    print(f"  To use with sarp_attack_engine.py:")
    print(f"    --weights {os.path.join(weights_dir, 'model.weights.h5')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

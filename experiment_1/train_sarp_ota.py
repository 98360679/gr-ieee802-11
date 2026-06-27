"""
train_sarp_ota.py
=================
Train a CVNN (Complex-Valued Neural Network) for SARP RF fingerprinting
on the newly collected OTA data.

Architecture (matches original SARP CVNN):
  Input(288, 1, complex64)
  4x [ComplexConv1D -> ComplexBatchNorm -> CReLU -> ComplexAvgPooling1D(2)]
     filters: [32, 64, 128, 256], kernel_size=3, L2=1e-4
     (no AvgPool after last block)
  GlobalAveragePooling1D
  Lambda: concat(real, imag) -> 512 real features
  Dense(256, relu, L2=1e-4) -> BatchNormalization -> Dropout(0.3)
  Dense(num_classes, softmax)

Usage:
  python3 train_sarp_ota.py \
      --data /media/nghoselab/T9/Data/ota_processed/wifi_7devices_ota_new.npz \
      --output /media/nghoselab/T9/Data/models/sarp_ota_cvnn/ \
      --epochs 100 \
      --batch-size 256

"""

import argparse
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Custom complex-valued layers
# CRITICAL: dtype='complex64' on all complex layers prevents Keras from
# auto-casting complex64 inputs to float32, which would silently discard
# imaginary parts and destroy the fingerprint signal.
# ─────────────────────────────────────────────────────────────────────────────

class ComplexConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same',
                 kernel_regularizer=None, **kwargs):
        kwargs['dtype'] = 'complex64'
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        cfg = dict(strides=self.strides, padding=self.padding,
                   kernel_regularizer=self.kernel_regularizer)
        self.conv_rr = layers.Conv1D(self.filters, self.kernel_size, **cfg)
        self.conv_ri = layers.Conv1D(self.filters, self.kernel_size, **cfg)
        self.conv_ir = layers.Conv1D(self.filters, self.kernel_size, **cfg)
        self.conv_ii = layers.Conv1D(self.filters, self.kernel_size, **cfg)
        super().build(input_shape)

    def call(self, x):
        x = tf.cast(x, tf.complex64)
        r, i = tf.math.real(x), tf.math.imag(x)
        return tf.complex(self.conv_rr(r) - self.conv_ii(i),
                          self.conv_ri(r) + self.conv_ir(i))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self.filters, 'kernel_size': self.kernel_size,
                    'strides': self.strides, 'padding': self.padding,
                    'kernel_regularizer': None})
        return cfg


class ComplexBatchNorm(layers.Layer):
    def __init__(self, **kwargs):
        kwargs['dtype'] = 'complex64'
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.bn_real = layers.BatchNormalization()
        self.bn_imag = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, x, training=None):
        x = tf.cast(x, tf.complex64)
        return tf.complex(
            self.bn_real(tf.math.real(x), training=training),
            self.bn_imag(tf.math.imag(x), training=training))

    def get_config(self):
        return super().get_config()


class CReLU(layers.Layer):
    def __init__(self, **kwargs):
        kwargs['dtype'] = 'complex64'
        super().__init__(**kwargs)

    def call(self, x):
        x = tf.cast(x, tf.complex64)
        return tf.complex(tf.nn.relu(tf.math.real(x)),
                          tf.nn.relu(tf.math.imag(x)))

    def get_config(self):
        return super().get_config()


class ComplexAvgPooling1D(layers.Layer):
    def __init__(self, pool_size=2, **kwargs):
        kwargs['dtype'] = 'complex64'
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.pool = layers.AveragePooling1D(pool_size)

    def call(self, x):
        x = tf.cast(x, tf.complex64)
        return tf.complex(self.pool(tf.math.real(x)),
                          self.pool(tf.math.imag(x)))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'pool_size': self.pool_size})
        return cfg


class ComplexGlobalAvgPooling1D(layers.Layer):
    def __init__(self, **kwargs):
        kwargs['dtype'] = 'complex64'
        super().__init__(**kwargs)

    def call(self, x):
        x = tf.cast(x, tf.complex64)
        return tf.complex(tf.reduce_mean(tf.math.real(x), axis=1),
                          tf.reduce_mean(tf.math.imag(x), axis=1))

    def get_config(self):
        return super().get_config()


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

def build_sarp_cvnn(input_length=288, num_classes=7,
                    l2_reg=1e-4, dropout_rate=0.3):
    """
    Build SARP CVNN model.

    Returns a compiled Keras model ready for training.
    """
    l2 = regularizers.l2(l2_reg)
    filters_list = [32, 64, 128, 256]

    inp = layers.Input(shape=(input_length, 1), dtype='complex64', name='input')
    x = inp

    for i, filters in enumerate(filters_list):
        x = ComplexConv1D(filters, kernel_size=3, padding='same',
                          kernel_regularizer=l2,
                          name=f'complex_conv1d{"" if i==0 else f"_{i}"}')(x)
        x = ComplexBatchNorm(
                name=f'complex_batch_norm{"" if i==0 else f"_{i}"}')(x)
        x = CReLU(name=f'c_re_lu{"" if i==0 else f"_{i}"}')(x)
        if i < len(filters_list) - 1:
            x = ComplexAvgPooling1D(pool_size=2,
                name=f'complex_avg_pooling1d{"" if i==0 else f"_{i}"}')(x)

    # Global average pooling then concat real+imag → real feature vector
    x = ComplexGlobalAvgPooling1D(name='complex_global_avg_pooling1d')(x)
    x = layers.Lambda(
            lambda z: tf.concat([tf.math.real(z), tf.math.imag(z)], axis=-1),
            name='lambda')(x)

    # Dense head
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=l2, name='dense')(x)
    x = layers.BatchNormalization(name='batch_normalization')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    out = layers.Dense(num_classes, activation='softmax', name='dense_1')(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name='sarp_ota_cvnn')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(npz_path, val_split=0.15, test_split=0.15, seed=42):
    print(f"Loading data from {npz_path} ...")
    data = np.load(npz_path)
    X = data['X']   # (N, 288) complex64
    y = data['y']   # (N,) int32

    print(f"  X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  y shape: {y.shape}, classes: {np.unique(y)}")
    print(f"  Samples per class: { {int(c): int(np.sum(y==c)) for c in np.unique(y)} }")

    # Add channel dim: (N, 288, 1) complex64
    X = X[:, :, np.newaxis]

    # Split: train / val / test
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_split + test_split,
        stratify=y, random_state=seed)
    relative_test = test_split / (val_split + test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=relative_test,
        stratify=y_tmp, random_state=seed)

    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history, output_dir):
    """Save loss and accuracy curves (train + val) as a single figure."""
    epochs = range(1, len(history.history['loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss ──
    ax1.plot(epochs, history.history['loss'],     'b-o', markersize=3,
             label='Train loss')
    ax1.plot(epochs, history.history['val_loss'], 'r-o', markersize=3,
             label='Val loss')
    ax1.set_title('Loss vs Epoch', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Accuracy ──
    train_acc = [a * 100 for a in history.history['accuracy']]
    val_acc   = [a * 100 for a in history.history['val_accuracy']]
    ax2.plot(epochs, train_acc, 'b-o', markersize=3, label='Train accuracy')
    ax2.plot(epochs, val_acc,   'r-o', markersize=3, label='Val accuracy')
    ax2.set_title('Accuracy vs Epoch', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    best_val_acc = max(val_acc)
    best_epoch   = val_acc.index(best_val_acc) + 1
    ax2.axvline(best_epoch, color='green', linestyle='--', alpha=0.6,
                label=f'Best val ({best_val_acc:.1f}% @ ep {best_epoch})')
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {path}")


def plot_confusion_matrix(y_true, y_pred, num_classes, output_dir):
    """Save a normalised confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label='Recall (%)')

    labels = [f'Dev {i+1}' for i in range(num_classes)]
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix (Recall %)', fontsize=13, fontweight='bold')

    thresh = 50.0
    for i in range(num_classes):
        for j in range(num_classes):
            color = 'white' if cm_norm[i, j] > thresh else 'black'
            ax.text(j, i, f'{cm_norm[i, j]:.1f}',
                    ha='center', va='center', fontsize=9, color=color)

    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train SARP OTA CVNN')
    parser.add_argument('--data', type=str,
        default='/media/nghoselab/T9/Data/ota_processed/wifi_7devices_ota_new.npz')
    parser.add_argument('--output', type=str,
        default='/media/nghoselab/T9/Data/models/sarp_ota_cvnn/')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--test-split', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Register custom layers globally so .keras files can be reloaded
    # without passing custom_objects= every time.
    tf.keras.utils.get_custom_objects().update({
        'ComplexConv1D':             ComplexConv1D,
        'ComplexBatchNorm':          ComplexBatchNorm,
        'CReLU':                     CReLU,
        'ComplexAvgPooling1D':       ComplexAvgPooling1D,
        'ComplexGlobalAvgPooling1D': ComplexGlobalAvgPooling1D,
    })

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(
        args.data, args.val_split, args.test_split, args.seed)

    num_classes = len(np.unique(y_train))
    print(f"\nNum classes: {num_classes}")

    # Build model
    model = build_sarp_cvnn(
        input_length=288,
        num_classes=num_classes,
        l2_reg=args.l2,
        dropout_rate=args.dropout)
    model.summary()

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    # Callbacks
    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1),
        callbacks.CSVLogger(
            os.path.join(args.output, 'training_log.csv')),
    ]

    # Train
    print(f"\nTraining for up to {args.epochs} epochs ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb_list,
        verbose=1)

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    test_loss, test_acc = model.evaluate(X_test, y_test,
                                          batch_size=args.batch_size,
                                          verbose=0)
    print(f"Test accuracy: {test_acc*100:.2f}%")
    print(f"Test loss:     {test_loss:.4f}")

    # Per-class accuracy
    y_pred = np.argmax(model.predict(X_test, batch_size=args.batch_size,
                                      verbose=0), axis=1)
    print("\nPer-class accuracy:")
    for c in range(num_classes):
        mask = y_test == c
        acc_c = np.mean(y_pred[mask] == c)
        print(f"  Device {c+1} (class {c}): {acc_c*100:.2f}%  ({mask.sum()} samples)")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_training_history(history, args.output)
    plot_confusion_matrix(y_test, y_pred, num_classes, args.output)

    # ── Save model ──────────────────────────────────────────────────────────
    # We ONLY use .keras format (never .h5 weights) because TF's h5 weight
    # loader reconstructs layers from scratch and silently casts complex64
    # weights to float32, destroying imaginary parts.
    # .keras format serializes the full model config + weights together,
    # preserving all dtypes exactly as trained.
    # ────────────────────────────────────────────────────────────────────────
    final_model_path = os.path.join(args.output, 'final_model.keras')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Best model saved to:  {os.path.join(args.output, 'best_model.keras')}")

    # Also dump weights as .npz (numpy) for use by sarp_attack_engine.py.
    # numpy preserves complex64 exactly — no TF dtype coercion involved.
    weights_npz_path = os.path.join(args.output, 'model_weights.npz')
    weight_dict = {}
    for layer in model.layers:
        for w in layer.weights:
            # Use the weight name as key, replacing '/' with '_' for npz compat
            key = w.name.replace('/', '_').replace(':', '_')
            arr = w.numpy()
            weight_dict[key] = arr
    np.savez(weights_npz_path, **weight_dict)
    print(f"Numpy weights saved to: {weights_npz_path} "
          f"({len(weight_dict)} tensors, dtypes preserved as-is)")


if __name__ == '__main__':
    main()

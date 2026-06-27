#!/usr/bin/env python3
"""
Improved CVNN Training for RF Fingerprinting
- Data augmentation (phase rotation, noise injection)
- Batch normalization
- Multiple architecture options (small/medium)
- Class balancing
- Configurable trace length for fair comparison with SARP

v3: Added cvnn_small and real_iq architectures to address overfitting
"""

import numpy as np
import argparse
import os
import json
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# Data Augmentation for RF Signals
# ============================================================================

class RFDataAugmentation:
    """Data augmentation techniques for RF signals."""
    
    def __init__(self, phase_range=np.pi/4, noise_std=0.01, 
                 time_shift_range=5, freq_offset_range=0.01,
                 snr_range=(10, 30)):
        self.phase_range = phase_range
        self.noise_std = noise_std
        self.time_shift_range = time_shift_range
        self.freq_offset_range = freq_offset_range
        self.snr_range = snr_range
    
    def random_phase_rotation(self, x):
        """Apply random phase rotation."""
        phase = np.random.uniform(-self.phase_range, self.phase_range)
        return x * np.exp(1j * phase)
    
    def add_noise(self, x):
        """Add small Gaussian noise."""
        noise = (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)) * self.noise_std
        return x + noise.astype(np.complex64)
    
    def add_snr_noise(self, x):
        """Add noise at random SNR level (more realistic than fixed std)."""
        snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])
        signal_power = np.mean(np.abs(x) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise_std = np.sqrt(noise_power / 2)
        noise = (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)) * noise_std
        return x + noise.astype(np.complex64)
    
    def time_shift(self, x):
        """Random circular time shift."""
        shift = np.random.randint(-self.time_shift_range, self.time_shift_range + 1)
        return np.roll(x, shift, axis=-1)
    
    def frequency_offset(self, x):
        """Apply small frequency offset."""
        offset = np.random.uniform(-self.freq_offset_range, self.freq_offset_range)
        t = np.arange(x.shape[-1])
        return x * np.exp(1j * 2 * np.pi * offset * t)
    
    def amplitude_scale(self, x):
        """Random amplitude scaling (±10%)."""
        scale = np.random.uniform(0.9, 1.1)
        return x * scale
    
    def augment(self, x, augmentations=['phase', 'snr_noise']):
        """Apply selected augmentations."""
        x_aug = x.copy()
        
        if 'phase' in augmentations:
            x_aug = self.random_phase_rotation(x_aug)
        if 'noise' in augmentations:
            x_aug = self.add_noise(x_aug)
        if 'snr_noise' in augmentations:
            x_aug = self.add_snr_noise(x_aug)
        if 'time_shift' in augmentations:
            x_aug = self.time_shift(x_aug)
        if 'freq_offset' in augmentations:
            x_aug = self.frequency_offset(x_aug)
        if 'amplitude' in augmentations:
            x_aug = self.amplitude_scale(x_aug)
        
        return x_aug.astype(np.complex64)


def augment_dataset(X, y, augmentation_factor=2,
                    augmentations=['phase', 'snr_noise', 'time_shift']):
    """
    Augment dataset by generating additional samples.
    """
    aug = RFDataAugmentation()
    
    X_aug_list = [X]  # Include original
    y_aug_list = [y]
    
    for _ in range(augmentation_factor):
        X_new = np.array([aug.augment(x, augmentations) for x in X])
        X_aug_list.append(X_new)
        y_aug_list.append(y)
    
    X_aug = np.concatenate(X_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    
    # Shuffle
    idx = np.random.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]


# ============================================================================
# Complex-Valued Layers
# ============================================================================

class ComplexConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same',
                 kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_regularizer = kernel_regularizer
        
    def build(self, input_shape):
        self.conv_rr = layers.Conv1D(self.filters, self.kernel_size, 
                                      strides=self.strides, padding=self.padding,
                                      kernel_regularizer=self.kernel_regularizer)
        self.conv_ri = layers.Conv1D(self.filters, self.kernel_size, 
                                      strides=self.strides, padding=self.padding,
                                      kernel_regularizer=self.kernel_regularizer)
        self.conv_ir = layers.Conv1D(self.filters, self.kernel_size, 
                                      strides=self.strides, padding=self.padding,
                                      kernel_regularizer=self.kernel_regularizer)
        self.conv_ii = layers.Conv1D(self.filters, self.kernel_size, 
                                      strides=self.strides, padding=self.padding,
                                      kernel_regularizer=self.kernel_regularizer)
        super().build(input_shape)
        
    def call(self, inputs):
        real = tf.math.real(inputs)
        imag = tf.math.imag(inputs)
        real_out = self.conv_rr(real) - self.conv_ii(imag)
        imag_out = self.conv_ri(real) + self.conv_ir(imag)
        return tf.complex(real_out, imag_out)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters, 'kernel_size': self.kernel_size,
            'strides': self.strides, 'padding': self.padding,
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer)
            if self.kernel_regularizer else None,
        })
        return config


class ComplexBatchNorm(layers.Layer):
    """Batch normalization for complex values."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.bn_real = layers.BatchNormalization()
        self.bn_imag = layers.BatchNormalization()
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        real = self.bn_real(tf.math.real(inputs), training=training)
        imag = self.bn_imag(tf.math.imag(inputs), training=training)
        return tf.complex(real, imag)


class ComplexAvgPooling1D(layers.Layer):
    def __init__(self, pool_size=2, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.pool = layers.AveragePooling1D(pool_size)
        
    def call(self, inputs):
        real = self.pool(tf.math.real(inputs))
        imag = self.pool(tf.math.imag(inputs))
        return tf.complex(real, imag)

    def get_config(self):
        config = super().get_config()
        config.update({'pool_size': self.pool_size})
        return config


class CReLU(layers.Layer):
    def call(self, inputs):
        real = tf.nn.relu(tf.math.real(inputs))
        imag = tf.nn.relu(tf.math.imag(inputs))
        return tf.complex(real, imag)


class ComplexDropout(layers.Layer):
    """Dropout for complex values."""
    def __init__(self, rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
    
    def call(self, inputs, training=None):
        if training:
            mask = tf.nn.dropout(tf.ones_like(tf.math.real(inputs)), self.rate)
            return inputs * tf.complex(mask, tf.zeros_like(mask))
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate})
        return config


# ============================================================================
# Model Architectures
# ============================================================================

def build_cvnn_small(input_shape, num_classes, dropout_rate=0.4):
    """
    Small CVNN -- fewer filters, L2 regularization, complex dropout.
    Designed for short traces (288) to avoid overfitting.
    """
    l2 = keras.regularizers.l2(1e-4)

    model = keras.Sequential([
        keras.layers.Input(shape=input_shape, dtype=tf.complex64),

        # Block 1: 16 filters
        ComplexConv1D(16, 7, padding='same', kernel_regularizer=l2),
        ComplexBatchNorm(),
        CReLU(),
        ComplexAvgPooling1D(2),
        ComplexDropout(dropout_rate * 0.5),

        # Block 2: 32 filters
        ComplexConv1D(32, 5, padding='same', kernel_regularizer=l2),
        ComplexBatchNorm(),
        CReLU(),
        ComplexAvgPooling1D(2),
        ComplexDropout(dropout_rate * 0.5),

        # Block 3: 64 filters
        ComplexConv1D(64, 3, padding='same', kernel_regularizer=l2),
        ComplexBatchNorm(),
        CReLU(),

        # Global Average Pooling
        keras.layers.GlobalAveragePooling1D(),

        # Convert complex to real (64 complex -> 128 real)
        keras.layers.Lambda(lambda x: tf.concat(
            [tf.math.real(x), tf.math.imag(x)], axis=-1)),

        # Single dense layer
        keras.layers.Dense(64, activation='relu',
                           kernel_regularizer=l2),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model


def build_cvnn_medium(input_shape, num_classes, dropout_rate=0.3):
    """
    Medium CVNN with ~500K parameters.
    Balance between capacity and generalization.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape, dtype=tf.complex64),
        
        # Block 1: 32 filters
        ComplexConv1D(32, 7, padding='same'),
        ComplexBatchNorm(),
        CReLU(),
        ComplexAvgPooling1D(2),
        
        # Block 2: 64 filters
        ComplexConv1D(64, 5, padding='same'),
        ComplexBatchNorm(),
        CReLU(),
        ComplexAvgPooling1D(2),
        
        # Block 3: 128 filters
        ComplexConv1D(128, 5, padding='same'),
        ComplexBatchNorm(),
        CReLU(),
        ComplexAvgPooling1D(2),
        
        # Block 4: 256 filters
        ComplexConv1D(256, 3, padding='same'),
        ComplexBatchNorm(),
        CReLU(),
        
        # Global Average Pooling instead of Flatten
        keras.layers.GlobalAveragePooling1D(),
        
        # Convert complex to real (256 complex -> 512 real)
        keras.layers.Lambda(lambda x: tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)),
        
        # Dense layers
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),
        
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),
        
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def build_cvnn_stable(input_shape, num_classes, dropout_rate=0.5):
    """
    Stabilized CVNN -- designed for smooth training curves.
    3 conv blocks (32->64->128) with L2 reg + complex dropout.
    ~150K params -- appropriate for ~5K training traces.
    """
    l2 = keras.regularizers.l2(1e-3)

    model = keras.Sequential([
        keras.layers.Input(shape=input_shape, dtype=tf.complex64),

        # Block 1: 32 filters
        ComplexConv1D(32, 7, padding='same', kernel_regularizer=l2),
        ComplexBatchNorm(),
        CReLU(),
        ComplexAvgPooling1D(2),
        ComplexDropout(dropout_rate * 0.3),

        # Block 2: 64 filters
        ComplexConv1D(64, 5, padding='same', kernel_regularizer=l2),
        ComplexBatchNorm(),
        CReLU(),
        ComplexAvgPooling1D(2),
        ComplexDropout(dropout_rate * 0.3),

        # Block 3: 128 filters
        ComplexConv1D(128, 3, padding='same', kernel_regularizer=l2),
        ComplexBatchNorm(),
        CReLU(),

        # Global Average Pooling
        keras.layers.GlobalAveragePooling1D(),

        # Convert complex to real (128 complex -> 256 real)
        keras.layers.Lambda(lambda x: tf.concat(
            [tf.math.real(x), tf.math.imag(x)], axis=-1)),

        # Single dense layer
        keras.layers.Dense(128, activation='relu',
                           kernel_regularizer=l2),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model


def build_cvnn_sarp(input_shape, num_classes, dropout_rate=0.3):
    """
    SARP-matched CVNN architecture (Afrin et al., CCNC 2025, Fig. 4).
    4 conv blocks (32->64->128->256) with CReLU + AvgPooling.
    Matches SARP paper: ComplexInput -> 4 blocks -> ComplexFlatten -> ComplexDense.
    ~500K params -- designed for larger sliding-window datasets.
    """
    l2 = keras.regularizers.l2(1e-4)

    model = keras.Sequential([
        keras.layers.Input(shape=input_shape, dtype=tf.complex64),

        # Block 1: 32 filters
        ComplexConv1D(32, 3, padding='same', kernel_regularizer=l2),
        ComplexBatchNorm(),
        CReLU(),
        ComplexAvgPooling1D(2),

        # Block 2: 64 filters
        ComplexConv1D(64, 3, padding='same', kernel_regularizer=l2),
        ComplexBatchNorm(),
        CReLU(),
        ComplexAvgPooling1D(2),

        # Block 3: 128 filters
        ComplexConv1D(128, 3, padding='same', kernel_regularizer=l2),
        ComplexBatchNorm(),
        CReLU(),
        ComplexAvgPooling1D(2),

        # Block 4: 256 filters
        ComplexConv1D(256, 3, padding='same', kernel_regularizer=l2),
        ComplexBatchNorm(),
        CReLU(),

        # Global Average Pooling
        keras.layers.GlobalAveragePooling1D(),

        # Convert complex to real (256 complex -> 512 real)
        keras.layers.Lambda(lambda x: tf.concat(
            [tf.math.real(x), tf.math.imag(x)], axis=-1)),

        # Dense head
        keras.layers.Dense(256, activation='relu',
                           kernel_regularizer=l2),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model



def build_real_iq(input_shape, num_classes, dropout_rate=0.4):
    """
    Real-valued CNN that splits complex IQ into 2 channels (I, Q).
    Simpler, more stable training than complex-valued approach.
    """
    l2 = keras.regularizers.l2(1e-4)
    trace_len = input_shape[0]

    # Input is complex64 (trace_len, 1) -- split to real (trace_len, 2)
    inp = keras.layers.Input(shape=input_shape, dtype=tf.complex64)
    real_part = keras.layers.Lambda(
        lambda x: tf.math.real(x))(inp)
    imag_part = keras.layers.Lambda(
        lambda x: tf.math.imag(x))(inp)
    x = keras.layers.Concatenate(axis=-1)([real_part, imag_part])
    # x shape: (trace_len, 2)

    # Block 1
    x = layers.Conv1D(32, 7, padding='same', kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)

    # Block 2
    x = layers.Conv1D(64, 5, padding='same', kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)

    # Block 3
    x = layers.Conv1D(64, 3, padding='same', kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    out = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inp, outputs=out)
    return model


def _resnet_block(x, filters, kernel_size, strides=1, l2_reg=None,
                  dropout_rate=0.0, downsample=False):
    """Single residual block with pre-activation (BN -> ReLU -> Conv)."""
    shortcut = x

    # Pre-activation
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Conv 1
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same',
                       kernel_regularizer=l2_reg, use_bias=False)(x)

    # BN + ReLU + Conv 2
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same',
                       kernel_regularizer=l2_reg, use_bias=False)(x)

    # Match dimensions for shortcut
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=strides, padding='same',
                                  kernel_regularizer=l2_reg, use_bias=False)(shortcut)

    x = layers.Add()([x, shortcut])
    return x


def build_resnet(input_shape, num_classes, dropout_rate=0.3):
    """
    Compact ResNet for RF fingerprinting.
    Pre-activation residual blocks, designed for short IQ/preamble traces.
    ~40K params -- small enough for ~200 traces/class with augmentation.
    """
    l2 = keras.regularizers.l2(1e-3)

    inp = keras.layers.Input(shape=input_shape, dtype=tf.complex64)

    # Split complex -> 2-channel real
    real_part = keras.layers.Lambda(lambda x: tf.math.real(x))(inp)
    imag_part = keras.layers.Lambda(lambda x: tf.math.imag(x))(inp)
    x = keras.layers.Concatenate(axis=-1)([real_part, imag_part])

    # Initial conv (wider kernel to capture preamble structure)
    x = layers.Conv1D(32, 11, strides=1, padding='same',
                       kernel_regularizer=l2, use_bias=False)(x)

    # ResNet blocks -- gradually increase filters, downsample
    x = _resnet_block(x, 32, 7, strides=1, l2_reg=l2,
                       dropout_rate=dropout_rate * 0.3)
    x = _resnet_block(x, 32, 7, strides=2, l2_reg=l2,
                       dropout_rate=dropout_rate * 0.3, downsample=True)

    x = _resnet_block(x, 64, 5, strides=1, l2_reg=l2,
                       dropout_rate=dropout_rate * 0.5)
    x = _resnet_block(x, 64, 5, strides=2, l2_reg=l2,
                       dropout_rate=dropout_rate * 0.5, downsample=True)

    x = _resnet_block(x, 128, 3, strides=1, l2_reg=l2,
                       dropout_rate=dropout_rate * 0.5)
    x = _resnet_block(x, 128, 3, strides=2, l2_reg=l2,
                       dropout_rate=dropout_rate * 0.5, downsample=True)

    # Final BN + ReLU (pre-activation style)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=l2)(x)
    x = layers.Dropout(dropout_rate)(x)

    out = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inp, outputs=out)
    return model


# ============================================================================
# Training
# ============================================================================



class WarmupCosineDecay(keras.callbacks.Callback):
    """
    Linear warmup + cosine annealing LR schedule.
    Produces much smoother training curves than ReduceLROnPlateau.
    """
    def __init__(self, max_lr, warmup_epochs, total_epochs, min_lr=1e-6):
        super().__init__()
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup: 0 -> max_lr
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay: max_lr -> min_lr
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(
            self.model.optimizer.learning_rate))
        if logs is not None:
            logs['lr'] = lr


def load_data(filepath):
    print(f"Loading {filepath}")
    data = np.load(filepath)
    X = data['X']
    y = data['y']
    sessions = data['sessions'] if 'sessions' in data else None
    print(f"  {len(X)} samples, {len(np.unique(y))} classes")
    if sessions is not None:
        print(f"  Sessions: {sorted(np.unique(sessions))}")
    else:
        print(f"  WARNING: No session info -- will use random split")
    return X, y, sessions


def normalize_iq(X, mode='rms'):
    """Normalize complex I/Q data.
    
    Args:
        X: complex array (N, trace_length)
        mode: 'rms' (unit power, default) or 'per_sample' (max) or 'global' (global max)
    """
    if mode == 'rms':
        # RMS normalization -- unit power per trace
        # Preserves waveform shape and phase, removes gross power differences
        rms = np.sqrt(np.mean(np.abs(X) ** 2, axis=1, keepdims=True))
        rms[rms == 0] = 1
        return X / rms
    elif mode == 'global':
        # Global normalization -- preserves relative amplitudes
        global_max = np.abs(X).max()
        if global_max == 0:
            global_max = 1
        return X / global_max
    else:
        # Per-sample normalization
        magnitudes = np.abs(X).max(axis=1, keepdims=True)
        magnitudes[magnitudes == 0] = 1
        return X / magnitudes


def prepare_data(X, y, sessions=None, test_size=0.2, val_size=0.1,
                 trace_length=None, norm_mode='rms', test_session=3):
    """
    Prepare data with proper session-based or random splitting.
    
    If sessions array is provided, uses session-based splitting:
      - Train: sessions != test_session
      - Test: session == test_session
      - Val: split from training data
    Otherwise falls back to random stratified split.
    """
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_mapped = np.array([label_map[label] for label in y])
    
    # Normalize I/Q
    X = normalize_iq(X, mode=norm_mode)
    print(f"Normalization: {norm_mode}")
    
    # Adjust trace length if specified
    if trace_length is not None:
        current_length = X.shape[1]
        if current_length > trace_length:
            print(f"Truncating traces from {current_length} to {trace_length}")
            X = X[:, :trace_length]
        elif current_length < trace_length:
            print(f"Padding traces from {current_length} to {trace_length}")
            pad_width = trace_length - current_length
            X = np.pad(X, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    
    if sessions is not None and test_session in np.unique(sessions):
        # ── Session-based split ──
        print(f"\nSession-based split: test=session {test_session}, "
              f"train=sessions {sorted(s for s in np.unique(sessions) if s != test_session)}")
        
        test_mask = sessions == test_session
        train_mask = ~test_mask
        
        X_train_full = X[train_mask]
        y_train_full = y_mapped[train_mask]
        X_test = X[test_mask]
        y_test = y_mapped[test_mask]
        
        # Split validation from training data
        val_fraction = val_size / (1 - test_size)  # ~12.5% of train
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=val_fraction, random_state=42, stratify=y_train_full
        )
        
        print(f"  Train sessions data: {len(X_train_full)}")
        print(f"  Test session {test_session} data: {len(X_test)}")
        
        # Print per-device counts in test set
        for i in range(len(unique_labels)):
            cnt = np.sum(y_test == i)
            print(f"    Device {unique_labels[i]} test samples: {cnt}")
    else:
        # ── Random split (fallback) ──
        print("\nUsing random stratified split (no session info)")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_mapped, test_size=test_size, random_state=42, stratify=y_mapped
        )
        
        val_fraction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=42, stratify=y_temp
        )
    
    # Reshape for Conv1D
    X_train = X_train.reshape(-1, X_train.shape[1], 1)
    X_val = X_val.reshape(-1, X_val.shape[1], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], 1)
    
    print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"Trace length: {X_train.shape[1]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, len(unique_labels), label_map


def plot_results(history, y_test, y_pred, num_classes, output_dir, label_map=None):
    """Plot training history and confusion matrices."""
    
    if label_map is not None:
        reverse_map = {v: k for k, v in label_map.items()}
        target_names = [f'Device {reverse_map[i]}' for i in range(num_classes)]
    else:
        target_names = [f'Device {i+1}' for i in range(num_classes)]
    
    # Figure 1: Training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    plt.close()
    
    test_acc = np.mean(y_pred == y_test) * 100
    
    # Confusion Matrix (raw counts)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix (Test Accuracy: {test_acc:.2f}%)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # Normalized Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=100)
    
    plt.title(f'Normalized Confusion Matrix (Test Accuracy: {test_acc:.2f}%)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--augment_factor', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--class_weights', action='store_true', help='Use class weights')
    parser.add_argument('--trace_length', type=int, default=None, 
                        help='Trace length (default: use original, set 288 for SARP comparison)')
    parser.add_argument('--model', type=str, default='cvnn_small',
                        choices=['cvnn_small', 'cvnn_medium', 'cvnn_stable', 'cvnn_sarp', 'real_iq', 'resnet'],
                        help='Model architecture (default: cvnn_small)')
    parser.add_argument('--norm', type=str, default='rms',
                        choices=['rms', 'per_sample', 'global'],
                        help='Normalization mode (default: rms)')
    parser.add_argument('--test_session', type=int, default=3,
                        help='Session number for test set (default: 3)')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print(f"CVNN Training ({args.model})")
    print("=" * 60)
    if args.trace_length:
        print(f"Trace length: {args.trace_length}")
    else:
        print(f"Trace length: original")
    print(f"Normalization: {args.norm}")
    
    # Load data
    X, y, sessions = load_data(args.data)
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes, label_map = prepare_data(
        X, y, sessions=sessions, trace_length=args.trace_length,
        norm_mode=args.norm, test_session=args.test_session
    )
    
    # Data augmentation on training set only
    if args.augment:
        print(f"\nApplying data augmentation (factor={args.augment_factor})...")
        X_train_flat = X_train.squeeze()
        X_train_aug, y_train_aug = augment_dataset(
            X_train_flat, y_train, 
            augmentation_factor=args.augment_factor,
            augmentations=['phase', 'snr_noise', 'time_shift']
        )
        X_train = X_train_aug.reshape(-1, X_train_aug.shape[1], 1)
        y_train = y_train_aug
        print(f"  Augmented training set: {len(X_train)} samples")
    
    # Convert to complex64
    X_train = X_train.astype(np.complex64)
    X_val = X_val.astype(np.complex64)
    X_test = X_test.astype(np.complex64)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Build model
    print(f"\nBuilding model: {args.model}")
    if args.model == 'cvnn_small':
        model = build_cvnn_small(input_shape, num_classes, args.dropout)
    elif args.model == 'cvnn_medium':
        model = build_cvnn_medium(input_shape, num_classes, args.dropout)
    elif args.model == 'cvnn_stable':
        model = build_cvnn_stable(input_shape, num_classes, args.dropout)
    elif args.model == 'cvnn_sarp':
        model = build_cvnn_sarp(input_shape, num_classes, args.dropout)
    elif args.model == 'real_iq':
        model = build_real_iq(input_shape, num_classes, args.dropout)
    elif args.model == 'resnet':
        model = build_resnet(input_shape, num_classes, args.dropout)
    
    model.summary()
    
    # Class weights
    class_weights = None
    if args.class_weights:
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(weights))
        print(f"Class weights: {class_weights}")
    
    # Compile
    if args.model == 'cvnn_stable':
        # Label smoothing reduces overconfidence and stabilizes training
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False)
        # Convert labels for label smoothing manually
        y_train_smooth = keras.utils.to_categorical(y_train, num_classes)
        y_val_smooth = keras.utils.to_categorical(y_val, num_classes)
        y_train_smooth = y_train_smooth * 0.9 + 0.1 / num_classes
        y_val_smooth = y_val_smooth * 0.9 + 0.1 / num_classes
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        use_label_smoothing = True
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        use_label_smoothing = False
    
    # Callbacks
    if args.model in ('cvnn_sarp',):
        # Warmup + cosine annealing for smooth training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=30, restore_best_weights=True
            ),
            WarmupCosineDecay(
                max_lr=args.lr,
                warmup_epochs=5,
                total_epochs=args.epochs,
                min_lr=1e-6
            )
        ]
    elif args.model == 'cvnn_stable':
        # Cosine decay is smoother than ReduceLROnPlateau
        total_steps = (len(y_train) // args.batch_size + 1) * args.epochs
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=25, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6
            )
        ]
    else:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            )
        ]
    
    # Train
    print("\n=== Training ===")
    if use_label_smoothing:
        history = model.fit(
            X_train, y_train_smooth,
            validation_data=(X_val, y_val_smooth),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
    
    # Evaluate
    print("\n=== Evaluation ===")
    if use_label_smoothing:
        y_test_eval = keras.utils.to_categorical(y_test, num_classes)
        test_loss, test_acc = model.evaluate(X_test, y_test_eval, verbose=0)
    else:
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    # Classification report
    reverse_map = {v: k for k, v in label_map.items()}
    target_names = [f'Device {reverse_map[i]}' for i in range(num_classes)]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Per-class accuracy
    print("\n=== Per-Device Accuracy ===")
    for i in range(num_classes):
        mask = y_test == i
        acc = np.mean(y_pred[mask] == y_test[mask])
        print(f"Device {reverse_map[i]}: {acc*100:.2f}%")
    
    # Save results
    plot_results(history, y_test, y_pred, num_classes, args.output, label_map)
    
    # Save model (with fallback to weights-only if serialization fails)
    model_path = os.path.join(args.output, f'{args.model}_model.keras')
    try:
        model.save(model_path)
        print(f"Model saved: {model_path}")
    except (NotImplementedError, TypeError):
        weights_path = os.path.join(args.output, f'{args.model}_weights.h5')
        model.save_weights(weights_path)
        print(f"Model weights saved: {weights_path} (full save failed, weights only)")
    
    results = {
        'model': args.model,
        'norm_mode': args.norm,
        'trace_length': args.trace_length if args.trace_length else 'original',
        'test_accuracy': float(test_acc),
        'augmentation': args.augment,
        'class_weights': args.class_weights,
        'epochs_trained': len(history.history['loss']),
        'args': vars(args)
    }
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"FINAL TEST ACCURACY: {test_acc*100:.2f}%")
    print(f"Model: {args.model}")
    print(f"Normalization: {args.norm}")
    print(f"Trace length: {args.trace_length if args.trace_length else 'original'}")
    print(f"{'='*60}")
    
    # SARP comparison
    print(f"\n{'='*60}")
    print("Comparison with SARP Paper (same-location):")
    print(f"{'='*60}")
    print(f"  SARP reported:    78.45%")
    print(f"  Your result:      {100*test_acc:.2f}%")
    diff = test_acc * 100 - 78.45
    if diff > 0:
        print(f"  Difference:       +{diff:.2f}% (better)")
    else:
        print(f"  Difference:       {diff:.2f}%")


if __name__ == "__main__":
    main()

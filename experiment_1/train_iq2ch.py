#!/usr/bin/env python3
"""
train_iq2ch.py
==============
Train a standard float32 CNN on 2-channel IQ data.
No complex arithmetic, no dtype issues.

Input:  X (N, 288, 2) float32,  y (N,) int 0-6
Output: models/sarp_iq2ch/best_model.keras

Usage:
    python3 train_iq2ch.py \
        --npz  data/sarp_iq2ch.npz \
        --output-dir models/sarp_iq2ch \
        --epochs 100 \
        --batch-size 256
"""
import os, argparse, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def build_iq2ch_model(input_length=288, num_classes=7,
                      filters=(32, 64, 128, 256),
                      kernel_size=3, dropout=0.3, l2=1e-4):
    """
    Standard float32 1-D CNN matching the SARP architecture.
    Input: (batch, 288, 2) — channel 0 = I, channel 1 = Q.
    This is functionally equivalent to ComplexConv1D but in plain float32.
    """
    reg = keras.regularizers.l2(l2)
    inp = layers.Input(shape=(input_length, 2), dtype='float32', name='iq_input')
    x = inp
    for i, f in enumerate(filters):
        name = '' if i == 0 else f'_{i}'
        x = layers.Conv1D(f, kernel_size, padding='same',
                          use_bias=False, kernel_regularizer=reg,
                          name=f'conv1d{name}')(x)
        x = layers.BatchNormalization(name=f'bn{name}')(x)
        x = layers.ReLU(name=f'relu{name}')(x)
        x = layers.AveragePooling1D(pool_size=2, name=f'pool{name}')(x)

    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dense(256, activation='relu', name='dense')(x)
    x = layers.BatchNormalization(name='bn_head')(x)
    x = layers.Dropout(dropout, name='dropout')(x)
    out = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return keras.Model(inp, out, name='sarp_iq2ch')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',         required=True)
    p.add_argument('--output-dir',  default='models/sarp_iq2ch')
    p.add_argument('--epochs',      type=int, default=100)
    p.add_argument('--batch-size',  type=int, default=256)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--num-classes', type=int, default=7)
    p.add_argument('--val-split',   type=float, default=0.15)
    p.add_argument('--test-split',  type=float, default=0.15)
    p.add_argument('--seed',        type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {args.npz} ...")
    d = np.load(args.npz)
    X, y = d['X'].astype(np.float32), d['y'].astype(np.int64)
    assert X.ndim == 3 and X.shape[2] == 2, \
        f"Expected (N,288,2), got {X.shape}. Run convert_iq2ch.py first."
    print(f"  X: {X.shape}  y: {y.shape}  classes: {sorted(set(y.tolist()))}")

    # ── Train/val/test split ──────────────────────────────────────────────────
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=args.val_split + args.test_split,
        stratify=y, random_state=args.seed)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=args.test_split / (args.val_split + args.test_split),
        stratify=y_tmp, random_state=args.seed)

    print(f"  Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_te)}")

    # ── Build & compile ───────────────────────────────────────────────────────
    model = build_iq2ch_model(num_classes=args.num_classes)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, 'best_model.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10,
            min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20,
            restore_best_weights=True, verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\nTest accuracy: {acc*100:.2f}%  (loss={loss:.4f})")

    # Confusion matrix
    preds = np.argmax(model.predict(X_te, verbose=0), axis=1)
    from sklearn.metrics import classification_report
    print(classification_report(y_te, preds,
                                 target_names=[f'Dev{i+1}' for i in range(args.num_classes)]))

    # Save results
    results = {
        'test_accuracy': float(acc),
        'test_loss': float(loss),
        'num_classes': args.num_classes,
        'input_shape': [288, 2],
        'architecture': 'sarp_iq2ch',
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    model.save(os.path.join(args.output_dir, 'final_model.keras'))
    print(f"\nSaved to {args.output_dir}/")


if __name__ == '__main__':
    main()

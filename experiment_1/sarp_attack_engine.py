"""
SARP Attack Engine
====================
Handles SARP model loading and adversarial perturbation computation.
Runs in a SEPARATE PROCESS to avoid LLVM conflicts between TF and GNU Radio.

PURE-TF IMPLEMENTATION: This version loads weights from h5 directly into
tf.Variables and runs forward passes using raw TF ops (tf.nn.conv1d, etc.).
No Keras layers are used in the forward path, which completely avoids
Keras 3's complex64→float32 auto-casting bug.

Usage standalone test:
    python3 sarp_attack_engine.py --weights /path/to/extracted/model.weights.h5 --test

Usage from GRC (automatic):
    The embedded block spawns this with --serve mode.

Place this file at: ~/research/fingerprinting/sarp_attack_engine.py
"""

import numpy as np
import sys
import os
import struct

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SARPModel:
    """
    Pure-TF SARP CVNN model — NO Keras layers.

    Architecture:
      Input(288, 1, complex64)
      4x [ComplexConv1D → ComplexBatchNorm → CReLU → ComplexAvgPool1D]
         filters: [32, 64, 128, 256], kernel_size=3
         (no AvgPool after last block)
      ComplexGlobalAveragePooling1D
      Concat(real, imag) → 512 float features
      Dense(256, relu) → BatchNorm → Dropout(0.3) → Dense(num_classes, softmax)
    """

    def __init__(self, weights_path, num_classes=7):
        import tensorflow as tf
        self.tf = tf
        self.num_classes = num_classes
        self.weights = {}
        self._load_weights(weights_path)

    def _load_weights(self, path):
        """Load all weights from h5 into plain numpy arrays."""
        import h5py

        def get_vars(f, h5_path):
            grp = f[f'{h5_path}/vars']
            return [grp[str(i)][:] for i in range(len(grp))]

        with h5py.File(path, 'r') as f:
            w = self.weights
            filters = [32, 64, 128, 256]

            for i in range(4):
                suffix = '' if i == 0 else f'_{i}'
                prefix = f'layers/complex_conv1d{suffix}'

                # Conv: kernel (k, in_ch, out_ch), bias (out_ch)
                for sub in ['conv_rr', 'conv_ri', 'conv_ir', 'conv_ii']:
                    vals = get_vars(f, f'{prefix}/{sub}')
                    w[f'conv{i}_{sub}_w'] = vals[0]  # kernel
                    w[f'conv{i}_{sub}_b'] = vals[1]  # bias

                # BatchNorm: gamma, beta, moving_mean, moving_var
                bn_prefix = f'layers/complex_batch_norm{suffix}'
                for part in ['bn_real', 'bn_imag']:
                    vals = get_vars(f, f'{bn_prefix}/{part}')
                    w[f'cbn{i}_{part}_gamma'] = vals[0]
                    w[f'cbn{i}_{part}_beta'] = vals[1]
                    w[f'cbn{i}_{part}_mean'] = vals[2]
                    w[f'cbn{i}_{part}_var'] = vals[3]

            # Dense(256): kernel (512, 256), bias (256)
            vals = get_vars(f, 'layers/dense')
            w['dense_w'] = vals[0]
            w['dense_b'] = vals[1]

            # BatchNorm after Dense: gamma, beta, mean, var
            vals = get_vars(f, 'layers/batch_normalization')
            w['bn_gamma'] = vals[0]
            w['bn_beta'] = vals[1]
            w['bn_mean'] = vals[2]
            w['bn_var'] = vals[3]

            # Dense(num_classes): kernel (256, num_classes), bias
            vals = get_vars(f, 'layers/dense_1')
            w['dense1_w'] = vals[0]
            w['dense1_b'] = vals[1]

        # Convert to tf.constant for use in graph
        tf = self.tf
        self.W = {k: tf.constant(v) for k, v in self.weights.items()}

        print(f"[SARP Model] Weights loaded from {path}", file=sys.stderr)
        print(f"[SARP Model] Conv blocks: 4, Dense: 256 → {self.num_classes}",
              file=sys.stderr)

    @property
    def input_shape(self):
        return (None, 288, 1)

    @property
    def output_shape(self):
        return (None, self.num_classes)

    def __call__(self, x, training=False):
        """
        Forward pass using pure TF ops.
        x: complex64 tensor of shape (batch, 288, 1)
        Returns: float32 logits of shape (batch, num_classes)
        """
        tf = self.tf
        W = self.W
        eps_bn = 1e-3  # Keras default BN epsilon

        x = tf.cast(x, tf.complex64)

        for i in range(4):
            # ── ComplexConv1D ──
            r = tf.math.real(x)  # float32
            im = tf.math.imag(x)  # float32

            # Complex multiply: (r + j*im) * (Wrr + j*Wri) etc.
            out_rr = tf.nn.conv1d(r, W[f'conv{i}_conv_rr_w'], stride=1,
                                  padding='SAME') + W[f'conv{i}_conv_rr_b']
            out_ii = tf.nn.conv1d(im, W[f'conv{i}_conv_ii_w'], stride=1,
                                  padding='SAME') + W[f'conv{i}_conv_ii_b']
            out_ri = tf.nn.conv1d(r, W[f'conv{i}_conv_ri_w'], stride=1,
                                  padding='SAME') + W[f'conv{i}_conv_ri_b']
            out_ir = tf.nn.conv1d(im, W[f'conv{i}_conv_ir_w'], stride=1,
                                  padding='SAME') + W[f'conv{i}_conv_ir_b']

            x = tf.complex(out_rr - out_ii, out_ri + out_ir)

            # ── ComplexBatchNorm (inference mode: use moving stats) ──
            r = tf.math.real(x)
            im = tf.math.imag(x)

            r = tf.nn.batch_normalization(
                r,
                mean=W[f'cbn{i}_bn_real_mean'],
                variance=W[f'cbn{i}_bn_real_var'],
                offset=W[f'cbn{i}_bn_real_beta'],
                scale=W[f'cbn{i}_bn_real_gamma'],
                variance_epsilon=eps_bn)
            im = tf.nn.batch_normalization(
                im,
                mean=W[f'cbn{i}_bn_imag_mean'],
                variance=W[f'cbn{i}_bn_imag_var'],
                offset=W[f'cbn{i}_bn_imag_beta'],
                scale=W[f'cbn{i}_bn_imag_gamma'],
                variance_epsilon=eps_bn)

            x = tf.complex(r, im)

            # ── CReLU ──
            x = tf.complex(tf.nn.relu(tf.math.real(x)),
                           tf.nn.relu(tf.math.imag(x)))

            # ── ComplexAvgPooling1D (skip after last block) ──
            if i < 3:
                r = tf.math.real(x)
                im = tf.math.imag(x)
                # AveragePooling1D with pool_size=2: need 4D for tf.nn.avg_pool
                r = tf.expand_dims(r, axis=1)   # (B, 1, T, C)
                im = tf.expand_dims(im, axis=1)
                r = tf.nn.avg_pool2d(r, ksize=[1, 1, 2, 1],
                                     strides=[1, 1, 2, 1], padding='VALID')
                im = tf.nn.avg_pool2d(im, ksize=[1, 1, 2, 1],
                                      strides=[1, 1, 2, 1], padding='VALID')
                r = tf.squeeze(r, axis=1)
                im = tf.squeeze(im, axis=1)
                x = tf.complex(r, im)

        # ── Complex Global Average Pooling ──
        x = tf.reduce_mean(x, axis=1)  # (B, C) complex64

        # ── Split complex → real float: concat(real, imag) → (B, 2*C) ──
        feat = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)

        # ── Dense(256, relu) ──
        feat = tf.matmul(feat, W['dense_w']) + W['dense_b']
        feat = tf.nn.relu(feat)

        # ── BatchNormalization (inference) ──
        feat = tf.nn.batch_normalization(
            feat,
            mean=W['bn_mean'], variance=W['bn_var'],
            offset=W['bn_beta'], scale=W['bn_gamma'],
            variance_epsilon=eps_bn)

        # ── Dropout (no-op at inference) ──
        # (training=False by default)

        # ── Dense(num_classes, softmax) ──
        logits = tf.matmul(feat, W['dense1_w']) + W['dense1_b']
        return logits


# For backward compatibility: experiment scripts that import build_sarp_model
# can use this wrapper. But SARPModel is preferred.
def build_sarp_model(num_classes=7):
    """Legacy wrapper — returns a SARPModel (not a Keras model)."""
    raise NotImplementedError(
        "build_sarp_model() is no longer needed. "
        "Use SARPModel(weights_path, num_classes) directly.")


class SARPAttackEngine:
    def __init__(self, model_path, device_label=0, epsilon=0.01,
                 attack='fgsm', pgd_steps=10, num_classes=7,
                 target_label=None):
        self.model_path = model_path
        self.device_label = device_label
        self.epsilon = epsilon
        self.attack = attack
        self.pgd_steps = pgd_steps
        self.target_label = target_label  # None = untargeted

        self.fft_size = 64
        self.cp_len = 16
        self.n_data_sc = 48

        data_sc = (
            list(range(-26, -21)) + list(range(-20, -7)) +
            list(range(-6, 0)) + list(range(1, 7)) +
            list(range(8, 21)) + list(range(22, 27))
        )
        self.data_bins = [sc + 32 for sc in data_sc]
        self.pilot_bins = [11, 25, 39, 53]

        self.sarp_fft = 64
        self.trace_len = 288
        self.stride = 288

        self._mapping_tf = None
        self._pilot_tf = None
        self._cached_n_ofdm = None
        self._cached_n_traces = None

        self.n_packets = 0
        self.total_acc_before = 0.0
        self.total_acc_after = 0.0

        if model_path is not None:
            self._load_model(num_classes)

    def _load_model(self, num_classes):
        import tensorflow as tf
        self.tf = tf
        self.keras = tf.keras

        self.model = SARPModel(self.model_path, num_classes)

        print(f"[SARP Engine] Input: {self.model.input_shape}, "
              f"Output: {self.model.output_shape}", file=sys.stderr)

    def _precompute(self, n_ofdm):
        if self._cached_n_ofdm == n_ofdm:
            return
        tf = self.tf

        mapping = np.zeros((48, 64), dtype=np.complex64)
        for i, b in enumerate(self.data_bins):
            mapping[i, b] = 1.0 + 0j
        self._mapping_tf = tf.constant(mapping, dtype=tf.complex64)

        pilot = np.zeros((1, 64), dtype=np.complex64)
        for b in self.pilot_bins:
            pilot[0, b] = 1.0 + 0j
        self._pilot_tf = tf.constant(pilot, dtype=tf.complex64)

        n_time = n_ofdm * (self.fft_size + self.cp_len)
        n_chunks = n_time // self.sarp_fft
        n_fft_samples = n_chunks * self.sarp_fft
        self._cached_n_traces = n_fft_samples // self.trace_len

        self._cached_n_ofdm = n_ofdm
        print(f"[SARP Engine] Precomputed: {n_ofdm} OFDM syms -> "
              f"{self._cached_n_traces} SARP traces", file=sys.stderr)

    def _tf_forward(self, sym_real, sym_imag, n_ofdm):
        tf = self.tf

        sym_complex = tf.complex(sym_real, sym_imag)
        data_syms = tf.reshape(sym_complex, [n_ofdm, self.n_data_sc])

        freq_vectors = tf.matmul(data_syms, self._mapping_tf)
        freq_vectors = freq_vectors + self._pilot_tf

        left = freq_vectors[:, :32]
        right = freq_vectors[:, 32:]
        unshifted = tf.concat([right, left], axis=1)

        time_syms = tf.signal.ifft(unshifted)

        cp = time_syms[:, -self.cp_len:]
        with_cp = tf.concat([cp, time_syms], axis=1)

        time_signal = tf.reshape(with_cp, [-1])

        n_time = n_ofdm * (self.fft_size + self.cp_len)
        n_chunks = n_time // self.sarp_fft
        trimmed = time_signal[:n_chunks * self.sarp_fft]
        chunks = tf.reshape(trimmed, [n_chunks, self.sarp_fft])

        fft_out = tf.signal.fft(chunks)
        left_f = fft_out[:, :32]
        right_f = fft_out[:, 32:]
        fft_shifted = tf.concat([right_f, left_f], axis=1)

        fft_stream = tf.reshape(fft_shifted, [-1])
        n_traces = self._cached_n_traces
        usable = n_traces * self.trace_len
        traces = tf.reshape(fft_stream[:usable], [n_traces, self.trace_len])

        #         rms = tf.sqrt(tf.reduce_mean(tf.abs(traces) ** 2,
        #                                       axis=1, keepdims=True))
        #         rms = tf.cast(tf.maximum(rms, 1e-10), tf.complex64)
        #         traces = traces / rms
        traces = tf.reshape(traces, [-1, self.trace_len, 1])
        logits = self.model(traces, training=False)
        return logits

    def _fgsm(self, symbols, n_ofdm):
        tf = self.tf
        n = n_ofdm * self.n_data_sc
        targeted = self.target_label is not None

        self._precompute(n_ofdm)
        # For targeted: use target_label; for untargeted: use device_label
        ref_label = self.target_label if targeted else self.device_label
        labels = tf.constant(
            np.full(self._cached_n_traces, ref_label, dtype=np.int32))

        sym_real = tf.Variable(np.real(symbols[:n]).astype(np.float32))
        sym_imag = tf.Variable(np.imag(symbols[:n]).astype(np.float32))

        with tf.GradientTape() as tape:
            logits = self._tf_forward(sym_real, sym_imag, n_ofdm)
            # CW loss: meaningful gradients even for very confident models
            batch_size = tf.shape(logits)[0]
            correct_logits = tf.gather_nd(logits, 
                tf.stack([tf.range(batch_size), labels[:batch_size]], axis=1))
            mask = tf.one_hot(labels[:batch_size], tf.shape(logits)[1])
            other_logits = tf.reduce_max(logits - mask * 1e9, axis=1)
            loss = tf.reduce_mean(other_logits - correct_logits)

        grad_r, grad_i = tape.gradient(loss, [sym_real, sym_imag])

        if grad_r is None or grad_i is None:
            print("[SARP Engine] WARNING: No gradient!", file=sys.stderr)
            return symbols, 0.0

        # Targeted: gradient DESCENT (toward target) → negative sign
        # Untargeted: gradient ASCENT (away from correct) → positive sign
        sign = -1.0 if targeted else 1.0

        perturbed = np.copy(symbols)
        perturbed[:n] = (
            (np.real(symbols[:n]) + sign * self.epsilon * np.sign(grad_r.numpy())) +
            1j * (np.imag(symbols[:n]) + sign * self.epsilon * np.sign(grad_i.numpy()))
        ).astype(np.complex64)

        return perturbed, float(loss.numpy())

    def _pgd(self, symbols, n_ofdm):
        tf = self.tf
        n = n_ofdm * self.n_data_sc
        alpha = self.epsilon / 4
        targeted = self.target_label is not None
        sign = -1.0 if targeted else 1.0

        self._precompute(n_ofdm)
        ref_label = self.target_label if targeted else self.device_label
        labels = tf.constant(
            np.full(self._cached_n_traces, ref_label, dtype=np.int32))

        orig_real = np.real(symbols[:n]).astype(np.float32)
        orig_imag = np.imag(symbols[:n]).astype(np.float32)

        curr_real = orig_real + np.random.uniform(
            -self.epsilon, self.epsilon, orig_real.shape).astype(np.float32)
        curr_imag = orig_imag + np.random.uniform(
            -self.epsilon, self.epsilon, orig_imag.shape).astype(np.float32)

        final_loss = 0.0
        for step in range(self.pgd_steps):
            sym_real = tf.Variable(curr_real)
            sym_imag = tf.Variable(curr_imag)

            with tf.GradientTape() as tape:
                logits = self._tf_forward(sym_real, sym_imag, n_ofdm)
                # CW loss
                batch_size = tf.shape(logits)[0]
                correct_logits = tf.gather_nd(logits,
                    tf.stack([tf.range(batch_size), labels[:batch_size]], axis=1))
                mask = tf.one_hot(labels[:batch_size], tf.shape(logits)[1])
                other_logits = tf.reduce_max(logits - mask * 1e9, axis=1)
                loss = tf.reduce_mean(other_logits - correct_logits)

            grad_r, grad_i = tape.gradient(loss, [sym_real, sym_imag])
            if grad_r is None:
                break

            curr_real = curr_real + sign * alpha * np.sign(grad_r.numpy())
            curr_imag = curr_imag + sign * alpha * np.sign(grad_i.numpy())

            curr_real = np.clip(curr_real, orig_real - self.epsilon,
                                orig_real + self.epsilon)
            curr_imag = np.clip(curr_imag, orig_imag - self.epsilon,
                                orig_imag + self.epsilon)
            final_loss = float(loss.numpy())

        perturbed = np.copy(symbols)
        perturbed[:n] = (curr_real + 1j * curr_imag).astype(np.complex64)
        return perturbed, final_loss

    def perturb(self, constellation_symbols):
        tf = self.tf
        n_total = len(constellation_symbols)
        n_ofdm = n_total // self.n_data_sc

        if n_ofdm == 0:
            return constellation_symbols

        self._precompute(n_ofdm)

        n = n_ofdm * self.n_data_sc
        sym_r = tf.constant(np.real(constellation_symbols[:n]).astype(np.float32))
        sym_i = tf.constant(np.imag(constellation_symbols[:n]).astype(np.float32))
        logits_before = self._tf_forward(sym_r, sym_i, n_ofdm)
        labels = np.full(len(logits_before), self.device_label, dtype=np.int32)
        preds_before = np.argmax(logits_before.numpy(), axis=1)
        acc_before = np.mean(preds_before == labels)

        if self.attack == 'fgsm':
            perturbed, loss = self._fgsm(constellation_symbols, n_ofdm)
        elif self.attack == 'pgd':
            perturbed, loss = self._pgd(constellation_symbols, n_ofdm)
        else:
            return constellation_symbols

        p_r = tf.constant(np.real(perturbed[:n]).astype(np.float32))
        p_i = tf.constant(np.imag(perturbed[:n]).astype(np.float32))
        logits_after = self._tf_forward(p_r, p_i, n_ofdm)
        preds_after = np.argmax(logits_after.numpy(), axis=1)
        acc_after = np.mean(preds_after == labels)

        self.n_packets += 1
        self.total_acc_before += acc_before
        self.total_acc_after += acc_after

        targ_str = ""
        if self.target_label is not None:
            targ_rate = np.mean(preds_after == self.target_label)
            self.total_targ_rate = getattr(self, 'total_targ_rate', 0.0) + targ_rate
            targ_str = f", targ_rate={targ_rate:.3f}"

        if self.n_packets <= 5 or self.n_packets % 100 == 0:
            print(f"[SARP] Pkt {self.n_packets}: "
                  f"acc {acc_before:.3f}->{acc_after:.3f}, "
                  f"loss={loss:.4f}, eps={self.epsilon}{targ_str}, "
                  f"{n_ofdm} OFDM, {self._cached_n_traces} traces",
                  file=sys.stderr)

        return perturbed

    def summary(self):
        if self.n_packets > 0:
            avg_b = self.total_acc_before / self.n_packets
            avg_a = self.total_acc_after / self.n_packets
            mode = "TARGETED" if self.target_label is not None else "UNTARGETED"
            print(f"\n[SARP] ========== Summary ==========", file=sys.stderr)
            print(f"[SARP] Packets: {self.n_packets}", file=sys.stderr)
            print(f"[SARP] Avg accuracy: {avg_b:.4f} -> {avg_a:.4f}", file=sys.stderr)
            print(f"[SARP] Attack: {self.attack.upper()} ({mode}), eps={self.epsilon}", file=sys.stderr)
            print(f"[SARP] Device label: {self.device_label}", file=sys.stderr)
            if self.target_label is not None:
                avg_t = self.total_targ_rate / self.n_packets
                print(f"[SARP] Target label: {self.target_label}, avg target rate: {avg_t:.4f}", file=sys.stderr)
            print(f"[SARP] ================================\n", file=sys.stderr)


def serve(model_path, device_label, epsilon, attack, pgd_steps, num_classes,
          target_label=None):
    engine = SARPAttackEngine(
        model_path=model_path, device_label=device_label,
        epsilon=epsilon, attack=attack,
        pgd_steps=pgd_steps, num_classes=num_classes,
        target_label=target_label
    )
    mode = f"TARGETED→{target_label}" if target_label is not None else "UNTARGETED"
    print(f"[SARP Server] Ready ({mode})", file=sys.stderr)

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        header = stdin.read(4)
        if len(header) < 4:
            break
        n = struct.unpack('<I', header)[0]
        if n == 0:
            break
        data = stdin.read(n * 8)
        if len(data) < n * 8:
            break

        symbols = np.frombuffer(data, dtype=np.complex64).copy()
        perturbed = engine.perturb(symbols)

        stdout.write(struct.pack('<I', len(perturbed)))
        stdout.write(perturbed.tobytes())
        stdout.flush()

    engine.summary()
    print("[SARP Server] Shutting down", file=sys.stderr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True,
                        help='Path to extracted model.weights.h5')
    parser.add_argument('--device-label', type=int, default=6)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--attack', default='fgsm', choices=['fgsm', 'pgd'])
    parser.add_argument('--pgd-steps', type=int, default=10)
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--target-label', type=int, default=None,
                        help='Target class for targeted attack (None = untargeted)')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.serve:
        serve(args.weights, args.device_label, args.epsilon,
              args.attack, args.pgd_steps, args.num_classes,
              args.target_label)
    elif args.test:
        engine = SARPAttackEngine(
            model_path=args.weights, device_label=args.device_label,
            epsilon=args.epsilon, attack=args.attack,
            num_classes=args.num_classes,
            target_label=args.target_label
        )
        n_ofdm = 177
        fake_syms = np.random.choice([-1.0, 1.0], size=n_ofdm * 48) + 0j
        fake_syms = fake_syms.astype(np.complex64)
        result = engine.perturb(fake_syms)
        engine.summary()
        print(f"Max perturbation: {np.max(np.abs(result - fake_syms)):.6f}")
    else:
        print("Use --serve or --test")

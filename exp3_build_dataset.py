#!/usr/bin/env python3
"""
exp3_build_dataset.py — Stage 2 of the fingerprinting pipeline
────────────────────────────────────────────────────────────────
Turn the per-device full-frame archives (Stage 1) into a windowed,
normalized, labeled train/val/test dataset for the 1D-CNN.

For each frame we slide a fixed window over the ACTIVE region (preamble +
payload), per-window power-normalize, and store 2 channels (I, Q).

Leakage-safe split:
  • run_3 frames        -> TEST            (unseen capture session)
  • run_1 + run_2 frames -> TRAIN / VAL    split BY FRAME (so no two windows
                                            from one frame land in both)

Each window also keeps origin metadata (device, run, frame index, window
offset) so the later OTA payload-perturbation study can map a window back to
the exact frame/sample it came from.

Output: <out>/dataset_win<L>.npz with
  X_train (N,2,L) f32, y_train (N,) i64, m_train (N,4) i32  [dev,run,frame,off]
  ... and _val / _test equivalents, plus label_names.

Run:
  python3 exp3_build_dataset.py                      # defaults (L=1024)
  python3 exp3_build_dataset.py --win 512 --val-frac 0.15
"""
import os, argparse, glob
import numpy as np

FRAMES_DIR_DEFAULT = '/media/nghoselab/T9/Data/session12/processed/frames'
OUT_DIR_DEFAULT    = '/media/nghoselab/T9/Data/session12/processed'
PRE_ROLL  = 256          # must match Stage 1
TAIL_PAD  = 200          # extra samples of active region to keep past measured dur
EPS       = 1e-12
SEED      = 1234


def windows_from_frame(frame, active_len, L, stride):
    """Yield per-window 2-channel arrays + window start offsets over [0, active_len)."""
    last = active_len - L
    if last < 0:
        return
    for off in range(0, last + 1, stride):
        seg = frame[off:off + L]
        power = np.mean(seg.real.astype(np.float32) ** 2 +
                        seg.imag.astype(np.float32) ** 2)
        scale = 1.0 / np.sqrt(power + EPS)          # per-window unit avg power
        ch = np.empty((2, L), dtype=np.float32)
        ch[0] = seg.real.astype(np.float32) * scale
        ch[1] = seg.imag.astype(np.float32) * scale
        yield ch, off


def build_split(frame_records, L, stride):
    """frame_records: list of (frame, active_len, dev, run, frame_idx)."""
    X, y, M = [], [], []
    for frame, active_len, dev, run, fidx in frame_records:
        for ch, off in windows_from_frame(frame, active_len, L, stride):
            X.append(ch)
            y.append(dev - 1)                        # labels 0..5
            M.append((dev, run, fidx, off))
    if not X:
        return (np.empty((0, 2, L), np.float32),
                np.empty((0,), np.int64),
                np.empty((0, 4), np.int32))
    return (np.stack(X), np.asarray(y, np.int64), np.asarray(M, np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames-dir', default=FRAMES_DIR_DEFAULT)
    ap.add_argument('--out',        default=OUT_DIR_DEFAULT)
    ap.add_argument('--win',     type=int,   default=1024, help='window length (IQ samples)')
    ap.add_argument('--train-stride', type=int, default=512,  help='stride for train (overlap aug)')
    ap.add_argument('--eval-stride',  type=int, default=1024, help='stride for val/test (no overlap)')
    ap.add_argument('--test-run', type=int,   default=3,    help='which run is held out as test')
    ap.add_argument('--val-frac', type=float, default=0.15, help='fraction of train frames -> val')
    args = ap.parse_args()

    L = args.win
    rng = np.random.default_rng(SEED)

    # ── Gather frame records, splitting by run/frame BEFORE windowing ──
    train_recs, val_recs, test_recs = [], [], []
    files = sorted(glob.glob(os.path.join(args.frames_dir, 'frames_dev*.npz')))
    if not files:
        raise SystemExit(f"no frame archives in {args.frames_dir}")

    print(f"window={L}  train_stride={args.train_stride}  eval_stride={args.eval_stride}")
    print(f"test_run={args.test_run}  val_frac={args.val_frac}\n")

    per_dev_counts = {}
    for fp in files:
        z = np.load(fp)
        frames, run, dur = z['frames'], z['run'], z['dur']
        dev = int(z['device'])
        # active extent per frame = pre-roll + measured burst duration (+pad)
        active = np.minimum(PRE_ROLL + dur.astype(np.int64) + TAIL_PAD, frames.shape[1])

        is_test = run == args.test_run
        trainval_idx = np.where(~is_test)[0]
        test_idx     = np.where(is_test)[0]

        # split train/val BY FRAME among run_1+run_2
        perm = rng.permutation(trainval_idx)
        n_val = int(round(len(perm) * args.val_frac))
        val_idx   = set(perm[:n_val].tolist())
        train_idx = set(perm[n_val:].tolist())

        for i in range(len(frames)):
            rec = (frames[i], int(active[i]), dev, int(run[i]), i)
            if is_test[i]:
                test_recs.append(rec)
            elif i in val_idx:
                val_recs.append(rec)
            else:
                train_recs.append(rec)
        per_dev_counts[dev] = dict(
            train=len(train_idx), val=len(val_idx), test=len(test_idx))
        print(f"  dev{dev}: frames train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # ── Window each split ──
    print("\nwindowing...")
    Xtr, ytr, mtr = build_split(train_recs, L, args.train_stride)
    Xva, yva, mva = build_split(val_recs,   L, args.eval_stride)
    Xte, yte, mte = build_split(test_recs,  L, args.eval_stride)

    def dist(y):
        return np.bincount(y, minlength=6).tolist()
    print(f"  train windows: {len(ytr):6d}  per-class {dist(ytr)}")
    print(f"  val   windows: {len(yva):6d}  per-class {dist(yva)}")
    print(f"  test  windows: {len(yte):6d}  per-class {dist(yte)}")

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"dataset_win{L}.npz")
    np.savez(out,
             X_train=Xtr, y_train=ytr, m_train=mtr,
             X_val=Xva,   y_val=yva,   m_val=mva,
             X_test=Xte,  y_test=yte,  m_test=mte,
             label_names=np.array([f"device_{i}" for i in range(1, 7)]),
             win=L, test_run=args.test_run,
             train_stride=args.train_stride, eval_stride=args.eval_stride)
    mb = os.path.getsize(out) / 1e6
    print(f"\nsaved -> {out}  ({mb:.0f} MB)")


if __name__ == '__main__':
    main()

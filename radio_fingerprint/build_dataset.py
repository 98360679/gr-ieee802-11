#!/usr/bin/env python3
"""
Stage 1 — Build a windowed, power-normalized IQ dataset from the Day1 captures.

For each (device, run):
  • memmap the complex64 capture,
  • build a coarse power envelope and threshold it to find signal bursts,
  • keep only substantial bursts (real data frames, not ACKs/noise),
  • slide L-sample windows (stride S) inside each burst,
  • per-window unit-average-power normalize,
  • tag each window with (device, run, global burst id).

Split: run_1 + run_2 -> train/val (split BY BURST so windows from one burst
never straddle train & val); run_3 -> test (unseen session).

Output: artifacts/windows_L512.npz
  X_train (N,L) c64, y_train (N,), g_train (N,)  + _val / _test
  label_names, fs, win_len
"""
import os, time
import numpy as np
import config as C

rng = np.random.default_rng(C.SEED)


def detect_bursts(path):
    """Return [(onset, end)] sample spans of substantial signal bursts."""
    nsamp = os.path.getsize(path) // 8
    mm = np.memmap(path, dtype=np.complex64, mode="r", shape=(nsamp,))

    # coarse power envelope (chunked to bound RAM)
    W = C.ENV_WIN
    env_parts = []
    CH = 16_000_000
    for off in range(0, nsamp, CH):
        seg = mm[off:off + CH]
        m = (len(seg) // W) * W
        if m == 0:
            continue
        p = (seg[:m].real.astype(np.float32) ** 2 +
             seg[:m].imag.astype(np.float32) ** 2)
        env_parts.append(p.reshape(-1, W).mean(1))
    env = np.concatenate(env_parts)

    floor = np.median(env)
    thr = max(floor * C.THR_FLOOR_K, env.mean() * 2.0)
    active = env > thr

    # group contiguous active envelope bins into bursts
    spans = []
    i, na = 0, len(active)
    while i < na:
        if active[i]:
            j = i
            while j < na and active[j]:
                j += 1
            onset = i * W
            end = j * W
            if end - onset >= C.MIN_BURST:
                spans.append((onset, end))
            i = j
        else:
            i += 1
    return mm, nsamp, spans


def windows_from_span(mm, onset, end, L, S):
    """Yield power-normalized complex windows over [onset, end)."""
    start = max(onset - C.PRE_ROLL, 0)
    last = end - L
    for off in range(start, last + 1, S):
        seg = np.asarray(mm[off:off + L], dtype=np.complex64)
        power = np.mean(seg.real ** 2 + seg.imag ** 2)
        scale = np.float32(1.0 / np.sqrt(power + C.EPS))
        yield seg * scale


def collect_device(dev, dev_label, burst_ctr):
    """Return train-pool and test window lists for one device."""
    pool_X, pool_g, pool_run = [], [], []   # run1+run2
    test_X, test_g = [], []                 # run3
    for run in C.RUNS:
        path = os.path.join(C.DATA_ROOT, dev, f"run_{run}.bin")
        t0 = time.time()
        mm, nsamp, spans = detect_bursts(path)
        n_win = 0
        for (onset, end) in spans:
            gid = burst_ctr[0]; burst_ctr[0] += 1
            for w in windows_from_span(mm, onset, end, C.WIN_LEN, C.WIN_STRIDE):
                if run == C.TEST_RUN:
                    test_X.append(w); test_g.append(gid)
                else:
                    pool_X.append(w); pool_g.append(gid); pool_run.append(run)
                n_win += 1
        del mm
        print(f"  {dev}/run_{run}: {len(spans):4d} bursts -> {n_win:6d} windows "
              f"({time.time()-t0:.1f}s)")
    return (pool_X, pool_g, pool_run), (test_X, test_g)


def cap(idx, n_max):
    if len(idx) <= n_max:
        return idx
    return rng.choice(idx, size=n_max, replace=False)


def main():
    t0 = time.time()
    burst_ctr = [0]
    Xtr, ytr, gtr = [], [], []
    Xva, yva, gva = [], [], []
    Xte, yte, gte = [], [], []

    for lbl, dev in enumerate(C.DEVICES):
        print(f"[{lbl}] {dev}")
        (pX, pg, prun), (tX, tg) = collect_device(dev, lbl, burst_ctr)
        pX = np.asarray(pX, dtype=np.complex64)
        pg = np.asarray(pg, dtype=np.int64)
        tX = np.asarray(tX, dtype=np.complex64)
        tg = np.asarray(tg, dtype=np.int64)

        # val split BY BURST GROUP from the run1+run2 pool
        groups = np.unique(pg)
        rng.shuffle(groups)
        n_val_g = max(1, int(len(groups) * C.VAL_FRAC))
        val_groups = set(groups[:n_val_g].tolist())
        is_val = np.array([g in val_groups for g in pg])

        tr_idx = cap(np.where(~is_val)[0], C.MAX_TRAIN_PER_DEV)
        va_idx = cap(np.where(is_val)[0],  C.MAX_VAL_PER_DEV)
        te_idx = cap(np.arange(len(tX)),   C.MAX_TEST_PER_DEV)

        Xtr.append(pX[tr_idx]); ytr.append(np.full(len(tr_idx), lbl)); gtr.append(pg[tr_idx])
        Xva.append(pX[va_idx]); yva.append(np.full(len(va_idx), lbl)); gva.append(pg[va_idx])
        Xte.append(tX[te_idx]); yte.append(np.full(len(te_idx), lbl)); gte.append(tg[te_idx])
        print(f"    -> train {len(tr_idx)}  val {len(va_idx)}  test {len(te_idx)}")

    def cat(parts, dt):
        return np.concatenate(parts).astype(dt)

    X_train = cat(Xtr, np.complex64); y_train = cat(ytr, np.int64); g_train = cat(gtr, np.int64)
    X_val   = cat(Xva, np.complex64); y_val   = cat(yva, np.int64); g_val   = cat(gva, np.int64)
    X_test  = cat(Xte, np.complex64); y_test  = cat(yte, np.int64); g_test  = cat(gte, np.int64)

    print(f"\nTOTAL  train {len(X_train)}  val {len(X_val)}  test {len(X_test)}")
    np.savez(C.DATASET,
             X_train=X_train, y_train=y_train, g_train=g_train,
             X_val=X_val,     y_val=y_val,     g_val=g_val,
             X_test=X_test,   y_test=y_test,   g_test=g_test,
             label_names=np.array(C.DEVICES), fs=C.FS, win_len=C.WIN_LEN)
    mb = os.path.getsize(C.DATASET) / 1e6
    print(f"saved {C.DATASET}  ({mb:.0f} MB)  in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

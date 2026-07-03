#!/usr/bin/env python3
"""
exp3_cvnn_singlerun.py — fine-tune the 6/30 CVNN on ONE run, 80/10/10 frame split.

Removes cross-run channel drift (train+val+test all from the same run), so it measures
whether the devices separate when the channel is held fixed — the honest within-session
number, and closer to what an OTA attack session needs. Loops all 3 runs and reports the
TEST (held-out 10%) accuracy per run so we can pick the best.
"""
import os, re, glob, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from exp3_cvnn import CVNN, iq_to_complex, drift_augment, per_device_eval
from exp3_extract_frames import extract_frames_for_file
from exp3_fp_model import frame_to_windows, NUM_CLASSES, DEVICE_NAMES, WIN

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE = None   # set in main()


def run_path(d, run):
    hits = glob.glob(f'{BASE}/device_{d}/*run_{run}.bin') or glob.glob(f'{BASE}/device_{d}/*run{run}.bin')
    return hits[0] if hits else None


def detect_runs():
    runs = set()
    for f in glob.glob(f'{BASE}/device_1/*.bin'):
        m = re.search(r'run_?(\d+)', os.path.basename(f))
        if m:
            runs.add(int(m.group(1)))
    return sorted(runs)


def build_run(run, val_frac=0.1, test_frac=0.1, hop_train=384):
    T = {k: [] for k in ('Xtr', 'ytr', 'Xva', 'yva', 'vfid', 'Xte', 'yte', 'tfid')}
    fid = tid = 0
    for d in range(1, NUM_CLASSES + 1):
        fr = extract_frames_for_file(run_path(d, run), floor_pct=20.0, thr_mult=2.0)[0]
        n = len(fr)
        ntr, ntv = int((1 - val_frac - test_frac) * n), int((1 - test_frac) * n)
        for i, f in enumerate(fr):
            sp = 'tr' if i < ntr else ('va' if i < ntv else 'te')
            x = iq_to_complex(frame_to_windows(f, hop=hop_train if sp == 'tr' else WIN))
            if sp == 'tr':
                T['Xtr'].append(x); T['ytr'].append(np.full(len(x), d - 1))
            elif sp == 'va':
                T['Xva'].append(x); T['yva'].append(np.full(len(x), d - 1)); T['vfid'].append(np.full(len(x), fid)); fid += 1
            else:
                T['Xte'].append(x); T['yte'].append(np.full(len(x), d - 1)); T['tfid'].append(np.full(len(x), tid)); tid += 1
    return (np.concatenate(T['Xtr']), np.concatenate(T['ytr']).astype(np.int64),
            np.concatenate(T['Xva']), np.concatenate(T['yva']).astype(np.int64), np.concatenate(T['vfid']).astype(np.int64),
            np.concatenate(T['Xte']), np.concatenate(T['yte']).astype(np.int64), np.concatenate(T['tfid']).astype(np.int64))


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='enrollment dir with device_N/*.bin')
    ap.add_argument('--init', default='fingerprint_cvnn_6_30_validate.pt',
                    help='checkpoint to fine-tune from; use "scratch" to train from random init')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--val-frac', type=float, default=0.10)
    ap.add_argument('--test-frac', type=float, default=0.10)
    ap.add_argument('--save', default=None, help='save the (best-run) fine-tuned checkpoint here')
    a = ap.parse_args()
    BASE = a.root
    torch.manual_seed(0); np.random.seed(0)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    runs = detect_runs()
    split = f"{int(100*(1-a.val_frac-a.test_frac))}/{int(100*a.val_frac)}/{int(100*a.test_frac)}"
    print(f"root {BASE}   runs {runs}   split {split}   "
          f"mode {'from-scratch' if a.init=='scratch' else 'fine-tune '+a.init}")
    results = {}
    for run in runs:
        Xtr, ytr, Xva, yva, vfid, Xte, yte, tfid = build_run(run, a.val_frac, a.test_frac)
        print(f"\n=== RUN {run}  train {Xtr.shape} val {Xva.shape} test {Xte.shape} ===")
        m = CVNN(NUM_CLASSES).to(DEV)
        if a.init != 'scratch':
            m.load_state_dict(torch.load(a.init, map_location=DEV))
        dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=256, shuffle=True)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-4)
        best, best_state = 0.0, None
        for ep in range(1, a.epochs + 1):
            m.train()
            for xb, yb in dl:
                xb = drift_augment(xb.to(DEV)); re, im = xb.real.unsqueeze(1), xb.imag.unsqueeze(1); yb = yb.to(DEV)
                opt.zero_grad(); loss = lossf(m(re, im), yb); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            w, f, _, _ = per_device_eval(m, DEV, Xva, yva, vfid, NUM_CLASSES)   # val for selection
            if w > best:
                best = w; best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        m.load_state_dict(best_state)
        w, f, per, C = per_device_eval(m, DEV, Xte, yte, tfid, NUM_CLASSES)      # TEST report
        pd = {DEVICE_NAMES[i]: per[i][0] / max(1, per[i][1]) for i in range(NUM_CLASSES)}
        results[run] = (f, pd, best_state)
        print(f"  RUN {run} TEST frame acc {f:.3f}   " + "  ".join(f"{k}={v:.2f}" for k, v in pd.items()))
    best_run = max(results, key=lambda r: results[r][0])
    print(f"\n>>> BEST RUN = run_{best_run}  (test frame acc {results[best_run][0]:.3f})")
    for r in sorted(results):
        print(f"    run_{r}: {results[r][0]:.3f}")
    if a.save:
        torch.save(results[best_run][2], a.save)
        print(f"  saved best-run checkpoint -> {a.save}")


if __name__ == '__main__':
    main()

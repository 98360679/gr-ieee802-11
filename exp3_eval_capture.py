#!/usr/bin/env python3
"""
exp3_eval_capture.py — evaluate the fingerprinter on (attacked) re-captures
─────────────────────────────────────────────────────────────────────────────
After running the legit TX + adversary (exp3_adversary_tx.py) simultaneously and
re-capturing, point this at the new captures to measure the REAL over-the-air
fooling rate. It reuses the EXACT Stage-1 extraction and Stage-2 windowing so
numbers are directly comparable to the clean baseline (window 0.956 / frame 0.990).

Each capture's true device label is known (you captured a known device), so we
can report per-device accuracy, a confusion matrix, and the accuracy DROP vs clean.

Input (either form):
  --root DIR        scan DIR/Device_<d>/*.bin   (label = d, mirrors training layout)
  --capture FILE --device D     a single capture with explicit label
                                (repeat the pair to pass several)

Run:
  python3 exp3_eval_capture.py --root /media/.../session13_attacked/eval
  python3 exp3_eval_capture.py --capture /tmp/dev3_attacked.bin --device 3
"""
import os, sys, glob, re, json, argparse
import numpy as np
import torch

import exp3_attack_lib as A
from exp3_extract_frames import extract_frames_for_file, FRAME_LEN
from exp3_build_dataset import windows_from_frame, PRE_ROLL, TAIL_PAD

OUT_DEFAULT = '/media/nghoselab/T9/Data/session12/processed'
CLEAN_WIN_ACC, CLEAN_FRAME_ACC = 0.9564, 0.9897   # clean run_3 baseline (this model)


def device_from_dirname(name):
    m = re.search(r'device[_\-]?(\d+)', name, re.I)
    return int(m.group(1)) if m else None


def collect_captures(args):
    """Return list of (path, device_label_int)."""
    items = []
    if args.root:
        for dd in sorted(glob.glob(os.path.join(args.root, '*'))):
            if not os.path.isdir(dd):
                continue
            dev = device_from_dirname(os.path.basename(dd))
            if dev is None:
                continue
            for f in sorted(glob.glob(os.path.join(dd, '*.bin'))):
                items.append((f, dev))
    for f, d in zip(args.capture or [], args.device or []):
        items.append((f, int(d)))
    return items


def windows_from_capture(path, win, stride):
    """Extract full frames then window them exactly as Stage 1+2 do.

    Returns X (M,2,win) float32, frame_id (M,) int — frame_id groups windows from
    the same frame for majority voting.
    """
    frames, starts, peaks, durs, floor, thresh = extract_frames_for_file(path)
    X, fid = [], []
    for k, (fr, dur) in enumerate(zip(frames, durs)):
        active = min(PRE_ROLL + int(dur) + TAIL_PAD, FRAME_LEN)
        for ch, off in windows_from_frame(np.asarray(fr, np.complex64), active, win, stride):
            X.append(ch); fid.append(k)
    if not X:
        return np.empty((0, 2, win), np.float32), np.empty((0,), np.int64), len(frames)
    return np.stack(X), np.asarray(fid, np.int64), len(frames)


@torch.no_grad()
def predict(model, X, device, bs=512):
    model.eval()
    out = []
    Xt = torch.from_numpy(X)
    for i in range(0, len(Xt), bs):
        out.append(model(Xt[i:i+bs].to(device)).argmax(1).cpu())
    return torch.cat(out).numpy() if out else np.empty((0,), np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None)
    ap.add_argument('--capture', action='append', help='capture .bin (repeatable, pair with --device)')
    ap.add_argument('--device',  action='append', help='device label for the matching --capture')
    ap.add_argument('--win', type=int, default=None, help='window len (default: model win)')
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--tag', default='attacked', help='label for output files')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    dev = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    model, names, model_win = A.load_model(device=dev)
    nC = len(names)
    win = args.win or model_win
    stride = win                                   # no overlap (matches test protocol)

    items = collect_captures(args)
    if not items:
        sys.exit("no captures found — pass --root DIR or --capture FILE --device D")
    print(f"evaluating {len(items)} capture(s); window={win}, no-overlap\n")

    cm_w = np.zeros((nC, nC), int)                 # window-level confusion
    cm_f = np.zeros((nC, nC), int)                 # frame-vote confusion
    per_dev = {}
    gframe_off = 0
    for path, devlbl in items:
        X, fid, nframes = windows_from_capture(path, win, stride)
        if len(X) == 0:
            print(f"  [{os.path.basename(path)}] dev{devlbl}: NO frames detected (adversary saturating?)")
            continue
        pred = predict(model, X, dev)
        true = devlbl - 1
        # window-level
        for p in pred:
            cm_w[true, p] += 1
        w_acc = (pred == true).mean()
        # frame-vote
        fr_pred, fr_corr = [], 0
        for k in np.unique(fid):
            vote = np.bincount(pred[fid == k], minlength=nC).argmax()
            fr_pred.append(vote)
            cm_f[true, vote] += 1
            fr_corr += int(vote == true)
        f_acc = fr_corr / len(fr_pred)
        d = per_dev.setdefault(devlbl, dict(win_correct=0, win_total=0, fr_correct=0, fr_total=0, frames=0))
        d['win_correct'] += int((pred == true).sum()); d['win_total'] += len(pred)
        d['fr_correct'] += fr_corr; d['fr_total'] += len(fr_pred); d['frames'] += nframes
        print(f"  [{os.path.basename(path)}] dev{devlbl}: {nframes} frames, {len(pred)} win  "
              f"win_acc={w_acc:.3f}  frame_acc={f_acc:.3f}")

    # ── summary ──
    print("\n" + "=" * 62)
    print("PER-DEVICE (attacked re-capture)")
    print("  dev   frames   win_acc   frame_acc")
    tot_wc = tot_wt = tot_fc = tot_ft = 0
    for devlbl in sorted(per_dev):
        d = per_dev[devlbl]
        wa = d['win_correct']/max(d['win_total'],1); fa = d['fr_correct']/max(d['fr_total'],1)
        print(f"  d{devlbl}   {d['frames']:5d}    {wa:.3f}     {fa:.3f}")
        tot_wc += d['win_correct']; tot_wt += d['win_total']
        tot_fc += d['fr_correct']; tot_ft += d['fr_total']
    win_acc = tot_wc/max(tot_wt,1); frame_acc = tot_fc/max(tot_ft,1)
    print("=" * 62)
    print(f"OVERALL window acc : {win_acc:.4f}   (clean {CLEAN_WIN_ACC:.4f}, "
          f"drop {CLEAN_WIN_ACC-win_acc:+.4f})")
    print(f"OVERALL frame  acc : {frame_acc:.4f}   (clean {CLEAN_FRAME_ACC:.4f}, "
          f"drop {CLEAN_FRAME_ACC-frame_acc:+.4f})")
    print(f"OTA fooling rate   : {1-win_acc:.4f} window-level, {1-frame_acc:.4f} frame-level")
    print("\nwindow-level confusion (rows=true, cols=pred):")
    print("       " + " ".join(f"d{j+1:>4}" for j in range(nC)))
    for i in range(nC):
        if cm_w[i].sum() == 0: continue
        print(f"  d{i+1}  " + " ".join(f"{cm_w[i,j]:5d}" for j in range(nC)))

    res = dict(tag=args.tag, n_captures=len(items), window=win,
               window_acc=win_acc, frame_acc=frame_acc,
               clean_window_acc=CLEAN_WIN_ACC, clean_frame_acc=CLEAN_FRAME_ACC,
               window_drop=CLEAN_WIN_ACC-win_acc, frame_drop=CLEAN_FRAME_ACC-frame_acc,
               confusion_window=cm_w.tolist(), confusion_frame=cm_f.tolist(),
               per_device={str(k): v for k, v in per_dev.items()})
    op = os.path.join(args.out, f'eval_capture_{args.tag}.json')
    with open(op, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved -> {op}")


if __name__ == '__main__':
    main()

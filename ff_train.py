#!/usr/bin/env python3
"""
ff_train.py — "Fading Fingerprints" training / evaluation wrapper.

Trains the CVNN fingerprinter (same architecture + unit-RMS complex preprocessing as
the attack work) on one experiment's devices, under either split protocol:

  within-day : per-day separability   (train runs 1,2 -> test held-out run of --day)
  cross-day  : temporal drift         (train ALL runs of --train-day -> test --val-day)

The class count is taken from the devices actually present, so Exp 1 (same model) and
Exp 2 (different model) each train on their own device set. Reports per-device accuracy
and a confusion matrix; for cross-day it prints the Day->Day frame accuracy, the headline
Fading-Fingerprints stability number.

CFO handling (the analysis knob that explains *why* same-model may fade):
  --cfo-correct : sync out each window's carrier offset before the model  -> CFO-invariant.
  --cfo-max N   : magnitude of the CFO drift augmentation (default 3000 Hz).
  --no-aug      : disable drift augmentation entirely — use this for the RAW cross-day
                  drop (no synthetic drift masking the real day-to-day drift).

Suggested study sweep for one experiment/day pair:
  1) baseline   :  --protocol cross-day --no-aug              (raw drift, honest drop)
  2) drift-aug  :  --protocol cross-day                        (does augmentation recover it?)
  3) cfo-robust :  --protocol cross-day --cfo-correct          (is the drift just CFO?)

  python3 ff_train.py --exp 1 --protocol cross-day --train-day 1 --val-day 2 --epochs 40
"""
import os, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from exp3_fp_model import WIN
from exp3_cvnn import CVNN, drift_augment, to_reim, per_device_eval, n_params
from ff_dataset import build_dataset, DEFAULT_ROOT
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--exp', type=int, required=True, choices=[1, 2])
    ap.add_argument('--protocol', required=True, choices=['within-day', 'cross-day'])
    ap.add_argument('--day', type=int, default=1, help='within-day: which day')
    ap.add_argument('--val-run', type=int, default=3, help='within-day: run held out for val')
    ap.add_argument('--train-day', type=int, default=1, help='cross-day: train day')
    ap.add_argument('--val-day', type=int, default=2, help='cross-day: test day')
    ap.add_argument('--devices', default=None,
                    help='explicit comma list of folder names (e.g. B200,USRP2,N2922), else auto-discover')
    # training
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--hop-train', type=int, default=384)
    ap.add_argument('--thr-mult', type=float, default=2.0)
    ap.add_argument('--init', default=None, help='CVNN checkpoint to FINE-TUNE from')
    ap.add_argument('--no-aug', action='store_true', help='disable drift augmentation')
    ap.add_argument('--cfo-correct', action='store_true', help='CFO-sync each window (frequency-invariant)')
    ap.add_argument('--cfo-max', type=float, default=3000.0)
    ap.add_argument('--snr-lo', type=float, default=8.0)
    ap.add_argument('--snr-hi', type=float, default=30.0)
    ap.add_argument('--tag', default=None, help='output name suffix (auto if unset)')
    ap.add_argument('--rebuild', action='store_true')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); np.random.seed(0)

    if a.protocol == 'within-day':
        span = f'day{a.day}-run{a.val_run}'
    else:
        span = f'day{a.train_day}->day{a.val_day}'
    print(f'FF train  exp{a.exp}  {a.protocol} ({span})  cfo_correct={a.cfo_correct}  aug={not a.no_aug}  device {dev}')

    Xtr, ytr, Xva, yva, vfid, id2label, names, counts = build_dataset(
        a.root, a.exp, a.protocol, train_day=a.train_day, val_day=a.val_day,
        day=a.day, val_run=a.val_run,
        devices=a.devices.split(',') if a.devices else None,
        hop_train=a.hop_train, thr_mult=a.thr_mult, cfo_correct=a.cfo_correct, rebuild=a.rebuild)
    nc = len(names)
    print(f'  devices ({nc}): {names}')
    print(f'  frames/device: {counts}')
    print(f'  train {Xtr.shape}  val {Xva.shape}  ({len(np.unique(vfid))} val frames)')

    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=256, shuffle=True)
    m = CVNN(nc).to(dev)
    if a.init:
        m.load_state_dict(torch.load(a.init, map_location=dev))
        print(f'  FINE-TUNING from {a.init}  (lr {a.lr})')
    print(f'  CVNN params: {n_params(m):,}')
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_state = 0.0, None
    for ep in range(1, a.epochs + 1):
        m.train(); tot = nb = 0
        for xb, yb in dl:
            xb = xb.to(dev)
            if not a.no_aug:
                xb = drift_augment(xb, a.cfo_max, a.snr_lo, a.snr_hi)
            re, im = xb.real.unsqueeze(1), xb.imag.unsqueeze(1); yb = yb.to(dev)
            opt.zero_grad(); loss = lossf(m(re, im), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        w, f, _, _ = per_device_eval(m, dev, Xva, yva, vfid, nc)
        if ep % 5 == 0 or ep == 1:
            print(f'  epoch {ep:2d}  loss {tot/nb:.4f}  val_win {w:.4f}  val_frame {f:.4f}')
        if w > best:
            best = w; best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    w, f, per, C = per_device_eval(m, dev, Xva, yva, vfid, nc)

    label = 'CROSS-DAY drift' if a.protocol == 'cross-day' else 'within-day held-out'
    print(f'\n  === {label}: {span} ===')
    print(f'  val_win {w:.4f}   val_frame {f:.4f}   <- headline accuracy')
    for i in range(nc):
        ok, t = per[i]; print(f'    {names[i]}: {ok/max(1,t):.3f} ({ok}/{t})')
    print('\n  confusion (rows=true):')
    print('  true\\pred ' + ''.join(f'{n:>11}' for n in names))
    for i in range(nc):
        print(f'  {names[i]:>9}' + ''.join(f'{C[i,j]:>11}' for j in range(nc)) + f'  n={C[i].sum()}')

    tag = a.tag or f'ff_exp{a.exp}_{a.protocol.replace("-","")}_{span.replace("->","to").replace("day","d")}' \
                   f'{"_cfocorr" if a.cfo_correct else ""}{"_noaug" if a.no_aug else ""}'
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'fingerprint_{tag}.pt')
    torch.save(best_state, out)
    meta = {'exp': a.exp, 'protocol': a.protocol, 'span': span, 'devices': names,
            'id2label': {str(k): v for k, v in id2label.items()},
            'cfo_correct': a.cfo_correct, 'no_aug': a.no_aug, 'cfo_max': a.cfo_max,
            'val_win': round(w, 4), 'val_frame': round(f, 4)}
    with open(out.replace('.pt', '.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f'\n  saved -> {out}\n         + {out.replace(".pt", ".json")}')


if __name__ == '__main__':
    main()

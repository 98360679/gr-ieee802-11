#!/usr/bin/env python3
"""
ff_dataset.py — "Fading Fingerprints" dataset builder.

Turns the ff_capture.py tree

    <root>/day{D}/exp{E}/device_{id}/clean_run_{k}.bin

into (train, val) window tensors for the CVNN, with two split protocols:

  * within-day  — per-day separability. Train on runs {1,2}, hold out run {val_run}
                  as validation, WITHIN a single day. Answers "can we tell these
                  devices apart today?" (leakage-safe: whole runs held out, never
                  frames from a training run).

  * cross-day   — temporal stability / drift. Train on ALL runs of --train-day,
                  test on ALL runs of --val-day. Answers "does a model trained
                  yesterday still recognise these devices today?" — the headline
                  Fading-Fingerprints number.

Unlike the fixed 6-class attack pipeline, the class set is discovered from the
device_* folders present, so each experiment (Exp 1 same-model, Exp 2 different-
model) trains on exactly its own devices. Device IDs need not be contiguous; they
are mapped to labels 0..K-1 by sorted ID. For cross-day, only devices present on
BOTH days are used (you can't measure drift for a device captured once).

Frame extraction and unit-RMS complex windowing are identical to exp3_cvnn.py, so a
model trained here is directly comparable to the attack-era fingerprinters.

  # inventory only (no training):
  python3 ff_dataset.py --exp 1 --protocol within-day --day 1
  python3 ff_dataset.py --exp 1 --protocol cross-day --train-day 1 --val-day 2
"""
import os, re, glob, json, argparse, numpy as np
from exp3_fp_model import frame_to_windows, WIN
from exp3_extract_frames import extract_frames_for_file
from exp3_cvnn import iq_to_complex          # single source of truth for CFO-correct + unit-RMS

DEFAULT_ROOT = '/home/nghoselab/captures/fading_fingerprints'
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.ff_cache')
os.makedirs(CACHE_DIR, exist_ok=True)


def exp_dir(root, day, exp):
    return os.path.join(root, f'day{day}', f'exp{exp}')


def discover_devices(root, day, exp):
    """sorted list of device folder NAMES under day{day}/exp{exp}. These are the class
    labels: 'device_1'/'device_2'/... for Exp 1 (same model → numbered), or model names
    like 'B200'/'USRP2'/'N2922' for Exp 2 (different models → named)."""
    names = []
    for p in glob.glob(os.path.join(exp_dir(root, day, exp), '*')):
        if os.path.isdir(p):
            names.append(os.path.basename(p))
    return sorted(set(names))


def _runs_for(root, day, exp, dev):
    # ONLY the RX captures — never the TX-side frame_run_*.bin tap that shares the folder.
    return sorted(glob.glob(os.path.join(exp_dir(root, day, exp), dev, 'clean_run_*.bin')))


def _run_no(cap):
    m = re.search(r'run_?(\d+)', os.path.basename(cap))
    return int(m.group(1)) if m else 1


def _windows_from_cap(cap, hop, thr_mult, cfo_correct):
    """extract frames from one clean_run .bin -> list of per-frame window arrays."""
    frames = extract_frames_for_file(cap, floor_pct=20.0, thr_mult=thr_mult)[0]
    return [iq_to_complex(frame_to_windows(f, hop=hop), cfo_correct) for f in frames]


def build_dataset(root, exp, protocol, train_day=1, val_day=2, day=1, val_run=3,
                  devices=None, hop_train=384, thr_mult=2.0, cfo_correct=False,
                  cache=True, rebuild=False, verbose=True):
    """Returns (Xtr, ytr, Xva, yva, vfid, id2label, names, counts).
    vfid is a per-validation-window frame id for frame-level majority voting."""
    if protocol == 'within-day':
        tag_days = f'd{day}vr{val_run}'
        train_src = val_src = [day]
    elif protocol == 'cross-day':
        tag_days = f't{train_day}v{val_day}'
        train_src, val_src = [train_day], [val_day]
    else:
        raise ValueError(f'unknown protocol {protocol!r}')

    # device set: explicit, else discovered (intersection across all involved days)
    if devices is None:
        day_sets = [set(discover_devices(root, d, exp)) for d in sorted(set(train_src + val_src))]
        devices = sorted(set.intersection(*day_sets)) if day_sets and all(day_sets) else \
                  sorted(set.union(*day_sets)) if day_sets else []
    if not devices:
        raise SystemExit(f'no devices found under {exp_dir(root, train_day if protocol=="cross-day" else day, exp)} '
                         f'(and matching day) — capture some first.')
    id2label = {d: i for i, d in enumerate(devices)}
    names = list(devices)

    key = f'exp{exp}_{protocol}_{tag_days}_c{int(cfo_correct)}_dev{"-".join(map(str,devices))}'
    cpath = os.path.join(CACHE_DIR, key + '.npz')
    if cache and os.path.exists(cpath) and not rebuild:
        z = np.load(cpath)
        if verbose:
            print(f'  (cache {cpath})')
        return (z['Xtr'], z['ytr'], z['Xva'], z['yva'], z['vfid'],
                id2label, names, json.loads(str(z['counts'])))

    Xtr, ytr, Xva, yva, vfid, counts = [], [], [], [], [], {}
    fid = 0
    for d in devices:
        lab = id2label[d]
        ntr = nva = 0
        # training windows: train_src days, overlapping hop (augment); within-day excludes val_run
        for dd in train_src:
            for cap in _runs_for(root, dd, exp, d):
                if protocol == 'within-day' and _run_no(cap) == val_run:
                    continue
                for w in _windows_from_cap(cap, hop_train, thr_mult, cfo_correct):
                    Xtr.append(w); ytr.append(np.full(len(w), lab)); ntr += 1
        # validation windows: val_src days, non-overlapping (hop=WIN); within-day only val_run
        for dd in val_src:
            for cap in _runs_for(root, dd, exp, d):
                if protocol == 'within-day' and _run_no(cap) != val_run:
                    continue
                for w in _windows_from_cap(cap, WIN, thr_mult, cfo_correct):
                    Xva.append(w); yva.append(np.full(len(w), lab))
                    vfid.append(np.full(len(w), fid)); fid += 1; nva += 1
        counts[d] = {'train_frames': ntr, 'val_frames': nva}
        if verbose:
            print(f'  {d} (label {lab}): {ntr} train / {nva} val frames')

    if not Xtr or not Xva:
        raise SystemExit('empty train or val set — check --protocol / day / run availability.')
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr).astype(np.int64)
    Xva = np.concatenate(Xva); yva = np.concatenate(yva).astype(np.int64)
    vfid = np.concatenate(vfid).astype(np.int64)
    if cache:
        np.savez(cpath, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, vfid=vfid, counts=json.dumps(counts))
    return Xtr, ytr, Xva, yva, vfid, id2label, names, counts


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
                    help='explicit comma list of folder names (e.g. device_1,device_2 or B200,USRP2), else auto-discover')
    ap.add_argument('--cfo-correct', action='store_true')
    ap.add_argument('--rebuild', action='store_true')
    a = ap.parse_args()
    devices = a.devices.split(',') if a.devices else None
    print(f'ff_dataset  exp{a.exp}  {a.protocol}  root {a.root}')
    Xtr, ytr, Xva, yva, vfid, id2label, names, counts = build_dataset(
        a.root, a.exp, a.protocol, train_day=a.train_day, val_day=a.val_day,
        day=a.day, val_run=a.val_run, devices=devices,
        cfo_correct=a.cfo_correct, rebuild=a.rebuild)
    print(f'\n  devices ({len(names)}): {names}')
    print(f'  label map: {id2label}')
    print(f'  train windows {Xtr.shape}  val windows {Xva.shape}  val frames {len(np.unique(vfid))}')


if __name__ == '__main__':
    main()

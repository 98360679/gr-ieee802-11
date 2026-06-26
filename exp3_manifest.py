#!/usr/bin/env python3
"""
exp3_manifest.py — ground-truth device manifest for a capture session
──────────────────────────────────────────────────────────────────────
Records, AT CAPTURE TIME, which PHYSICAL radio is which `device_N` label and
its role (victim / adversary), plus the fixed RX/capture parameters. This is the
single source of truth for device identity — the thing whose absence forced the
session13 recollection (see the recollect decision). The extractor/trainer read
labels from here instead of guessing from folder names.

Schema (manifest.json):
{
  "session": "session14",
  "created": "2026-06-25",
  "capture": {
    "samp_rate_hz":   5000000,
    "center_freq_hz": 2450000000,
    "rx_serial":      "<NEW RX serial>",   # the ONE receiver, fixed all session
    "rx_antenna":     "RX2",
    "rx_gain_norm":   0.75,
    "tx_period_ms":   300
  },
  "layout": "{label}/clean_run_{run}.bin",
  "devices": [
    {"label":"device_1","tx_id":"<serial/sticker>","role":"victim","runs":[1,2,3],"notes":""},
    {"label":"device_3","tx_id":"<serial/sticker>","role":"victim","runs":[1,2,3],"notes":""},
    {"label":"device_4","tx_id":"<serial/sticker>","role":"victim","runs":[1,2,3],"notes":""},
    {"label":"device_5","tx_id":"<serial/sticker>","role":"victim","runs":[1,2,3],"notes":""},
    {"label":"device_6","tx_id":"<serial/sticker>","role":"adversary","runs":[1],"notes":"transmits perturbation"}
  ]
}

Convention: victims are the fingerprint classes; their class indices are assigned
in ASCENDING label-number order (device_1->0, device_3->1, ...) — exactly the
rule exp3_train_fingerprint uses — so class_labels(manifest) matches the model.

CLI:
  python3 exp3_manifest.py --template > manifest.json          # emit a blank to fill in
  python3 exp3_manifest.py --validate manifest.json            # schema + placeholder check
  python3 exp3_manifest.py --validate manifest.json --root DIR # also check the .bin files exist
"""
import os
import re
import sys
import json
import argparse

ROLES = ('victim', 'adversary')

TEMPLATE = {
    "session": "session14",
    "created": "YYYY-MM-DD",
    "capture": {
        "samp_rate_hz": 5000000,
        "center_freq_hz": 2450000000,
        "rx_serial": "<NEW RX serial>",
        "rx_antenna": "RX2",
        "rx_gain_norm": 0.75,
        "tx_period_ms": 300,
    },
    "layout": "{label}/clean_run_{run}.bin",
    "devices": [
        {"label": "device_1", "tx_id": "<serial/sticker>", "role": "victim",
         "runs": [1, 2, 3], "notes": ""},
        {"label": "device_3", "tx_id": "<serial/sticker>", "role": "victim",
         "runs": [1, 2, 3], "notes": ""},
        {"label": "device_4", "tx_id": "<serial/sticker>", "role": "victim",
         "runs": [1, 2, 3], "notes": ""},
        {"label": "device_5", "tx_id": "<serial/sticker>", "role": "victim",
         "runs": [1, 2, 3], "notes": ""},
        {"label": "device_6", "tx_id": "<serial/sticker>", "role": "adversary",
         "runs": [1], "notes": "transmits perturbation"},
    ],
}


def _label_num(label):
    m = re.fullmatch(r'device_(\d+)', label)
    return int(m.group(1)) if m else None


def load(path):
    with open(path) as f:
        return json.load(f)


def victims(m):
    """Victim device records, sorted by ascending label number."""
    v = [d for d in m['devices'] if d.get('role') == 'victim']
    return sorted(v, key=lambda d: _label_num(d['label']))


def adversaries(m):
    return [d for d in m['devices'] if d.get('role') == 'adversary']


def class_labels(m):
    """{label: class_index} over victims, ascending label order (matches trainer)."""
    return {d['label']: i for i, d in enumerate(victims(m))}


def capture_files(m, root, include_adversary=False):
    """Expand the layout into [(path, label, class_idx_or_None, run, role), ...]."""
    layout = m.get('layout', '{label}/clean_run_{run}.bin')
    cls = class_labels(m)
    out = []
    devs = m['devices'] if include_adversary else \
        [d for d in m['devices'] if d.get('role') == 'victim']
    for d in devs:
        for r in d.get('runs', []):
            rel = layout.format(label=d['label'], run=r)
            out.append((os.path.join(root, rel), d['label'],
                        cls.get(d['label']), r, d.get('role')))
    return out


def validate(m, root=None):
    """Return a list of problem strings (empty list == valid)."""
    problems = []

    for key in ('session', 'capture', 'devices', 'layout'):
        if key not in m:
            problems.append(f"missing top-level key: {key}")
    cap = m.get('capture', {})
    for key in ('samp_rate_hz', 'center_freq_hz', 'rx_serial'):
        if key not in cap:
            problems.append(f"capture missing: {key}")
    if str(cap.get('rx_serial', '')).startswith('<'):
        problems.append("capture.rx_serial is still a placeholder")

    devs = m.get('devices', [])
    if not devs:
        problems.append("no devices listed")
    labels, tx_ids = [], []
    for d in devs:
        lab = d.get('label', '')
        if _label_num(lab) is None:
            problems.append(f"bad label (want device_<n>): {lab!r}")
        labels.append(lab)
        if d.get('role') not in ROLES:
            problems.append(f"{lab}: role must be one of {ROLES}, got {d.get('role')!r}")
        tx = str(d.get('tx_id', ''))
        if not tx or tx.startswith('<'):
            problems.append(f"{lab}: tx_id is empty/placeholder — fill the real radio id")
        tx_ids.append(tx)
        if not d.get('runs'):
            problems.append(f"{lab}: no runs listed")
    # uniqueness
    for name, seq in (('label', labels), ('tx_id', tx_ids)):
        dups = sorted({x for x in seq if seq.count(x) > 1 and not str(x).startswith('<')})
        if dups:
            problems.append(f"duplicate {name}(s): {dups}")
    if not victims(m):
        problems.append("no victim devices — need at least one fingerprint class")

    if root is not None:
        for path, lab, _idx, run, _role in capture_files(m, root, include_adversary=True):
            if not os.path.exists(path):
                problems.append(f"missing capture file: {path}")
    return problems


def main():
    ap = argparse.ArgumentParser(description="device manifest tool")
    ap.add_argument('--template', action='store_true', help='print a blank manifest')
    ap.add_argument('--validate', metavar='manifest.json', help='validate a manifest')
    ap.add_argument('--root', default=None, help='also check capture files exist under DIR')
    a = ap.parse_args()

    if a.template:
        print(json.dumps(TEMPLATE, indent=2))
        return
    if not a.validate:
        ap.error("use --template or --validate")

    m = load(a.validate)
    problems = validate(m, root=a.root)
    cls = class_labels(m)
    print(f"session : {m.get('session')}")
    print(f"victims : {cls}  (class indices)")
    print(f"adversary: {[d['label'] for d in adversaries(m)]}")
    if problems:
        print(f"\nINVALID — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nOK — manifest is valid.")


if __name__ == '__main__':
    main()

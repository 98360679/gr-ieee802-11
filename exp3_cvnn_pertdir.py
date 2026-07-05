#!/usr/bin/env python3
"""
exp3_cvnn_pertdir.py — per-frame EOT-robust targeted δ against the CVNN, for OTA (Exp 2/3).

Unlike Exp 1's plain digital PGD, an OTA δ must survive sub-sample time-shift + phase rotation
(alignment error over the air), so each PGD step averages the gradient over n_eot random
(shift, phase) transforms (EOT). Crafts one δ per TX frame-id (cycling the CVNN-domain frames),
writes the bare active-region δ as <out>/<id>.bin — consumed by exp3_assemble_pert to build
the 2-channel (Exp 3) and single-channel precombine (Exp 2) bundles.

  python3 exp3_cvnn_pertdir.py --model fingerprint_cvnn_7_03_attack.pt \
      --capture .../7_03_2026/device_6/clean_run_1.bin --target 4 --psr -10 \
      --ids-csv .../ota_dev6/eot_t4_ids.txt --out .../ota_dev6/eot_cvnn_pertdir
"""
import os, re, csv, math, argparse, numpy as np, torch, torch.nn.functional as F
from exp3_cvnn_attack import load_cvnn, logits_of, predict, C64, STARTS
from exp3_fp_model import PRE_ROLL, ACTIVE, WIN, NUM_CLASSES
from exp3_extract_frames import extract_frames_for_file

A0, A1 = PRE_ROLL, PRE_ROLL + ACTIVE


def frac_shift(x, s):
    """differentiable sub-sample shift of a 1-D complex tensor via FFT phase ramp."""
    N = x.shape[-1]
    k = torch.fft.fftfreq(N, device=x.device)
    return torch.fft.ifft(torch.fft.fft(x) * torch.exp(-2j * math.pi * k * s))


def craft_eot(model, frame_np, true, target, psr_db, dev, steps, n_eot, shift, phase_deg, step_frac, adv_eot=False):
    frame = torch.tensor(frame_np, dtype=torch.complex64, device=dev)
    budget = 10 ** (psr_db / 20) * float((frame[A0:A1].abs() ** 2).sum()) ** 0.5
    mask = torch.zeros(frame.shape, device=dev); mask[A0:A1] = 1.0
    delta = torch.zeros(frame.shape, dtype=torch.complex64, device=dev, requires_grad=True)
    untargeted = target is None
    y = torch.full((len(STARTS),), true if untargeted else target, device=dev)
    for _ in range(steps):
        if delta.grad is not None:
            delta.grad = None
        g = torch.zeros_like(delta)
        for _ in range(n_eot):
            if adv_eot:                       # 2-ch: transform δ ALONE (adversary channel h_a), frame fixed
                p = delta * mask
                if shift:
                    p = frac_shift(p, (torch.rand(1).item() * 2 - 1) * shift)
                if phase_deg:
                    ph = (torch.rand(1).item() * 2 - 1) * math.radians(phase_deg)
                    p = p * (math.cos(ph) + 1j * math.sin(ph))
                u = frame + p
            else:                             # single-ch: one shared transform on the composite
                u = frame + delta * mask
                if shift:                                          # skip when --no-eot -> plain PGD
                    u = frac_shift(u, (torch.rand(1).item() * 2 - 1) * shift)
                if phase_deg:
                    ph = (torch.rand(1).item() * 2 - 1) * math.radians(phase_deg)
                    u = u * (math.cos(ph) + 1j * math.sin(ph))
            loss = F.cross_entropy(logits_of(model, u), y)
            g = g + torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            direction = (g if untargeted else -g)
            dn = (direction * mask).abs().norm() + 1e-12
            delta.add_(step_frac * budget * direction / dn)
            cur = (delta * mask).abs().norm()
            if cur > budget:
                delta.mul_(budget / cur)
    return (delta * mask).detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--capture', required=True)
    p.add_argument('--device', type=int, default=6); p.add_argument('--target', type=int, default=4)
    p.add_argument('--untargeted', action='store_true')
    p.add_argument('--psr', type=float, default=-10.0)
    p.add_argument('--steps', type=int, default=100); p.add_argument('--n-eot', type=int, default=8)
    p.add_argument('--shift', type=float, default=1.5); p.add_argument('--phase', type=float, default=90.0)
    p.add_argument('--no-eot', action='store_true',
                   help='plain PGD, no EOT (n_eot=1, shift=0, phase=0)')
    p.add_argument('--adv-eot', action='store_true',
                   help='2-channel: transform δ ALONE (adversary channel h_a), frame fixed')
    p.add_argument('--step-frac', type=float, default=0.1)
    p.add_argument('--ids-csv', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = load_cvnn(a.model, dev); true = a.device - 1; tgt = a.target - 1
    target = None if a.untargeted else tgt

    txt = open(a.ids_csv).read()
    ids = ([int(r['frame_id']) for r in csv.DictReader(open(a.ids_csv))]
           if 'frame_id' in txt.splitlines()[0] else [int(t) for t in re.findall(r'\d+', txt)])
    allf = extract_frames_for_file(a.capture, floor_pct=20.0, thr_mult=2.0)[0]
    frames = [np.asarray(f, C64) for f in allf
              if predict(m, torch.tensor(np.asarray(f, C64), device=dev)) == true]
    n_eot = 1 if a.no_eot else a.n_eot
    shift = 0.0 if a.no_eot else a.shift
    phase = 0.0 if a.no_eot else a.phase
    os.makedirs(a.out, exist_ok=True)
    mode = f"UNTARGETED off device_{a.device}" if a.untargeted else f"device_{a.device}->device_{a.target}"
    eot = ("no-EOT (plain PGD)" if a.no_eot else
           f"{'ADV-channel ' if a.adv_eot else ''}EOT shift±{a.shift}/phase±{a.phase} x{a.n_eot}")
    print(f"{a.model}: {mode}  PSR {a.psr}  {eot}\n"
          f"  {len(ids)} ids, {len(frames)} clean device_{a.device} frames -> {a.out}")

    hit = 0
    for j, fid in enumerate(ids):
        fn = frames[j % len(frames)]
        d = craft_eot(m, fn, true, target, a.psr, dev, a.steps, n_eot, shift, phase, a.step_frac, a.adv_eot)
        d[A0:A1].cpu().numpy().astype(C64).tofile(os.path.join(a.out, f"{fid}.bin"))
        pred = predict(m, torch.tensor(fn, device=dev) + d)
        hit += (pred != true) if a.untargeted else (pred == tgt)
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(ids)}  (nominal hit {hit}/{j+1})")
    lbl = f"off device_{a.device}" if a.untargeted else f"->device_{a.target}"
    print(f"\ncrafted {len(ids)} EOT δ -> {a.out}/*.bin   nominal {lbl}: {100*hit/len(ids):.0f}%")


if __name__ == '__main__':
    main()

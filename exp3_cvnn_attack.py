#!/usr/bin/env python3
"""
exp3_cvnn_attack.py — PGD/FGSM adversarial attack against the CVNN fingerprinter (Exp 1).

Exp 1 = RECEIVER-SIDE (digital) attack: craft δ on the *received* frame and evaluate the
CVNN in software (the "does the attack work when δ reaches the input" ceiling). Targeted
(device_6 -> device_4) and untargeted (off device_6), PGD and FGSM, PSR sweep -30..+15 dB.
δ is confined to the frame's active region; budget = 10^(PSR/10) * ||active||^2 (L2).

The CVNN carries complex IQ as (re, im) real tensors, so autograd flows to the frame -> the
whole windowing + unit-RMS pipeline is differentiable and PGD/FGSM work directly.

  python3 exp3_cvnn_attack.py --model fingerprint_cvnn_7_03_attack.pt \
      --capture .../7_03_2026/device_6/clean_run_1.bin --device 6 --target 4
"""
import argparse, numpy as np, torch, torch.nn.functional as F
from exp3_cvnn import CVNN
from exp3_fp_model import PRE_ROLL, ACTIVE, WIN, NUM_CLASSES, DEVICE_NAMES
from exp3_extract_frames import extract_frames_for_file

C64 = np.complex64
STARTS = list(range(PRE_ROLL, PRE_ROLL + ACTIVE - WIN + 1, WIN))   # 14 non-overlapping windows


def load_cvnn(path, dev):
    m = CVNN(NUM_CLASSES).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.eval()
    return m


def logits_of(model, frame):
    """frame: [FRAME_LEN] complex tensor -> [n_win, nc] logits (differentiable)."""
    w = torch.stack([frame[s:s + WIN] for s in STARTS])           # [nwin, WIN] complex
    rms = torch.sqrt((w.abs() ** 2).mean(1, keepdim=True) + 1e-9)
    w = w / rms
    return model(w.real.unsqueeze(1), w.imag.unsqueeze(1))


@torch.no_grad()
def predict(model, frame_t):
    lg = logits_of(model, frame_t)
    return int(torch.bincount(lg.argmax(1), minlength=NUM_CLASSES).argmax())


def craft(model, frame_np, true, target, psr_db, method, steps, step_frac, dev):
    """Return δ (complex, active-region only). target=None => untargeted."""
    frame = torch.tensor(frame_np, dtype=torch.complex64, device=dev)
    a0, a1 = PRE_ROLL, PRE_ROLL + ACTIVE
    budget = (float((frame[a0:a1].abs() ** 2).sum()) * 10 ** (psr_db / 10)) ** 0.5
    mask = torch.zeros(frame.shape, device=dev); mask[a0:a1] = 1.0
    delta = torch.zeros(frame.shape, dtype=torch.complex64, device=dev, requires_grad=True)
    untargeted = target is None
    y = torch.full((len(STARTS),), true if untargeted else target, device=dev)
    for _ in range(steps if method == 'pgd' else 1):
        if delta.grad is not None:
            delta.grad = None
        loss = F.cross_entropy(logits_of(model, frame + delta * mask), y)
        loss.backward()
        g = delta.grad
        direction = g if untargeted else -g                       # ascend true / descend target
        with torch.no_grad():
            dn = (direction * mask).abs().norm() + 1e-12
            if method == 'fgsm':
                delta.copy_(direction / dn * budget)
            else:
                delta.add_(step_frac * budget * direction / dn)
                cur = (delta * mask).abs().norm()
                if cur > budget:
                    delta.mul_(budget / cur)
    return (delta * mask).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--capture', required=True)
    ap.add_argument('--device', type=int, default=6)
    ap.add_argument('--target', type=int, default=4)
    ap.add_argument('--nframes', type=int, default=40)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--step-frac', type=float, default=0.1)
    ap.add_argument('--psrs', type=float, nargs='+',
                    default=[-30, -25, -20, -15, -10, -5, 0, 5, 10, 15])
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = load_cvnn(a.model, dev)
    true, tgt = a.device - 1, a.target - 1

    allf = extract_frames_for_file(a.capture, floor_pct=20.0, thr_mult=2.0)[0]
    frames = [f for f in allf if predict(m, torch.tensor(np.asarray(f, C64), device=dev)) == true][:a.nframes]
    print(f"{a.model}  legit device_{a.device} -> target device_{a.target}\n"
          f"  {len(frames)} clean device_{a.device} frames (of {len(allf)})\n")

    print(f"{'PSR':>5} | {'PGD→d'+str(a.target):>9} {'PGD off':>8} | {'FGSM→d'+str(a.target):>10} {'FGSM off':>9}")
    for psr in a.psrs:
        row = {}
        for method in ('pgd', 'fgsm'):
            for mode, target in (('t', tgt), ('u', None)):
                hits = 0
                for f in frames:
                    fn = np.asarray(f, C64)
                    d = craft(m, fn, true, target, psr, method, a.steps, a.step_frac, dev)
                    pred = predict(m, torch.tensor(fn, device=dev) + d)
                    hits += (pred == tgt) if mode == 't' else (pred != true)
                row[(method, mode)] = hits / max(1, len(frames))
        print(f"{psr:>5.0f} | {row[('pgd','t')]:>9.2f} {row[('pgd','u')]:>8.2f} | "
              f"{row[('fgsm','t')]:>10.2f} {row[('fgsm','u')]:>9.2f}")


if __name__ == '__main__':
    main()

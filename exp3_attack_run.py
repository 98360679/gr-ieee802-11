#!/usr/bin/env python3
"""
exp3_attack_run.py — run + plot the channel-aware OTA attacks vs PSR/PNR
─────────────────────────────────────────────────────────────────────────
Untargeted comparison:  random baseline | UAP-PCA | UAP-opt | PGD ceiling
Targeted summary:       per-target UAP-opt success rate.
Saves a fooling-rate-vs-PSR figure, a metrics json, and the deployable UAP
(complex64) for later OTA transmission.
"""
import os, json, time, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import exp3_attack_lib as A
import exp3_attacks as K

OUT = '/media/nghoselab/T9/Data/session12/processed'
PSRS = [-40, -35, -30, -25, -20, -15]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--build-psr', type=float, default=-25)
    args = ap.parse_args()
    dev = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    model, names, win = A.load_model(device=dev)
    xtr, ytr, _ = A.load_raw_windows('train')
    xte, yte, _ = A.load_raw_windows('test')
    nC = len(names)
    n0 = A.estimate_noise_power(); ps = A.signal_power(xte).median().item()
    pnr = lambda psr: 10*np.log10((ps*10**(psr/10))/n0)
    clean = A.accuracy(model, xte, yte, device=dev)
    print(f"clean acc={clean:.4f}  SNR≈{10*np.log10(ps/n0):.1f}dB  noise={n0:.2e}\n")

    g = torch.Generator().manual_seed(0)
    phi = torch.rand(len(xte), generator=g) * 2*np.pi      # OTA-realistic random phase

    # build universal perturbations once (direction reused across PSR)
    print("building UAPs (untargeted)...")
    uap_pca = K.build_uap(model, xtr, ytr, psr_db=args.build_psr, device=dev, n_build=2000, steps=10)
    uap_opt = K.build_uap_opt(model, xtr, ytr, psr_db=args.build_psr, device=dev)

    # ── untargeted sweep ──
    rand_dir = A.unit_norm(torch.randn(win, generator=g)+1j*torch.randn(win, generator=g))
    res = {k: [] for k in ('random', 'uap_pca', 'uap_opt', 'pgd')}
    print("\nUNTARGETED  acc vs PSR (random phase):")
    print(" PSR  PNR   random  uap_pca  uap_opt   pgd")
    for psr in PSRS:
        al = A.alpha_for_psr(xte, psr)
        a_rand = A.accuracy(model, xte, yte, device=dev, delta=rand_dir, alpha=al, phi=phi)
        a_pca  = A.accuracy(model, xte, yte, device=dev, delta=uap_pca, alpha=al, phi=phi)
        a_opt  = A.accuracy(model, xte, yte, device=dev, delta=uap_opt, alpha=al, phi=phi)
        a_pgd  = K.pgd_attack(model, xte, yte, psr, steps=30, device=dev)['adv_acc']
        for k, v in zip(res, (a_rand, a_pca, a_opt, a_pgd)):
            res[k].append(v)
        print(f" {psr:4d} {pnr(psr):4.0f}  {a_rand:.3f}   {a_pca:.3f}    {a_opt:.3f}   {a_pgd:.3f}")

    # ── targeted: UAP-opt per target device ──
    print("\nTARGETED  UAP-opt success rate @ build PSR (force each device):")
    tgt_succ = {}
    for t in range(nC):
        d_t = K.build_uap_opt(model, xtr, ytr, psr_db=args.build_psr, target=t, device=dev, iters=200)
        al = A.alpha_for_psr(xte, args.build_psr)
        # success = fraction of NON-target windows now predicted as t
        with torch.no_grad():
            preds = []
            for i in range(0, len(xte), 512):
                rb = A.received(xte[i:i+512].to(dev), d_t.to(dev),
                                al[i:i+512].to(dev), phi[i:i+512].to(dev))
                preds.append(A.forward_logits(model, rb).argmax(1).cpu())
            preds = torch.cat(preds)
        nont = yte != t
        succ = (preds[nont] == t).float().mean().item()
        tgt_succ[names[t]] = succ
        print(f"  -> {names[t]:10s}: {succ:.3f} of other-device windows misread as {names[t]}")

    # ── plot untargeted ──
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xv = [pnr(p) for p in PSRS]
    ax.axhline(clean, ls=':', c='gray', label=f'clean ({clean:.2f})')
    ax.plot(xv, res['random'],  'o-', c='#888', label='random noise')
    ax.plot(xv, res['uap_pca'], 's-', c='#1f77b4', label='UAP (PCA, Kim Alg.3)')
    ax.plot(xv, res['uap_opt'], 'D-', c='#d62728', label='UAP (direct-opt, phase-robust)')
    ax.plot(xv, res['pgd'],     '^--', c='#2ca02c', label='PGD per-frame (ceiling)')
    ax.set_xlabel('PNR (dB)'); ax.set_ylabel('fingerprint accuracy')
    ax.set_title('Exp 3 — channel-aware OTA perturbation vs fingerprinter (untargeted)')
    ax.grid(alpha=0.3); ax.legend()
    secax = ax.secondary_xaxis('top', functions=(lambda v: v, lambda v: v))
    secax.set_xticks(xv); secax.set_xticklabels([str(p) for p in PSRS]); secax.set_xlabel('PSR (dB)')
    p = os.path.join(OUT, 'attack_untargeted_vs_pnr.png')
    fig.tight_layout(); fig.savefig(p, dpi=130); print(f"\nsaved figure -> {p}")

    # ── save deployable UAP (complex64) + metrics ──
    up = os.path.join(OUT, 'uap_untargeted.bin')
    uap_opt.cpu().numpy().astype(np.complex64).tofile(up)
    print(f"saved deployable UAP -> {up}  (len {win}, complex64)")
    with open(os.path.join(OUT, 'attack_metrics.json'), 'w') as f:
        json.dump(dict(clean=clean, psrs=PSRS, pnrs=[pnr(p) for p in PSRS],
                       untargeted=res, targeted=tgt_succ, build_psr=args.build_psr,
                       snr_db=10*np.log10(ps/n0)), f, indent=2)
    print(f"saved metrics -> {os.path.join(OUT,'attack_metrics.json')}")


if __name__ == '__main__':
    main()

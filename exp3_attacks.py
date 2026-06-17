#!/usr/bin/env python3
"""
exp3_attacks.py — channel-aware OTA adversarial attacks (Kim et al. 2005.05321)
─────────────────────────────────────────────────────────────────────────────────
Two attacks against the 6-device fingerprint CNN, both white-box:

  • PGD (per-frame)  — re-optimize a perturbation for every window. Strongest
    possible (white-box upper bound) but needs per-frame sync => NOT OTA-realizable.
    Used as the ceiling.

  • UAP (universal)  — ONE input-independent perturbation (Kim Alg. 3): craft
    per-input perturbations, stack, take the 1st PCA component as the UAP
    direction, scale to the power budget. Robust to random phase (and channel),
    needs no per-frame sync => the deployable OTA attack.

Perturbations are constrained by mean power; we sweep PSR (perturbation-to-signal)
and report the equivalent PNR. Supports untargeted (max CE) and targeted (force a
chosen device label).
"""
import numpy as np
import torch
import torch.nn.functional as F
import exp3_attack_lib as A


# ── PGD: per-window perturbation (the white-box ceiling) ───────────────
def pgd_attack(model, x, y, psr_db, steps=30, target=None, device='cuda:0',
               bs=1024, phi=None, h=None):
    """Return per-window adversarial accuracy at the given PSR.

    target=None -> untargeted (maximize loss on true label)
    target=int  -> targeted   (minimize loss toward `target` class)
    """
    model.eval()
    correct, total, succ = 0, 0, 0
    for i in range(0, len(x), bs):
        xb = x[i:i+bs].to(device)
        yb = y[i:i+bs].to(device)
        n = len(xb)
        budget = A.signal_power(xb) * (10.0 ** (psr_db / 10.0))   # (n,) mean-power cap
        # perturbation as real (n,2,L), start tiny
        p = (1e-3 * torch.randn(n, 2, xb.shape[-1], device=device)).requires_grad_(True)
        step = (budget.sqrt().mean().item()) * 2.0 / steps        # ~reach the ball
        tgt = None if target is None else torch.full((n,), target, device=device)
        for _ in range(steps):
            pc = torch.complex(p[:, 0], p[:, 1])
            if phi is not None:
                pc = pc * torch.polar(torch.ones(n, device=device),
                                      phi[i:i+bs].to(device)).unsqueeze(-1)
            r = xb + pc
            logits = A.forward_logits(model, r)
            if target is None:
                loss = F.cross_entropy(logits, yb)                # ascend
                g = torch.autograd.grad(loss, p)[0]
                with torch.no_grad():
                    p += step * g.sign()
            else:
                loss = F.cross_entropy(logits, tgt)               # descend
                g = torch.autograd.grad(loss, p)[0]
                with torch.no_grad():
                    p -= step * g.sign()
            # project each window onto its mean-power ball
            with torch.no_grad():
                pw = (p[:, 0] ** 2 + p[:, 1] ** 2).mean(-1)       # (n,)
                scale = torch.clamp((budget / (pw + A.EPS)).sqrt(), max=1.0)
                p *= scale.view(n, 1, 1)
            p.requires_grad_(True)
        with torch.no_grad():
            pc = torch.complex(p[:, 0], p[:, 1])
            if phi is not None:
                pc = pc * torch.polar(torch.ones(n, device=device),
                                      phi[i:i+bs].to(device)).unsqueeze(-1)
            pred = A.forward_logits(model, xb + pc).argmax(1)
            correct += (pred == yb).sum().item()
            if target is not None:
                succ += (pred == target).sum().item()
            total += n
    out = {'adv_acc': correct / total}
    if target is not None:
        out['target_success'] = succ / total
    return out


# ── UAP: universal perturbation via per-input grads + PCA (Kim Alg. 3) ──
def build_uap(model, x, y, psr_db, target=None, device='cuda:0', bs=1024,
              n_build=2000, steps=10, seed=0):
    """Craft a single complex UAP direction (L,) from `n_build` training windows.

    For each window we run a short PGD to get its optimal perturbation, stack the
    resulting (2L) real vectors, and take the first principal component as the UAP
    direction (Kim Alg. 3). Returned direction is unit mean-power.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)[:n_build]
    xb_all = x[idx].to(device)
    yb_all = y[idx].to(device)
    L = x.shape[-1]
    perts = []
    for i in range(0, len(xb_all), bs):
        xb = xb_all[i:i+bs]; yb = yb_all[i:i+bs]; n = len(xb)
        budget = A.signal_power(xb) * (10.0 ** (psr_db / 10.0))
        p = (1e-3 * torch.randn(n, 2, L, device=device)).requires_grad_(True)
        step = (budget.sqrt().mean().item()) * 2.0 / steps
        tgt = None if target is None else torch.full((n,), target, device=device)
        for _ in range(steps):
            pc = torch.complex(p[:, 0], p[:, 1])
            logits = A.forward_logits(model, xb + pc)
            if target is None:
                loss = F.cross_entropy(logits, yb)
                gr = torch.autograd.grad(loss, p)[0]
                with torch.no_grad(): p += step * gr.sign()
            else:
                loss = F.cross_entropy(logits, tgt)
                gr = torch.autograd.grad(loss, p)[0]
                with torch.no_grad(): p -= step * gr.sign()
            with torch.no_grad():
                pw = (p[:, 0] ** 2 + p[:, 1] ** 2).mean(-1)
                scale = torch.clamp((budget / (pw + A.EPS)).sqrt(), max=1.0)
                p *= scale.view(n, 1, 1)
            p.requires_grad_(True)
        perts.append(p.detach().reshape(n, 2 * L).cpu())
    M = torch.cat(perts).numpy()                       # (n_build, 2L)
    M = M - M.mean(0, keepdims=True)
    # first principal component via SVD
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    v = Vt[0]                                           # (2L,) dominant direction
    delta = torch.complex(torch.from_numpy(v[:L]).float(),
                          torch.from_numpy(v[L:]).float())
    delta = A.unit_norm(delta)
    # PCA sign is arbitrary — pick the sign that raises loss more (untargeted)
    # or lowers target loss more (targeted), evaluated on the build set.
    xs = xb_all[:512]; ys = yb_all[:512]
    al = A.alpha_for_psr(xs, psr_db).to(device)
    def avg_loss(dl, lab):
        r = A.received(xs, dl.to(device), al)
        return F.cross_entropy(A.forward_logits(model, r), lab).item()
    if target is None:
        keep = delta if avg_loss(delta, ys) >= avg_loss(-delta, ys) else -delta
    else:
        tg = torch.full((len(xs),), target, device=device)
        keep = delta if avg_loss(delta, tg) <= avg_loss(-delta, tg) else -delta
    return keep


def build_uap_opt(model, x, y, psr_db, target=None, device='cuda:0', bs=512,
                  n_build=4000, iters=300, lr=0.02, seed=0):
    """Directly optimize ONE universal complex direction (unit mean-power).

    Unlike the PCA build, this maximizes the *average* attack objective over the
    build set, with a fresh random phase per window each iteration so the UAP is
    explicitly robust to the unknown adversary->receiver phase (OTA-realistic).
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)[:n_build]
    xb_all = x[idx].to(device); yb_all = y[idx].to(device)
    L = x.shape[-1]
    d = (torch.randn(2, L, device=device)); d /= d.norm()
    d.requires_grad_(True)
    opt = torch.optim.Adam([d], lr=lr)
    nb = len(xb_all)
    for it in range(iters):
        j = torch.randint(0, nb, (bs,), generator=g)
        xb = xb_all[j]; yb = yb_all[j]
        dl = A.unit_norm(torch.complex(d[0], d[1]))         # unit mean-power
        al = A.alpha_for_psr(xb, psr_db).to(device)
        phi = torch.rand(bs, device=device) * 2 * np.pi     # random phase robustness
        r = A.received(xb, dl, al, phi)
        logits = A.forward_logits(model, r)
        if target is None:
            loss = -F.cross_entropy(logits, yb)             # maximize CE
        else:
            loss = F.cross_entropy(logits, torch.full((bs,), target, device=device))
        opt.zero_grad(); loss.backward(); opt.step()
    return A.unit_norm(torch.complex(d.detach()[0], d.detach()[1]))

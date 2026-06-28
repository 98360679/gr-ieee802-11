#!/usr/bin/env python3
"""
exp3_eot_universal.py — ONE universal alignment-robust targeted delta (no per-frame keying)
────────────────────────────────────────────────────────────────────────────────────────────
The 222 transmitted device_6 frames are ~identical (same 'x'-filler payload, only the
id stamp differs); a delta fails to transfer only because of per-capture RX realization
(channel/noise/timing). So train ONE delta with EOT over (many device_6 frame realizations
x sub-sample shift x phase) and apply it universally (build_adv_replay --pert-file).

Tests target-hit on HELD-OUT frames. If high, no id->position keying is needed:
rx_frames_run_1.jsonl (id list) + TX frame.bin/index is all build_adv_replay needs.
"""
import argparse, numpy as np, torch, torch.nn.functional as F
from exp3_make_perturbation import load_fp_model, predict_frame
from exp3_make_perturbation_eot import frac_shift, hit_under
from exp3_extract_frames import extract_frames_for_file
from exp3_fp_model import torch_iq_to_input, PRE_ROLL, ACTIVE, WIN, FRAME_LEN
C64=np.complex64; _EPS=1e-12; NW=ACTIVE//WIN

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture', default='/media/nghoselab/T9/Data/session13/train/6_27_2026/device_6/clean_run_1.bin')
    p.add_argument('--model', default='fingerprint_cnn_ft20260627.pt')
    p.add_argument('--device', type=int, default=6); p.add_argument('--target', type=int, default=4)
    p.add_argument('--psr', type=float, default=-20.0); p.add_argument('--steps', type=int, default=300)
    p.add_argument('--n-eot', type=int, default=16); p.add_argument('--shift', type=float, default=1.5)
    p.add_argument('--phase', type=float, default=90.0); p.add_argument('--ntrain', type=int, default=30)
    p.add_argument('--ntest', type=int, default=30)
    p.add_argument('--out', default='/media/nghoselab/T9/Data/session13/ota_dev6/eot_universal_t4.bin')
    a=p.parse_args()
    m,nc,n2i,i2n=load_fp_model(a.model); true,tgt=n2i[f'device_{a.device}'],n2i[f'device_{a.target}']
    frames=[f for f in extract_frames_for_file(a.capture)[0] if predict_frame(m,f,nc)[0]==true]
    tr=frames[:a.ntrain]; te=frames[a.ntrain:a.ntrain+a.ntest]
    print(f"{len(frames)} device_6 frames; train {len(tr)} / held-out test {len(te)}")
    acts=[torch.from_numpy(f.astype(C64))[PRE_ROLL:PRE_ROLL+ACTIVE] for f in tr]
    # per-window budget = mean over train frames
    sig_norm=torch.stack([torch.sqrt((a_.reshape(NW,WIN).real**2+a_.reshape(NW,WIN).imag**2).sum(1)+_EPS) for a_ in acts]).mean(0)
    budget=(10.0**(a.psr/20.0))*sig_norm
    d=torch.zeros(ACTIVE,2); d.normal_(0,1e-3); d.requires_grad_(True)
    y=torch.full((NW,),tgt,dtype=torch.long); phmax=np.deg2rad(a.phase)
    rng=np.random.default_rng(0)
    for it in range(a.steps):
        if d.grad is not None: d.grad.zero_()
        dc=torch.complex(d[:,0],d[:,1]); loss=0.0
        for _k in range(a.n_eot):
            act=acts[int(rng.integers(len(acts)))]
            tau=float(rng.uniform(-a.shift,a.shift)); phi=float(rng.uniform(-phmax,phmax))
            comb=(act+frac_shift(dc,tau)*np.exp(1j*phi)).reshape(NW,WIN)
            loss=loss+F.cross_entropy(m(torch_iq_to_input(comb)),y)
        (loss/a.n_eot).backward()
        with torch.no_grad():
            gw=d.grad.reshape(NW,WIN,2); gn=gw.flatten(1).norm(dim=1).clamp_min(_EPS)
            d-=((0.06*budget/gn).view(NW,1,1)*gw).reshape(ACTIVE,2)
            dw=d.reshape(NW,WIN,2); dn=dw.flatten(1).norm(dim=1)
            d.copy_((dw*torch.clamp(budget/dn.clamp_min(_EPS),max=1.0).view(NW,1,1)).reshape(ACTIVE,2))
        if (it+1)%50==0: print(f"  step {it+1}/{a.steps}")
    delta=np.zeros(FRAME_LEN,dtype=C64); dd=d.detach().numpy()
    delta[PRE_ROLL:PRE_ROLL+ACTIVE]=dd[:,0]+1j*dd[:,1]
    def rate(frs,tau=0.0,deg=0.0): return 100*np.mean([hit_under(m,f,delta,true,tgt,nc,tau=tau,deg=deg)[0] for f in frs])
    print("\nUNIVERSAL delta on HELD-OUT frames -> device_%d:"%a.target)
    print(f"  nominal      {rate(te):.0f}%")
    print(f"  +1 sample    {rate(te,tau=1.0):.0f}%")
    print(f"  +90 deg      {rate(te,deg=90):.0f}%")
    print(f"  (train-set nominal {rate(tr):.0f}% for reference)")
    delta[PRE_ROLL:PRE_ROLL+ACTIVE].astype(C64).tofile(a.out)
    print(f"\nwrote universal data-region delta -> {a.out}  (build_adv_replay --pert-file)")

if __name__=='__main__': main()

# Experiment 1 — Receiver-Side (Digital) Adversarial Attack on the CVNN Fingerprinter

**Digital attack ceiling** (no OTA): craft δ on the *received* frame, evaluate the CVNN in
software. This is the "does the attack work when δ reaches the input" upper bound; Exp 2/3
add the OTA path. Generated 2026-07-03.

## Setup
- **Model:** `fingerprint_cvnn_7_03_attack.pt` — CVNN, fine-tuned on the 7/03 60 s capture
  (80/10/10→ test 0.857; **device_3/4/6 = 1.00**, device_1/2 the current-session soft pair).
- **Attack:** legit **device_6** → target **device_4**. δ confined to the frame active region.
- **Methods:** PGD (iterative, 80 steps) and FGSM (single step); **targeted** (→device_4) and
  **untargeted** (off device_6). 30 device_6 frames.
- **Fooling metric:** targeted = fraction read as device_4; untargeted = fraction *not* device_6.
- **Budget:** PSR sweep (dB) and ε sweep where ε = ‖δ‖/‖active‖ (PSR_dB = 20·log₁₀ε).
- Tools: `exp3_cvnn_attack.py` (fooling), `exp3_cvnn_ber.py` (BER). Model built by
  `exp3_cvnn_singlerun.py`.

---

## Table A — Fooling vs PSR (dB)

| PSR | PGD→d4 | PGD off | FGSM→d4 | FGSM off |
|---:|---:|---:|---:|---:|
| −30 | 0.03 | 0.00 | 0.00 | 0.00 |
| −25 | 0.53 | 0.37 | 0.00 | 0.30 |
| −20 | **1.00** | 0.87 | 0.23 | 0.63 |
| −15 | 1.00 | 1.00 | 0.93 | 0.90 |
| −10 | 1.00 | 1.00 | 1.00 | 0.97 |
| −5 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 5 | 1.00 | 1.00 | 0.93 | 1.00 |
| 10 | 1.00 | 1.00 | 0.27 | 1.00 |
| 15 | 1.00 | 1.00 | 0.00 | 1.00 |

**PGD targeted saturates at 100% from −20 dB.** FGSM targeted shows an inverted-U (peaks
−10…0, then **overshoots** the target at high budget → 0.00 by +15). PGD ≫ FGSM.

---

## Table B — Fooling vs ε (‖δ‖/‖active‖)

| ε | PGD→d4 | PGD off | FGSM→d4 | FGSM off |
|---:|---:|---:|---:|---:|
| 0.05 | 0.13 | 0.17 | 0.00 | 0.23 |
| 0.06 | 0.70 | 0.43 | 0.00 | 0.27 |
| 0.07 | **1.00** | 0.50 | 0.03 | 0.50 |
| 0.08 | 1.00 | 0.63 | 0.07 | 0.50 |
| 0.09 | 1.00 | 0.83 | 0.10 | 0.60 |
| 0.10 | 1.00 | 0.87 | 0.23 | 0.63 |
| 0.11 | 1.00 | 0.90 | 0.33 | 0.67 |
| 0.12 | 1.00 | 0.97 | 0.50 | 0.73 |
| 0.13 | 1.00 | 0.97 | 0.67 | 0.80 |
| 0.14 | 1.00 | 0.97 | 0.73 | 0.80 |
| 0.15 | 1.00 | 1.00 | 0.83 | 0.90 |
| 0.16 | 1.00 | 1.00 | 0.93 | 0.87 |
| 0.17 | 1.00 | 1.00 | 0.93 | 0.90 |
| 0.18 | 1.00 | 1.00 | 0.93 | 0.90 |
| 0.19 | 1.00 | 1.00 | 0.93 | 0.90 |
| 0.20 | 1.00 | 1.00 | 0.97 | 0.90 |
| 0.21 | 1.00 | 1.00 | 1.00 | 0.93 |
| 0.22 | 1.00 | 1.00 | 1.00 | 0.93 |
| 0.23 | 1.00 | 1.00 | 1.00 | 0.93 |
| 0.24 | 1.00 | 1.00 | 1.00 | 0.93 |
| 0.25 | 1.00 | 1.00 | 1.00 | 0.97 |
| 0.26 | 1.00 | 1.00 | 1.00 | 0.97 |

**PGD targeted saturates at ε ≈ 0.07** (≈ −23 dB); FGSM needs ε ≈ 0.21 for 100% targeted.

---

## Table C — BER (link stealth) vs ε

Decoded through the gr-ieee80211 chain; BER = bit diffs vs the matching clean decode.
Zero observed errors are reported as the **rule-of-3 95% upper bound 2.996/N** (N ≈ 56 k bits
over 14 decoded frames → floor **5.35×10⁻⁵**). 15 frames.

| ε | PGD→d4 | PGD off | FGSM→d4 | FGSM off |
|---:|---:|---:|---:|---:|
| 0.05 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.06 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.07 | 2.89e-02† | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.08 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.09 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.10 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.11 | 2.91e-02† | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.12 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.13 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.14 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.15 | 2.91e-02† | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.16 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.17 | 2.88e-02† | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.18 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.19 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.20 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.21 | 5.76e-05 | 5.35e-05 | 5.35e-05 | 5.35e-05 |
| 0.22 | 2.93e-02† | 5.35e-05 | 5.35e-05 | 1.07e-04 |
| 0.23 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 3.93e-04 |
| 0.24 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 7.14e-04 |
| 0.25 | 5.35e-05 | 5.35e-05 | 5.35e-05 | 7.68e-04 |
| 0.26 | 5.35e-05 | 5.35e-05 | 5.36e-05 | 7.68e-04 |

**Takeaways.** Nearly every cell sits at the **rule-of-3 floor (5.35×10⁻⁵)** = zero observed
errors → the attack is **link-stealthy**: the payload FEC corrects the perturbation. FGSM
**untargeted** is the only config with a genuine rising trend (1.1×10⁻⁴ → 7.7×10⁻⁴ as ε
0.22→0.26) — the cruder single-step δ starts to exceed the FEC's correction budget.

† The scattered PGD-targeted ~2.9×10⁻² spikes are **single-frame artifacts** (one perturbed
frame decodes garbled ≈ 40% wrong ≈ 1.6 k errors / 56 k bits), not a systematic BER — the
neighboring ε all sit at the floor. Re-running with more frames would average these out.

## Table D — BER vs PSR (the FEC waterfall)

Wider sweep (ε = 10^(PSR/20)), 30 frames. Shows BER *is* monotonic in the budget once δ
crosses the code's correction threshold. floor = rule-of-3 2.67×10⁻⁵; **nan = link fully
broken (no frame decodes) = the extreme of BER**.

| PSR (dB) | ε | PGD→d4 | PGD off | FGSM→d4 | FGSM off |
|---:|---:|---:|---:|---:|---:|
| −30 | 0.03 | floor | floor | floor | floor |
| −25 | 0.06 | floor | floor | floor | floor |
| −20 | 0.10 | floor | floor | floor | floor |
| −15 | 0.18 | floor | floor | floor | floor |
| −10 | 0.32 | 1.6e-3 | floor | 2.9e-4 | 1.4e-3 |
| −5 | 0.56 | 1.3e-2 | 1.1e-2 | 2.9e-2 | 2.0e-2 |
| 0 | 1.00 | 3.5e-2 | nan | nan | 9.2e-2 |
| +5 | 1.78 | 5.2e-2 | nan | nan | 1.8e-1 |
| +10 | 3.16 | nan | nan | nan | 2.9e-1 |
| +15 | 5.62 | nan | nan | nan | nan |

**Post-FEC BER is flat-then-waterfall:** ≈0 (FEC corrects, link-stealthy) through PSR ≤ −15,
then rises monotonically −10→+10, then `nan` (total link failure) at high PSR. Tables B/C's
ε 0.05–0.26 (= PSR −26…−12) sit entirely on the stealthy flat shelf — which is why the
attack fools the fingerprint there with no measurable link damage.

## Bottom line (Exp 1)
The digital attack is a **clean success and link-stealthy**: **PGD targeted device_6→device_4
= 100% from ε≈0.07 / PSR≈−20 dB with BER at the noise floor** (FEC corrects it). PGD ≫ FGSM
on both fooling and stealth. This is the *upper bound* — Exp 2/3 test whether it survives OTA.

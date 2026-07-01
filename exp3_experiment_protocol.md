# Experiment Protocol — Adversarial Perturbations vs an RF-Fingerprint Classifier

**Session13 · legit TX = device_6 · fingerprinter = `fingerprint_cnn_ft20260630varied.pt`**
(honest content-disjoint accuracy **0.95**; device_6 target separability **1.000**).

## Thesis / organizing principle

A single adversarial perturbation δ is injected at **three points along the signal chain**,
relative to the hardware that *creates* the RF fingerprint (the PA/RF front-end). The fooling
rate collapses as δ moves **upstream** of that hardware, and the last transmitter's fingerprint
governs the outcome. This isolates *why* content-domain adversarial attacks (Kim et al.,
arXiv:2005.05321) do **not** transfer to hardware-domain RF-fingerprint classifiers.

**Independent variable:** injection point of δ (receiver input → legit TX → adversary TX).
**Controlled:** same δ, same model, same content (222 replay frames), fixed gain.

| | δ injected at | passes through | fingerprint(s) present | expected fooling |
|---|---|---|---|---|
| **Exp 1** | receiver input (post-hardware) | no radio | device_6 (in the captured frame) | **high** |
| **Exp 2** | TX file, **legit** replay | device_6 PA | device_6 only | **~0%** |
| **Exp 3** | **adversary** replay | adversary PA | device_6 ⊕ adversary | **measured** |

---

## Metrics (all experiments)

- **Fooling rate** = fraction of frames NOT classified as the legit device_6.
  - *targeted:* fraction classified as device_4 (chosen target).
  - *impersonation (Exp 3 replay):* fraction still read as the legit device_6 (**survival**).
- **BER** = bit error rate of the decoded payload vs known TX bits → **link stealth** (does δ
  break the link). Report clean (δ-off) vs attacked (δ-on) through the *same* path.
- **δ-off control** = identical path, ε=0. Mandatory every run; fooling := δ-on − δ-off.

---

## Experiment 1 — δ at the receiver input (digital feasibility ceiling)

- **Threat model:** white-box / signal-injection at the classifier input.
- **Method:** craft δ on the *received* frame (carries device_6's fingerprint); add to the
  captured IQ; classify. Targeted (→device_4) and untargeted (off-true). Sweep ε / PSR.
- **Success criterion:** fooling ≫ 0 at link-stealthy BER → confirms the attack is feasible
  *when δ reaches the feature*. (Have: targeted 94–100% digital, robust at −10 dB.)
- **Status:** mostly done; NEED tabulated untargeted numbers + BER pass.

## Experiment 2 — δ in the TX file, same (legit) transmitter (upstream reality check)

- **Threat model:** closed-loop replay; device_6 transmits frame+δ; RX recaptures OTA.
- **Method:** bake δ into the transmit file (2-channel: ch0 frame / ch1 δ; single-channel:
  precombined). PSR sweep −30…+10 + δ-off. Recapture, classify, decode.
- **Success criterion (falsification):** if fooling ≈ 0 at all PSR while Exp 1 succeeded on the
  *same* δ → δ is inert upstream of the PA. **Result: 0% fooling, 2-ch and single-ch, all PSR.**
- **Status:** DONE (δ-off control present). NEED BER tabulated from `ber_frames_*.jsonl`.

## Experiment 3 — δ through an independent adversary (adversary-fingerprint term)

Two variants × two adversary types. Define success per variant.

- **Variant A — δ alone (MIMO superposition):** legit TX sends frame, adversary sends δ; RX
  sees frame(device_6) ⊕ δ(adversary). *Predicted:* reading crosses device_6 → **adversary**
  as PSR rises, **never device_4**. Success = demonstrate the crossover poles are {device_6,
  adversary}. Confirm with a **δ-only** transmit (should read the adversary).
- **Variant B — full frame+δ replay (impersonation):** adversary re-transmits the whole
  fingerprinted signal. Success = **survival** of device_6 (impersonation viable).
  *Preliminary:* device_6's recording via device_1 → reads **device_6 0.76**, device_1 **0.00**
  (device_1's PA measurably compressed it: PAPR 7.9→6.3). SURPRISING — replay may preserve
  the fingerprint.
- **Adversary type:** *enrolled* (e.g. device_1 — can detect a flip *to* the adversary) and
  *unenrolled* (generic radio — clean "does the legit fingerprint survive" readout).
- **Status:** 1 preliminary trial (Variant B, enrolled). NEED controls + replication + BER.

### Controls required before any Exp 3 claim
1. **Generalization:** device_1 replays device_3's recording → reads device_3? (effect is general, not device_6-special)
2. **Sanity:** device_1 replays its own recording → reads device_1?
3. **Radio-independence:** a different adversary (device_2) replays device_6 → still device_6?
4. **Explain** the device_5 leakage (0.23) in the device_6→device_1 double-hop.

---

## Captures still needed

- Exp 3-A: δ-only transmit (`delta_alone_dacsafe.bin`) → recapture.
- Exp 3-B: `dev6_recording_dacsafe.bin` through device_2 (radio-independence); device_1 replays
  device_3 and device_1 (controls 1–2); ≥1 unenrolled adversary.
- BER: ensure each recapture logs `ber_frames_*.jsonl` against known TX bits.

## Narrative arc (for the write-up)

δ works at the receiver input (Exp 1) → dies through the legit PA (Exp 2) → and through a
foreign PA either shifts toward the *adversary's* identity (Exp 3-A) or, on full replay,
*preserves* the legit fingerprint (Exp 3-B). Conclusion: content-domain adversarial
perturbations cannot fool a hardware-domain fingerprinter; the only physical levers are the
**transmitting hardware's own fingerprint** and **replay fidelity** — motivating hardware
mimicry (`exp3_impairment_fit.py`) as the sole targeted-attack path.

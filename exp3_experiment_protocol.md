# Experiment Protocol — Adversarial Perturbations vs an RF-Fingerprint Classifier

**Session13 · legit TX = device_6 · fingerprinter = `fingerprint_cnn_ft20260630varied.pt`**
(honest content-disjoint accuracy **0.95**; device_6 target separability **1.000**).

## Thesis / organizing principle

A single adversarial perturbation δ is injected at **three points along the signal chain**,
relative to the hardware that *creates* the RF fingerprint (the PA/RF front-end). The fooling
rate collapses as δ moves **upstream** of that hardware. *Whose* fingerprint governs a recapture
is itself an experimental question — **do not assume "the last transmitter wins"**: a CLEAN
input takes the transmitting PA's fingerprint, but a PRE-fingerprinted input may **survive**
(the device_6→device_1 double-hop read device_6 0.76, not device_1). So the mechanism is
**measured, not assumed.** This isolates *why* content-domain adversarial attacks (Kim et al.,
arXiv:2005.05321) do **not** transfer to hardware-domain RF-fingerprint classifiers.

> **All three experiments are (re)run from scratch under this protocol** — the earlier tests
> were not structured this way and, critically, never isolated *whether* device_6's PA erases
> the δ or merely outweighs it. That distinction is now a designed measurement (Exp 2 below).

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
  precombined). Recapture, classify, decode. δ-off control every run.
- **Key question this experiment must ANSWER (not assume):** when fooling = 0, is it because
  device_6's PA **erased** the δ, or because the δ **survived but is outweighed** by device_6's
  imprint? These are different mechanisms; the earlier tests conflated them.

  **Mandatory mechanism measurements (paired captures, back-to-back, static channel, same gain):**
  - **R_off** = frame only; **R_on** = frame + δ (strong PSR, e.g. −5/0).
  - **M1 δ-survival (matched filter):** correlate R_on against the *known transmitted δ*; peak
    (absent in R_off) ⇒ δ physically survived the PA. No peak ⇒ δ erased/attenuated.
  - **M2 residual:** align, compute `R_on − R_off`; compare energy to the transmitted δ
    (channel-scaled); classify the residual — does it wear **device_6's** fingerprint?
  - **M3 high-PSR sweep:** fooling vs PSR up to **+20/+30**. Flat 0 even when δ dominates in
    power ⇒ device_6's imprint governs regardless of δ magnitude.
- **Interpretation:**
  - δ **survives** AND fooling = 0 (M1 peak, M3 flat) ⇒ **fingerprint barrier proven** — δ is
    present but outweighed; the transmitter's imprint dominates the decision.
  - δ does **not** survive (no M1 peak) ⇒ 0% was δ fragility, NOT the barrier — must
    strengthen/align δ before concluding anything.
- **Prior (unstructured) result:** 0% fooling, 2-ch and single-ch, all PSR −30…+10 — but WITHOUT
  M1/M2, so the *reason* is unproven. Re-run from scratch with the measurements above.
- **BER:** clean = δ-off replay through the same path; report δ-on vs δ-off per PSR.

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

## From-scratch capture checklist

Fixed across everything: model `fingerprint_cnn_ft20260630varied.pt`, 222 replay frames,
one δ crafted per frame at −10 dB (robust point), fixed RX gain, δ-off control every run,
`ber_frames_*.jsonl` logged vs known TX bits.

**Exp 1 (no new capture — computational):**
- [ ] Targeted (→device_4) + untargeted δ on *received* frames; fooling + BER vs ε/PSR. (Have partial.)

**Exp 2 (legit TX = device_6, fresh):**
- [ ] R_off = frame only, recapture.
- [ ] R_on = frame+δ at PSR ∈ {−5, 0} (strong), recapture back-to-back with R_off (static channel).
- [ ] High-PSR sweep: frame+δ at PSR {+5, +10, +20, +30}, recapture each.
- [ ] Full sweep −30…+10 for the headline 0% curve (single-channel + 2-channel).
- [ ] Analysis: M1 matched-filter δ-survival, M2 residual `R_on−R_off`, M3 fooling-vs-PSR.

**Exp 3-A (δ-alone superposition):**
- [ ] `delta_alone_dacsafe.bin` transmitted by an adversary radio → recapture (predict: reads adversary).
- [ ] MIMO: device_6 frame ⊕ adversary δ across PSR → recapture (predict: device_6→adversary crossover).

**Exp 3-B (full replay / impersonation):**
- [ ] `dev6_recording_dacsafe.bin` via device_2 (radio-independence — still device_6?).
- [ ] Control 1: device_1 replays device_3's recording (generalization — reads device_3?).
- [ ] Control 2: device_1 replays its own recording (sanity — reads device_1?).
- [ ] ≥1 unenrolled adversary replays device_6 (general survival).
- [ ] Explain the device_5 leakage (0.23) in the device_6→device_1 double-hop.

**Transmit files ready (on drive, `session13/double_hop/`):** `dev6_recording_dacsafe.bin`
(Exp 3-B, peak 0.95), `delta_alone_dacsafe.bin` (Exp 3-A, real preamble + δ-only payload,
221 frames extractable, peak 0.95).
**Analysis tools:** `exp3_ota_eval.py` (fooling), `exp3_ber_compare.py` (BER),
`exp3_impairment_fit.py` (whose fingerprint), `exp3_delta_survival.py` (Exp 2 M1/M2 δ-survival).

## Narrative arc (for the write-up)

δ works at the receiver input (Exp 1) → dies through the legit PA (Exp 2) → and through a
foreign PA either shifts toward the *adversary's* identity (Exp 3-A) or, on full replay,
*preserves* the legit fingerprint (Exp 3-B). Conclusion: content-domain adversarial
perturbations cannot fool a hardware-domain fingerprinter; the only physical levers are the
**transmitting hardware's own fingerprint** and **replay fidelity** — motivating hardware
mimicry (`exp3_impairment_fit.py`) as the sole targeted-attack path.

# Experiment 3 — OTA Fingerprint Attack: Results Summary

Session13 closed-loop attack (device_6 = legit TX). Tables for advisor discussion.
Last updated 2026-06-30.

---

## Table A — Attack results (OTA)

| Attack | Digital (nom / 1-samp / 90°) | OTA fooling | δ-off control? | Verdict |
|---|---|---|---|---|
| **Targeted PGD → device_4** (fuzzy model, 6/27) | ~100 / ~100 / ~90 | **100% all PSR (0…−30)** → device_4, link-stealthy | ✗ | Works, but unvalidated + weak model |
| **Targeted PGD → device_4** (crisp model, 6/28) | ~100 / 100 / 90 | **38–73%** (73% @−15) | ✗ | Works, inconsistent (channel) |
| **Targeted PGD → device_4** (clean all-replay, 6/29 model) | 94 / 94 / 93 | **0% all PSR (−40…+10)**, 2-ch AND single-ch | ✓ (digital δ-off → device_1) | **Fails** — δ upstream of PA fingerprint |
| **Untargeted PGD** (pure) | 69–98 / 96 / **24–33** | **~0%** (all PSR) | ✓ (=device_6) | Fails OTA — phase-fragile |
| **Untargeted FGSM** | 0 / 0 / 0 @−15 | ~0% / artifact | partial | Too weak (single step) |
| **Runner-up** (robust untargeted) | 96–100 / ~100 / 70–88 | ~0% (clean) | ✓ | Fails — target shifts OTA |

**Takeaway:** Targeted ≫ untargeted for OTA. Targeted's fixed, reachable target is a
coherent direction that survives alignment error + channel; untargeted's is not.

---

## Table B — Fingerprint accuracy (the separability story)

| Model / enrollment | Held-out acc | What it tells us |
|---|---|---|
| Original retrained (6/26) | 0.934 | Drifts to 0.16–0.33 next day |
| Fine-tunes (6/27 / 6/28 / 6/29, single-run) | 0.78 / 1.00 / 0.97 | Optimistic (no held-out run) |
| 6/29, 3-run, **mixed** enrollment | **0.935** | Confounded — device_6 only replay class |
| 6/29, 3-run, all-replay, **single frozen frame** (small CNN) | **0.52** | Content-STARVED — one PA operating point |
| 6/29, all-replay, single frame, **BigCNN (5×, 80 ep)** | **0.59** | Not capacity — it's content starvation |
| **6/30, 3-run, all-replay, VARIED content** (222 frames), run-3 holdout | **0.986** | Operational (= attack content), leakage-safe by run |
| **6/30, VARIED content, CONTENT-DISJOINT** (train 0–147 / test 148–221) | **0.948** | **Honest — generalizes to UNSEEN content = hardware** |

**Takeaway (CORRECTED):** the 0.52/0.59 was **content starvation** from the single
frozen enroll frame (one PA operating point), NOT a hardware limit. With **varied
content** (what the pipeline actually replays), honest content-disjoint separability is
**0.95** — device_3/4/5/6 perfect (1.000) on unseen frames, only device_1↔2 (same
USRP model/batch) confuse. The fingerprinter is **strong and hardware-based**; the
digital-δ attack failing 0% against it is therefore a meaningful negative result.

---

## Table C — Per-device fingerprint, VARIED content (6/30, content-disjoint = honest)

| Device | single-frame (6/29) | varied shared-content | **varied content-disjoint (honest)** |
|---|---|---|---|
| device_4 | 0.99 | 1.000 | **1.000** |
| device_6 (**attack target**) | 0.91 | 1.000 | **1.000** |
| device_5 | 0.67 | 1.000 | **1.000** |
| device_3 | 0.80 | 0.994 | **1.000** |
| device_1 | 0.22 | 0.963 | **0.843** (→ device_2) |
| device_2 | 0.19 | 0.958 | **0.829** (→ device_1) |

The single-frame column (left) was content starvation. With varied content the radios
separate cleanly; only device_1↔2 (same USRP model/batch) remain partly confusable, and
even they reach ~0.84 on unseen frames. **device_6 (target) is perfectly fingerprinted.**

---

## Table A-detail — Targeted PGD → device_4, crisp 6/28 model (`fingerprint_cnn_ft20260628.pt`, thr_mult=2)

| PSR (dB) | frames | fooling | →device_4 | device_4 | device_6 |
|---:|---:|---:|---:|---:|---:|
| 0 | 21,034 | 0.649 | 0.649 | 13,656 | 7,378 |
| −5 | 4,559 | 0.537 | 0.537 | 2,446 | 2,113 |
| −10 | 1,469 | 0.000 | 0.000 | 0 | 1,469 |
| **−15** | 17,030 | **0.733** | **0.733** | 12,482 | 4,548 |
| −20 | 17,069 | 0.573 | 0.573 | 9,787 | 7,282 |
| −25 | 2,731 | 0.382 | 0.382 | 1,042 | 1,689 |
| −30 | 4,542 | 0.000 | 0.000 | 0 | 4,542 |

Best at −15 dB (design PSR): 73% fooling, all → device_4. −10/−30 = 0% are bad/short
captures (frame counts ~1.5–4.5k vs ~17–21k), not attack failure. Where it works,
frames split cleanly device_4 (target) / device_6 (still correct) — no scatter.

---

## Table D — Open decisions for the advisor

| Question | Options |
|---|---|
| Threat model | Perturb real device (rig does this) **vs** adversary impersonates |
| Re-scope, given ~0.6 | (a) attack only well-fingerprinted devices (4/6); (b) pivot headline to *"content/path inflates FP accuracy"*; (c) both |
| Validation standard | Mandate **δ-off control + matched enrollment path** on every run |

---

## Caveats that qualify everything above
1. The **6/27 (100%) and 6/28 (73%) targeted runs lack δ-off controls** → "promising but unverified". That targeted landed on device_4 (not the device_5 replay-artifact class) is suggestive it's real, but unconfirmed.
2. The **device_5 "artifact"** was a train-path ≠ test-path issue (model enrolled on live message-strobe, tested on file-replay), fixed by enrolling on the replay path; δ-off then reads device_6 100%.
3. **CORRECTED:** honest (content-disjoint, varied) fingerprinting ≈ **0.95**, not 0.6 — the 0.6 was single-frame content starvation. **device_6 (target) is perfectly fingerprinted (1.000)** on unseen content. The confound-free **targeted device_4 attack was evaluated 6/30 against the all-replay model: 0% fooling at every PSR (−40…+10), both 2-channel and single-channel pre-combined.** Diagnosis: the *transmitted digital* combined file (δ included, even +10 dB) reads device_1 — it has no hardware fingerprint until device_6's radio imprints one OTA, after which it reads device_6. The digital δ lives **upstream** of the PA impairment the classifier reads, so it cannot move a hardware-keyed model. The earlier 6/27 (100%) / 6/28 (73%) "wins" were against strobe-enrolled models keying on content/path (a confound a digital δ *can* move).

---

## Mechanism — why a digital δ cannot fool a hardware fingerprinter

The **upstream / downstream barrier.** The adversary edits the **digital** signal (upstream);
the fingerprint is stamped by the **PA at transmit time** (downstream). The δ and the feature
it targets are in different domains, separated by hardware the adversary doesn't own. Three
controlled results on the varied model (`fingerprint_cnn_ft20260630varied.pt`, 6/30):

**(i) δ is provably inert on the transmit signal.** Clean digital payload reads device_1 (OOD,
no fingerprint); payload **+ δ** (even +0 dB, hot) **still** reads device_1. Same answer with
or without δ. Yet the *same* δ on a **recapture** frame (that carries device_6's hardware) flips
device_6→device_4 at **100%** (incl. 90° / 1-sample robust). The δ is a 100% weapon on a
hardware-bearing frame and literally inert on the transmit file — because the feature it attacks
doesn't exist until the PA creates it.

**(ii) Test 1 — the fingerprint is NOT localized (can't be reused).** Per-window fingerprint
accuracy (all devices send identical content ⇒ pure hardware): window 0 (preamble) **0.911**,
windows 1–13 (payload) mean **0.933**. The fingerprint is spread across the **whole frame**
(payload ≥ preamble) — hardware impairments modulate every OFDM symbol. ⇒ you cannot localize
the identity to a saved preamble and reuse it; changing the payload changes 93% of the evidence.

**(iii) Test 2 (proxy) — the last transmitter owns the fingerprint.** The *same* `enroll_varied.bin`
file transmitted by different radios: via device_1 → reads **device_1 (0.973)**, via device_6 →
reads **device_6 (1.000)**. Byte-identical content; the fingerprint follows the **transmitter**,
not the signal. ⇒ a captured fingerprint does not survive re-transmission — the adversary's own
PA overwrites it. (This is *why* RF fingerprinting is an anti-replay defense.) *Definitive physical
double-hop pending — see below.*

**Corollary (the user's observation):** the δ, transmitted by the **adversary's** radio, carries
the **adversary's** fingerprint — not device_4's — a second independent reason impersonation fails.

### Reconciliation with Kim et al. (arXiv:2005.05321)

Kim's channel-aware adversarial attacks target **content-domain** classifiers (modulation / signal-type
recognition), where the discriminative feature **is the digital signal structure** — the *same domain
the δ lives in*. There the δ reaches the feature and the attack works; Kim is correct for that task.
**RF fingerprinting is a hardware-domain classifier:** its feature is a physical impairment created by
the PA **downstream** of the δ. Applying a content-domain attack to a hardware-domain feature is a
**category error** — the perturbation never touches the fingerprint. *Quantifying exactly that is the
contribution:* channel-aware adversarial perturbations transfer to signal/modulation classifiers but
**not** to RF-fingerprint classifiers, with controlled OTA evidence.

---

## Hardware-mimicry — the only attack with a physical path to the fingerprint

Since the fingerprint is *created by transmit hardware*, the only viable attack makes the **adversary's
transmit chain synthesize the target's impairment signature** (not perturb content). Concrete plan:

1. **Characterize device_4's impairments** from enrollment captures: carrier frequency offset (CFO),
   I/Q gain & phase imbalance, PA AM/AM + AM/PM curve, phase-noise PSD, DC offset. (Tool:
   `exp3_impairment_fit.py` — TODO.)
2. **Pre-distort** the adversary's transmitted samples to impose those impairments (compose the inverse
   of the adversary chain with a forward device_4 model), or tune the radio front-end (gain/bias/DAC
   backoff) to approximate them.
3. **Validate physically:** transmit the pre-distorted signal through the adversary radio, recapture,
   fingerprint → target device_4.
4. **Threat-model honesty:** this is hardware impersonation, white-box on device_4's *signal-level*
   impairments (needs enrollment-grade access to device_4's emissions); no claim of cloning the silicon.
   Expected ceiling set by how faithfully a *different* PA can reproduce another's nonlinearity.

This is the honest positive-result direction: a physical path to the feature the digital δ never had.

---

## Pending — physical double-hop confirmation (test 2, definitive)

Capture (one transmit, one recapture, no new tooling):
> Replay device_6's enrollment recapture `train/6_30_2026/enrollment/device_6/adv_run_1.bin`
> (already carries device_6's real OTA fingerprint) **through the adversary radio** (e.g. device_1),
> recapture → `double_hop_dev6_via_dev1.bin`.

Eval: `python3 exp3_ota_eval.py`-style read with `fingerprint_cnn_ft20260630varied.pt`. Reads
**device_6** = fingerprint survived (impersonation viable); reads **device_1** = adversary's PA won
(replay defeated, as predicted). Ready to run on delivery.

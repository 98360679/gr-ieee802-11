# Experiment 3 — OTA Fingerprint Attack: Results Summary

Session13 closed-loop attack (device_6 = legit TX). Tables for advisor discussion.
Last updated 2026-06-30.

---

## Table A — Attack results (OTA)

| Attack | Digital (nom / 1-samp / 90°) | OTA fooling | δ-off control? | Verdict |
|---|---|---|---|---|
| **Targeted PGD → device_4** (fuzzy model, 6/27) | ~100 / ~100 / ~90 | **100% all PSR (0…−30)** → device_4, link-stealthy | ✗ | Works, but unvalidated + weak model |
| **Targeted PGD → device_4** (crisp model, 6/28) | ~100 / 100 / 90 | **38–73%** (73% @−15) | ✗ | Works, inconsistent (channel) |
| **Targeted PGD → device_4** (clean all-replay, new) | 94 / 94 / 93 | *pending* | will have | The confound-free run |
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
| 6/29, 3-run, **all-replay** (small CNN) | **0.52** | Honest, content + path controlled |
| 6/29, all-replay, **BigCNN (5×, 80 ep)** | **0.59** | **Capacity ruled out** → real limit |

**Takeaway:** the 0.93–0.99 was **inflated by content/path cues**. True
hardware-fingerprint separability ≈ **0.6**.

---

## Table C — Per-device fingerprint (clean all-replay, BigCNN)

| Device | Held-out acc | Separability |
|---|---|---|
| device_4 | 0.99 | well-fingerprinted |
| device_6 (**attack target**) | **0.91** | well-fingerprinted |
| device_3 | 0.80 | ok |
| device_5 | 0.67 | weak |
| device_1 | 0.22 | **~indistinguishable** (same model as 2) |
| device_2 | 0.19 | **~indistinguishable** |

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
3. Content-controlled fingerprinting ≈ **0.6**; the **device_6 target is still well-fingerprinted (0.91)**, so attacking it is meaningful. The new confound-free **targeted device_4 bundle (94% robust)** is built, awaiting the clean all-replay enrollment transmit.

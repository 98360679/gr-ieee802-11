# exp3 — MIMO loopback δ-delivery test (2026-06-27)

**Goal.** Decide *why* the OTA attack got ~0 fooling: is the perturbation δ failing
to arrive **time/phase-aligned** with the legit frame (a TX/MIMO problem), or is it
the **air channel** (multipath/drift)? Loopback replaces the air with a cable, so a
pass/fail isolates the two.

**Threat model (this rig).** Legit TX radiates the device_6 frames; the **adversary
radiates δ on a separate channel** (ch1). The adversary does *not* control the legit
frame — so this is genuinely 2-channel, not a single pre-combined signal.

**Why alignment is the crux.** `exp3_loopback_sim.py` (run 2026-06-27) showed the
targeted δ (device_6→device_4 @ PSR −20) is brutally alignment-sensitive:
- **timing:** 100% → device_4 at 0 skew, **0% at a 1-sample (200 ns @ 5 MHz) skew**
- **phase:** works to ±45°, **dead by 90°**; random phase/frame → 52%
- **noise:** survives 10 dB SNR (so weak signal is NOT the issue)

A MIMO USRP can time-align the two DACs (shared clock + timed start), but the
**ch0/ch1 phase offset is random per boot and uncalibrated** — and >45° kills it.
So this test must sweep ch1 phase to find/verify that offset.

---

## Files (all in `…/session13/ota_dev6/ft20260627_t4/`)

| file | role |
|---|---|
| `frame.bin` | **ch0** — legit device_6 frame, looped (50 ms period) |
| `perturbation.bin` | **ch1** — targeted δ (device_6→device_4, PSR −20), ε=1.0 |
| `phase_sweep/perturbation_phase_NNN.bin` | ch1 δ rotated by NNN° (calibration sweep) |
| `combined_singlechan.bin` | **upper-bound control** — frame+δ on ONE channel (perfectly aligned by construction) |
| `frame_only_singlechan.bin` | δ-off control at the same scale |

Model: `fingerprint_cnn_ft20260627.pt`. Legit class = **device_6**; expected spoof = **device_4**.

---

## Rig

- ch0 + ch1 → **RF power combiner** → coax + **30–40 dB attenuator** → RX. **No antennas, no air.**
- RX = `wifi_rx` capturing raw complex64 to a `.bin`. **AGC OFF / fixed RX gain**
  (the AGC pumps the gaps up and wrecks the burst extractor — see prior OTA notes).
- 5 MHz sample rate, center freq matched, USRP TX gain fixed across all captures.
- MIMO: shared time/clock so the two DACs start sample-aligned (timed command / PPS).

## Captures

1. **A — control, δ OFF** (ε=0 or ch1 disconnected) → expect **device_6**.
   Confirms the legit frame decodes/classifies through the cable+RF at all.
2. **B — δ ON** (ch1 = `perturbation.bin`, ε=1.0) → expect **device_4** *iff* δ lands aligned.
3. **C — phase sweep** (only if B fails): repeat B with ch1 =
   `phase_sweep/perturbation_phase_{000,045,090,135,180,225,270,315}.bin`, one capture
   each. If *some* phase flips to device_4, you've found the ch0/ch1 phase offset
   (timing is fine, just calibrate phase). If *none* do, timing/other is off.
4. **D — upper-bound control** (single channel): transmit `combined_singlechan.bin`
   on one TX port → expect **device_4**. This is δ perfectly aligned by construction;
   if even D fails, δ doesn't survive the DAC/ADC/RF front-end (more fundamental).

## Eval (per capture)

```
python3 exp3_attack_eval.py --attacked <capture>.bin --device 6 \
        --model fingerprint_cnn_ft20260627.pt --floor-pct 0
```
(`--floor-pct 0` = median floor: cable captures have clean silent gaps. Use `20`
only if you couldn't disable AGC.) Read `fooling` and `dominant_target`:
**dominant_target = device_4** with high fooling = δ delivered.

## Decision matrix

| B (δ on) | C (phase sweep) | D (single-ch) | conclusion → next step |
|---|---|---|---|
| → device_4 | — | — | δ delivers aligned over cable. Failure is the **AIR** → harden δ to the channel / improve OTA timing. |
| device_6 | some phase → device_4 | — | timing OK, **phase uncalibrated** → bake that phase into ch1 (or calibrate the MIMO) and re-OTA. |
| device_6 | none | → device_4 | δ survives RF but **2-ch alignment unreachable** → make δ **robust** (EOT over sub-sample shift + phase) or rethink delivery. |
| device_6 | none | device_6 | δ **doesn't survive the RF front-end** → too fragile; robust/EOT δ is mandatory. |

## Likely real fix (flagged, not yet built)

Given the <1-sample / <45° tolerance, even a successful loopback won't survive the
air robustly. The durable fix is an **alignment-robust δ**: craft with Expectation-
Over-Transformation — average the PGD gradient over random sub-sample time shifts
and random phase rotations each step — so δ tolerates ≥1 sample skew and arbitrary
phase. That trades a few dB of PSR for OTA viability. Build this if B/C/D show
alignment (not RF survival) is the wall.

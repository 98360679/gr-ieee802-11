# Exp 2 & 3 — CVNN OTA Transmit Bundles (manifest)

Transmit bundles for the OTA experiments against the CVNN fingerprinter. **The `.bin` files
live on T9 (data-on-drive convention); this manifest records their provenance + transmit
instructions.** Generated 2026-07-03.

## Generation
- **Model attacked:** `fingerprint_cvnn_7_03_attack.pt` (CVNN, 7/03 60 s fine-tune; device_4 &
  device_6 = 1.00).
- **δ:** per-frame **targeted device_6 → device_4**, **EOT-robust** (sub-sample shift ±1.5 /
  phase ±90°), design **PSR −10 dB**, 100 PGD steps × 8 EOT. `exp3_cvnn_pertdir.py` →
  `ota_dev6/eot_cvnn_pertdir/{id}.bin` (222 δ, **100% nominal targeted digital hit**).
- **Assembly:** `exp3_assemble_pert.py` → `exp3_psr_sweep.py` (2-ch) → `exp3_precombine_sweep.py`
  (1-ch). ch0 frame reused from `eot_t4/adv_frame.bin` (222 decoded device_6 frames).
- **PSR sweep:** −30 −25 −20 −15 −10 −5 0 +5 +10 +15 (matches Exp 1).

## Exp 2 — legit TX only (δ in the TX file)  `ota_dev6/eot_cvnn/combined_singlechan/` (4.9 GB)
Single-channel; **device_6 replays each combined file** (frame+δ baked in), perturbation
channel OFF, **fixed gain across the sweep**.
- `adv_combined_psr_{m30,m25,m20,m15,m10,m5,0,p5,p10,p15}.bin` — 10 PSR points.
- `adv_combined_psr_off.bin` — **δ-off control** (frame only, same scaling).
- **Expected:** ~0% fooling (δ upstream of device_6's PA → inert on the clean payload; the
  recapture wears device_6's own fingerprint). Documents the upstream/downstream barrier.

## Exp 3 — adversary δ ⊕ legit frame (MIMO superposition)  `ota_dev6/eot_cvnn/dac_safe_gapped/` (4.9 GB)
Two radios, summed over the air.
- `adv_frame.bin` — ch0, **device_6** transmits (fixed).
- `adv_perturbation_psr_{m30…p15}.bin` — ch1, **adversary** transmits (swap per PSR), ε=1.0.
- **δ-off control** = ch1 off (transmit `adv_frame.bin` alone).
- **Expected:** reading crosses device_6 → **adversary** as PSR rises (the δ wears the
  *adversary's* fingerprint), **never device_4**. Documents the superposition model.

## Evaluation (when recaptured)
- **Channel match:** capture a short clean device_6 run in the same session; fine-tune the CVNN
  to it before scoring (else the stale model won't read the recapture).
- **Metrics:** fooling (`exp3_cvnn_attack`-style read: →device_4 / off-device_6) + BER/FER
  (`exp3_cvnn_ber.py`), per PSR, with the δ-off control.
- Recapture to e.g. `attacked/7_0X_2026/{exp2_1ch,exp3_2ch}/device_6/adv_psr_*.bin`.

Generator/eval code is in git (`exp3_cvnn_pertdir.py`, `exp3_cvnn_attack.py`, `exp3_cvnn_ber.py`,
`exp3_assemble_pert.py`, `exp3_psr_sweep.py`, `exp3_precombine_sweep.py`) — fully reproducible.

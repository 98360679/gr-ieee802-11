# Closed-loop attack bundles — 2026-07-07

Single-channel + 2-channel, targeted + untargeted. Built on the fixed-`wifi_tx.grc`
replay enrollment (7_06_2026), which restored uniform strong captures.

## Provenance
- **Model:** `fingerprint_s14_ft0706_replay.pt` (device_6 = 1.00 cross-run, device_6-vs-device_4 = 0.997).
- **Reference `u`:** T91 `.../enrollment/7_06_2026/device_6/frame_run_2.bin` + `frame_index_run_2.csv` (762 frames; first **200 ids** used).
- **δ crafted on:** T9 `.../enrollment/7_06_2026/device_6/clean_run1.bin` (RX device_6 frames).
- **δ:** plain PGD (`--no-eot`), base PSR −12, 80 steps. Digital nominal: targeted 99% → device_4, untargeted 98% off device_6.
- **Payload boundary B:** `data_offset + 1360 .. data_offset + data_len` (header/FRID-id + tail stay clean).

## The 4 bundles  (dir: `session14/attack/closed_loop_0707/`)
| # | bundle | path | transmit |
|---|---|---|---|
| 1 | SINGLE targeted | `targeted/single/adv_combined_psr_*.bin` | device_6 replays each file on ch0, pert channel OFF. `off` = δ-off control |
| 2 | 2-CH targeted | `targeted/dac_safe/adv_frame.bin` + `adv_perturbation_psr_*.bin` | ch0 = adv_frame (device_6), ch1 = pert (adversary), ε=1, fixed gain |
| 3 | SINGLE untargeted | `untgt/single/adv_combined_psr_*.bin` | as #1 |
| 4 | 2-CH untargeted | `untgt/dac_safe/adv_frame.bin` + `adv_perturbation_psr_*.bin` | as #2 |

## PSR sweep
`p5, 0, m5, m10, m15, m20, m25, m30` (single-channel also has `off`). All DAC-safe (peak 0.950).

## Scoring after recapture (gate on physics FIRST)
1. **BER gate:** payload BER must rise with PSR (δ actually landed), like 7_04 targeted hit ~0.41. If BER ≈ 0 at all PSR, δ missed — result is void.
2. Then fingerprint sweep vs `fingerprint_s14_ft0706_replay.pt`: targeted → device_4 %, untargeted → off-device_6 %. δ-off baseline must read device_6.

## Channel-aware (adv-eot) 2-channel variants  (added 2026-07-07)
`--adv-eot` δ (transform δ ALONE → models adversary channel `h_a`, EOT shift±1.5/phase±90 ×8),
150 ids, base −12. Digital nominal: **targeted 97% → device_4, untargeted 47% off device_6**
(untargeted is weaker than the no-EOT 98% — the channel-robust constraint costs nominal strength).
- `targeted/dac_safe_eot/adv_frame.bin` + `adv_perturbation_psr_*.bin`
- `untgt/dac_safe_eot/adv_frame.bin` + `adv_perturbation_psr_*.bin`
Raw δ archived: `delta_{tgt,untgt}_adveot/` (150 each). All DAC-safe (peak 0.950), PSR −30…+5.

## Full inventory (6 bundles)
| targetedness | single (no-eot) | 2-ch (no-eot) | 2-ch (adv-eot) |
|---|---|---|---|
| targeted   | `targeted/single/` | `targeted/dac_safe/` | `targeted/dac_safe_eot/` |
| untargeted | `untgt/single/`    | `untgt/dac_safe/`    | `untgt/dac_safe_eot/` |

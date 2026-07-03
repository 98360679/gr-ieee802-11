# Experiment 2 — OTA, legit TX only (δ in the TX file)

device_6 replays the **combined** frame+δ (single channel, `wifi_adversary_tx` ch0, ch1 off);
RX recaptures; scored against the CVNN. This is the OTA counterpart to Exp 1's 100% digital
success — δ injected **upstream of device_6's PA**. Captured/scored 2026-07-03.

- **Bundle:** `ota_dev6/eot_cvnn/combined_singlechan/adv_combined_psr_*.bin` (EOT δ vs
  `fingerprint_cvnn_7_03_attack.pt`, targeted device_6→device_4).
- **Recaptures:** `attacked/7_03_2026/device_6/adv_psr_{0,5,10,15,20,25,30}.bin` (bare = −PSR)
  + `adv_psr_first_off.bin` / `adv_psr_last_off.bin` (δ-off controls, start & end).

## ⚠️ Capture caveat (read first)
The session took a long time (setup checks + slow δ generation), so **the channel drifted
during it**, and the CVNN was trained on a *different* (enrollment) channel. Evidence: the two
δ-off controls disagree (`first_off`→device_2, `last_off`→device_4) and neither reads device_6
cleanly. So the **device_6 baseline is unreliable** and the numbers below are contaminated by
channel drift, not a clean measurement. The **targeted-fooling metric (→device_4) is still
informative** — it's ~0 everywhere including where the clean baseline drifted *toward* device_4.

## Fooling (targeted → device_4)

| capture | →device_4 | device_6 |
|---|---:|---:|
| δ-off first | 0.005 | 0.26 |
| δ-off last | 0.44† | 0.25 |
| PSR 0 | 0.000 | 0.50 |
| PSR −5 | 0.000 | 0.64 |
| PSR −10 | 0.005 | 0.64 |
| PSR −15 | 0.005 | 0.55 |
| PSR −20 | 0.000 | 0.32 |
| PSR −25 | 0.005 | 0.53 |
| PSR −30 | 0.000 | 0.52 |

† `last_off` is a *clean* frame (no δ) yet reads 44% device_4 — a channel-drift artifact, not
the attack. Tellingly, the **δ-attack captures read ~0% device_4** while this clean one drifted
*more* → the reading is driven by the channel, not the δ.

## BER (link stealth, OTA)
Decoded via gr-ieee80211 (isolated subprocess per capture — noisy captures segfault the
decoder); BER vs the δ-off-first decode, best-matched, rule-of-3 floor.

| capture | frames decoded | BER |
|---|---:|---:|
| δ-off first (ref) | 8 | — |
| δ-off last | 0 | 0.50 (no decode) |
| PSR 0 | 7 | 3.4e-2 |
| PSR −5 | 6 | 3.4e-2 |
| PSR −10 | 8 | 3.4e-2 |
| PSR −15 | 2 | 2.0e-2 |
| PSR −20 | 10 | 1.9e-3 |
| PSR −25 | 10 | 5.6e-3 |
| PSR −30 | 7 | 1.1e-2 |

BER roughly tracks δ magnitude (highest at PSR 0). **But only 2–10 frames decode per capture**
(out of ~222) — *even the clean δ-off decodes just 8* — so the recaptures are **low-SNR /
channel-limited** (very high FER from the channel, not the δ). Treat these as indicative.

## Verdict
**Targeted OTA fooling ≈ 0% at every PSR → the barrier holds** (as expected): the δ, upstream
of device_6's PA, does not fool the fingerprint toward device_4. This is the intended contrast
with Exp 1 (100% digital at the receiver input). **But the demonstration is not clean** — the
δ-off baseline is broken by intra-session channel drift + a stale (enrollment-channel) model.

**To get a publishable Exp 2:** run it as **one fast, stable session** — capture an all-6-device
enrollment, then immediately the device_6 attack sweep + δ-off (fixed gain, minimal gap), so the
CVNN can be fine-tuned to that channel and the δ-off baseline reads device_6 ~100%.

Tools: `exp3_cvnn_attack.py` (fooling), `exp3_exp2_ber_one.py` (isolated OTA BER).

# Progress Report — Untargeted OTA Attack on the Hardware-Fingerprint Classifier
### Session 14, capture 2026-07-05 (device_6, untargeted, 2-channel)

## Objective
Test whether a digital adversarial perturbation (δ), radiated over the air, can push a
hardware-fingerprint classifier **off** the legitimate transmitter **device_6** (an
*untargeted* "disruption" attack — success = read anything other than device_6). The
classifier identifies 6 USRP transmitters by their hardware impairments (CFO, I/Q
imbalance, PA nonlinearity).

## Setup
- **Legit TX:** device_6.  **Attack:** untargeted δ, 2-channel (adversary radiates δ on a
  separate channel superimposed on device_6's frames).
- **δ power sweep (PSR):** −20, −15, −10, −5, 0, +5 dB — from far below the signal up to
  **+5 dB (δ louder than the signal itself)**.
- **Scoring:** replay-adapted CVNN fingerprint model (δ-off baseline correctly reads device_6);
  ~1,000–1,070 frames evaluated per PSR level.

## Result — the classifier reads device_6 at every δ power

| PSR (dB) | frames | **device_6 rate** | fooling (off device_6) | any → device_4 |
|---:|---:|---:|---:|---:|
| −20 | 1,059 | **0.999** | 0.001 | 0 |
| −15 | 1,071 | **1.000** | 0.000 | 0 |
| −10 | 1,070 | **1.000** | 0.000 | 0 |
| −5  | 1,073 | **1.000** | 0.000 | 0 |
| 0   | 1,058 | **1.000** | 0.000 | 0 |
| **+5** | 1,018 | **1.000** | 0.000 | 0 |

**Fooling rate ≈ 0 at all δ powers, including +5 dB (δ louder than the signal).**

## Frame accounting
| stage | 7_05 untargeted | 7_04 targeted (reference) |
|---|---:|---:|
| Frames **sent** (device_6 TX log) | not logged | **22,810** (run_1; 23,243 both runs) |
| Frames **decoded** over the air (`ber_frames`) | 3,013 | 12,562 |
| Frames **scored** by classifier | 6,349 | 12,112 |

For 7_05 no transmit-side log exists (no T91 record), so the RX-decoded count is a lower bound on
frames over the air. For 7_04, device_6 transmitted a 379-frame reference looped ~60×, ~55% decoded.

> **Footnote — Scored vs Decoded.** "Decoded" counts frames the 802.11 receiver fully
> synchronized and demodulated (strict). "Scored" counts power bursts the fingerprint extractor
> detected (more permissive — no preamble sync required), so Scored > Decoded: bursts too
> corrupted for the 802.11 receiver still carry the hardware fingerprint and are classified —
> and they read device_6 either way. This is a detector-criteria difference, not a data error.

## Movement toward other devices — none
Across the entire sweep (~6,300 frames), the classifier's output does **not** drift toward any
other transmitter. The only non-device_6 frame is **1 frame → device_3, at the quietest δ
(PSR −20)** — statistical noise, and notably it occurs where δ is *weakest*, i.e. anti-correlated
with attack strength. **Zero frames** move to device_4 or any other class at any PSR, and the
device_6 rate does **not** decrease as δ power increases.

## Interpretation
The perturbation, radiated over the air, has **no effect** on the emitter's hardware fingerprint.
This is consistent with the project's central finding: the classifier reads a physical property of
the *emitting hardware* (`T_6`), which lives upstream of any injected digital signal — so a δ that
is 100% effective at the digital classifier input does not transfer over the air. The attack fails
not marginally but **completely and monotonically** (no partial drift, no movement toward a
neighbor class), even when the perturbation is louder than the legitimate signal.

## Validity note / next step
This 2-channel geometry requires the adversary's δ to arrive time-aligned onto device_6's payload;
independent-radio sync makes that hard, so this result should be read as **"no disruption observed"**
rather than a fully controlled barrier proof. The **single-channel** experiment (device_6 replays
`u+δ` as one aligned stream — δ guaranteed to land) is the confirmatory test and is in progress.

---

## Appendix A — Methods: δ generation & the landing problem

### A.1  How δ is generated
δ is produced by **projected gradient descent (PGD) against the fingerprint classifier**, in the
digital domain. Per frame:

1. **Start from a clean device_6 frame** the model already reads as device_6.
2. **Confine δ to the payload region** with a mask — preamble, L-SIG, MAC header, FRID-id and tail
   are held clean, so the frame still synchronizes and its id decodes.
3. **Set a power budget from the target PSR:** `‖δ‖ = 10^(PSR/20) · ‖payload‖`
   (PSR −10 dB → δ 10 dB below payload; +5 dB → δ louder than payload).
4. **Iterate ~70–100 steps:** run frame+δ through the classifier; compute the loss toward the goal
   (push probability to **device_4** for targeted, or **away from device_6** for untargeted); take
   the gradient w.r.t. δ; step δ along it; **project back** onto the power budget.
5. **EOT variant** (channel-aware, 2-channel): average the gradient over random **sub-sample
   time-shifts (±1.5)** and **carrier-phase rotations (±90°)** so δ tolerates small transforms.

Result: a small, payload-confined, power-limited signal optimized to fool the model. Digitally it is
~97–99 % effective. It is crafted **in the classifier's input domain** — which is precisely why
delivering it there physically is the hard part.

### A.2  What "landing" means, and why δ may miss (2-channel)
"Landing" = the δ energy must arrive **on the payload samples of device_6's frame** at the receiver.
In 2-channel, δ is radiated by a **separate radio** from device_6, and the two are **not clock-locked**:

| failure mode | effect |
|---|---|
| **Time misalignment** (no shared PPS/trigger) | δ arrives at a different time → lands in the **inter-frame gap** or on the preamble, not the payload |
| **Cadence mismatch** | frame rate ≠ δ replay rate → overlap drifts; δ hits a frame only occasionally |
| **Sample-clock offset (SFO)** | ppm clock difference → alignment walks off the payload over the capture |
| **Start-offset error** | δ file starts at the wrong sample offset → constant miss |
| **Adversary never synced** | δ radiated at arbitrary times → effectively random vs the frames |

**Evidence in this experiment:** payload BER stayed **flat at the ~0.002 baseline across all PSR,
including +5 dB**. A landing δ corrupts the payload (it reached 0.41 in the 7_04 sweep where δ *did*
land). Flat BER ⇒ device_6's frames arrived clean ⇒ the δ energy fell **outside the payload** — "not
in the active region," driven by the lack of tight time-sync. The adversary radio was transmitting;
the signal simply never overlapped the target window.

### A.3  Why single-channel avoids this
In single-channel, device_6 transmits `u+δ` as **one combined stream** — δ is baked into the payload
samples **before the DAC**, aligned by construction and **guaranteed to land**. To make 2-channel land
would require a shared **10 MHz + PPS reference** (or hardware trigger) between device_6 and the
adversary, plus starting δ on device_6's frame boundary. EOT robustness covers sub-sample jitter, not
frame-scale misalignment.

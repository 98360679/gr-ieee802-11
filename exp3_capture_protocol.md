# Experiment 3 — Fingerprint Capture Protocol (fresh collection)

Why we're recollecting: the session13 device→label identities were lost and the
receiver (`serial=3256204`) failed. This protocol makes identity unambiguous (a
written manifest) and keeps the capture chain consistent so the model is valid.

**Source of truth for identity:** `manifest.json` (schema + tooling in
`exp3_manifest.py`). Fill it in *before* capturing; nothing downstream guesses
labels from folder names anymore.

---

## 0. Prerequisites
- Replacement **RX USRP** installed and visible (`uhd_find_devices`).
- `gr-foo` + `gr-ieee802-11` built/installed (see README).
- Victim TX radios + the adversary radio (device_6) on hand.
- The T9 drive mounted (currently `/media/nghoselab/T9`).

---

## 1. Identity & manifest (do this first, on the bench)
1. Physically label every radio (sticker) and read each one's id:
   `uhd_find_devices` → note each **TX serial** and the **new RX serial**.
2. Generate and fill the manifest:
   ```bash
   python3 exp3_manifest.py --template > /media/nghoselab/T9/Data/session14/manifest.json
   # edit: session id, rx_serial, every device's tx_id + role + runs
   ```
   - Victims = the fingerprint classes. **device_6 = adversary** (role `adversary`).
   - Class indices are assigned in ascending label order automatically
     (device_1→0, device_3→1, …) — same rule the trainer uses.
3. Validate:
   ```bash
   python3 exp3_manifest.py --validate /media/nghoselab/T9/Data/session14/manifest.json
   ```
   Must print **OK** (no placeholders, unique labels + tx_ids, ≥1 victim).

---

## 2. Fix the capture chain (and never change it mid-session)
The receiver is **common-mode** across all devices, so any RX change injects a
domain shift. Lock these for the *entire* session and record them in the manifest:
- **RX serial** (the one new receiver), **antenna**, **sample rate (5 MHz)**,
  **center freq (2.45 GHz)**, **RX gain**.
- In `wifi_rx.grc`: set `dev_addr` to the new RX serial, point the file-sink and
  `frame_logger` `log_path` at `/media/nghoselab/T9/...` (the real mount), and set
  `decode_mac log = False` (avoids the USRP-overflow regression).

**Gain tuning:** bring up the link and tune RX gain to avoid saturation before
capturing — `live_power.py` for a live power meter, `exp3_gain_sweep.sh` to pick
the gain with the best error-free frame rate. Use the SAME gain for all devices.

---

## 3. Clean capture, per victim device
For each victim in the manifest, for each run `R` in its `runs` list:
1. Start that victim transmitting with FRID frame-id stamping:
   `python3 exp3_wifi_tx.py` (stamps `b'FRID'+uint32_be(frame_id)+'x'*filler`).
2. Capture ~30 s of raw I/Q (≈ a few hundred frames at the 300 ms cadence) to the
   manifest layout path:
   `/media/nghoselab/T9/Data/session14/train/device_<N>/clean_run_<R>.bin`
3. Keep **runs balanced**: same count per device. Split is **runs 1,2 = train,
   run 3 = held-out val** (matches the existing pipeline; no window leakage).

Consistency rules:
- One receiver, fixed gain/freq/rate/antenna throughout (see §2).
- Keep TX position/orientation steady within a device; the multiple runs supply
  natural over-time channel variation for the train/val split.
- ≥3 runs/device recommended (2 train + 1 val); more runs = more data.

Optional: capture a **clean device_6** baseline too (role still `adversary`; not a
class) for attack-side reference.

---

## 4. Process → train → sanity-check
```bash
# Stage 1: extract full frames (preamble+payload) -> frames_dev<N>.npz
python3 exp3_extract_frames.py --root /media/nghoselab/T9/Data/session14/train --devices <victims>
# Stage 2/3: build + train (drop nothing yet; inspect first)
python3 exp3_train_fingerprint.py --epochs 50 --aug --wd 1e-4 --label-smooth 0.05
# inspect confusion; drop a near-duplicate device only if the matrix shows one
python3 exp3_confusion.py
# re-baseline is now trivially "same RX" -> expect high; it's the post-swap check
python3 exp3_rebaseline.py --root /media/nghoselab/T9/Data/session14/train
```
Note: `exp3_fp_model.py` currently hardcodes `NUM_CLASSES=5` / `DEVICE_NAMES`.
Update those to the new victim set (or thread them from the manifest) before
training the new model.

---

## 5. Attack stage (after a good model)
```bash
python3 exp3_make_perturbation.py --device <legit victim> --psr -20   # 4-device-aware
python3 exp3_adversary_tx.py ...   # device_6 radiates perturbation.bin
# recapture legit + adversary together, then:
python3 exp3_rebaseline.py / exp3_eval_capture.py   # measure OTA fooling
```

---

## Checklist (capture day)
- [ ] New RX serial known; `wifi_rx.grc` `dev_addr` updated
- [ ] `manifest.json` filled + `--validate` prints OK
- [ ] RX gain tuned (no saturation), fixed for all devices
- [ ] `decode_mac log=False`; sink/log paths on the mounted drive
- [ ] Each victim: ≥3 balanced runs captured to the layout path
- [ ] (opt) device_6 clean baseline captured
- [ ] `exp3_manifest.py --validate manifest.json --root <train dir>` → all files present

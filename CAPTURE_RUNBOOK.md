# Fading Fingerprints — Capture Runbook (one laptop)

How to run the **two flowgraphs on a single laptop** to collect a day's data, exactly
the way it was done on the bench. One driver, `ff_capture.py`, orchestrates both
flowgraphs; you run **one command per device**.

---

## 1. What `ff_capture.py` actually does

It runs **both** flowgraphs on the same machine:

- **TX** (`wifi_tx_updated.py`) — the device-under-test. Brought up **once** and kept
  **resident in-process** (the radio takes >60 s to settle, so we never re-init it
  between runs).
- **RX** (`wifi_rx.py`) — the shared receiver (**Ettus b200, serial `325936A`, antenna
  `TX/RX`**). Launched as short subprocess "probes": first a 5 s probe to read the
  received level, then one 30 s probe per run to record `clean_run_k.bin`.

Between probes it **nudges the TX gain live** until the received power hits a fixed
target, then records all runs in **one radio session** (one LO latch). Equalising the
received level across devices and days is the control that makes the cross-day
comparison honest — a Day-2 change is then *drift*, not a level difference.

Everything (patching the flowgraphs, gain search, recording, folder labelling) happens
inside the one `ff_capture.py` call. You do **not** launch the flowgraphs yourself.

---

## 2. Hardware — both radios on the ONE laptop

| Role | Device | Connection | Antenna |
|---|---|---|---|
| **RX (fixed, all sessions)** | b200 `325936A` | USB 3.0 | `TX/RX` |
| **TX (one at a time)** | device under test | see below | see table in §6 |

- **B200 / B205mini** connect by **USB** → `--tx-serial <serial>` (e.g. `325426C`).
- **USRP2 / N2922 (N210)** connect by **Gigabit Ethernet** → `--tx-serial <IP>` (e.g.
  `192.168.10.5`). Set the laptop's wired NIC to `192.168.10.1/24`; confirm with
  `uhd_find_devices`.
- Keep the RX b200 on its **own** USB controller. Do **not** capture to a USB drive that
  shares the b200's bus — the contention causes `BAD_PACKET`. Capture to the internal disk.

---

## 3. Prerequisites (install once on the laptop)

- **UHD 4.x** and **GNU Radio 3.10**
- **gr-ieee802-11** (maint-3.10) and **gr-foo** built & installed — these provide the
  WiFi PHY blocks the flowgraphs import. Confirm: `python3 -c "import ieee802_11"`.
- Python: **PyQt5**, **numpy**
- Verify both radios are visible: `uhd_find_devices`

---

## 4. Get the code

```bash
git clone git@github.com:98360679/gr-ieee802-11.git experiment_3   # or your clone URL
cd experiment_3
git checkout fading-fingerprints
git pull
```

`ff_capture.py`, `wifi_tx_updated.py`, and `wifi_rx.py` all live here together, so
`ff_capture.py` finds the flowgraphs next to itself with no extra flags. (It **edits the
two flowgraph files in place** each run to stamp freq/serial/antenna/paths — that dirty
state is expected; `git checkout -- wifi_tx_updated.py wifi_rx.py` to reset.)

---

## 5. General form of one capture

Bring up **one** device, position it, run its command, and **wait for it to finish**
before starting the next (the radio must be released first).

```bash
python3 ff_capture.py \
  --day <D> --exp <E> --device-id <id> [--name <label>] \
  --tx-serial <serial|IP> --tx-antenna <TX/RX|J2> \
  --freq <Hz> \
  [ --target-db -24 --tol 0.5  |  --fix-gain <g> ] \
  --run-ids 1,2,3 --capture-secs 30 --init-wait 40 \
  --out-root ~/captures/fading_fingerprints_2412
```

- `--target-db -24 --tol 0.5` → auto-search TX gain to land at **−24 dB** received
  (used for every device that can reach it).
- `--fix-gain <g>` → skip the search and transmit at exactly this gain (used only for a
  device already at its ceiling, e.g. the N2922 at `--fix-gain 1.0`).
- `--name` relabels the folder from `device_<id>` to a model name (Exp 2 only), merging
  into it without clobbering other runs.

Wrap in `timeout 320` if you want a hard stop (GNU Radio's teardown can occasionally hang
after the data is already written).

---

## 6. Day-1 commands — reuse IDENTICALLY on Day 2

**Consistency of frequency AND received level, per device, across both days, is the whole
experiment.** Same freq, same antenna, same distance, same target level. The gain *value*
may differ slightly day to day — that's fine; the −24 dB **level** is the control.

### Exp 1 — same-model (three B200minis), all @ 2412 MHz, `TX/RX`, 12"

```bash
# device_1  (B200 325426C)
python3 ff_capture.py --day 1 --exp 1 --device-id 1 --tx-serial 325426C \
  --tx-antenna TX/RX --freq 2.412e9 --target-db -24 --tol 0.5 \
  --run-ids 1,2,3 --capture-secs 30 --init-wait 40 --out-root ~/captures/fading_fingerprints_2412

# device_2  (B200 326364D)
python3 ff_capture.py --day 1 --exp 1 --device-id 2 --tx-serial 326364D \
  --tx-antenna TX/RX --freq 2.412e9 --target-db -24 --tol 0.5 \
  --run-ids 1,2,3 --capture-secs 30 --init-wait 40 --out-root ~/captures/fading_fingerprints_2412

# device_3  (B200 326361A)
python3 ff_capture.py --day 1 --exp 1 --device-id 3 --tx-serial 326361A \
  --tx-antenna TX/RX --freq 2.412e9 --target-db -24 --tol 0.5 \
  --run-ids 1,2,3 --capture-secs 30 --init-wait 40 --out-root ~/captures/fading_fingerprints_2412
```

### Exp 2 — different-model, each at its own consistent frequency

```bash
# B200   (326361A) @ 2412 MHz, TX/RX, 12"   → folder "B200"
python3 ff_capture.py --day 1 --exp 2 --device-id 3 --name B200 --tx-serial 326361A \
  --tx-antenna TX/RX --freq 2.412e9 --target-db -24 --tol 0.5 \
  --run-ids 1,2,3 --capture-secs 30 --init-wait 40 --out-root ~/captures/fading_fingerprints_2412

# USRP2  (192.168.10.3) @ 5.29 GHz, J2, 18"  → folder "USRP2"
#   (5.29 GHz because the XCVR2450's 2.4-band edge is too weak; its own per-device freq.)
python3 ff_capture.py --day 1 --exp 2 --device-id 4 --name USRP2 --tx-serial 192.168.10.3 \
  --tx-antenna J2 --freq 5.29e9 --target-db -24 --tol 0.5 \
  --run-ids 1,2,3 --capture-secs 30 --init-wait 40 --out-root ~/captures/fading_fingerprints_2412

# N2922  (192.168.10.5) @ 2412 MHz with VERT2450, TX/RX, 12"  → folder "N2922"
#   fix-gain 1.0 (device is at its ceiling; lands ~-24 dB with the VERT2450 at 12").
python3 ff_capture.py --day 1 --exp 2 --device-id 5 --name N2922 --tx-serial 192.168.10.5 \
  --tx-antenna TX/RX --freq 2.412e9 --fix-gain 1.0 \
  --run-ids 1,2,3 --capture-secs 30 --init-wait 40 --out-root ~/captures/fading_fingerprints_2412
```

For **Day 2**, change `--day 1` → `--day 2` in every command and keep everything else
identical. Antenna, frequency, and distance MUST match Day 1 per device.

Output tree:
```
~/captures/fading_fingerprints_2412/day{D}/exp{E}/<device>/clean_run_{1,2,3}.bin
```

---

## 7. Verify each capture

After a device finishes, check every run extracted a healthy number of frames:

```bash
cd experiment_3
R=~/captures/fading_fingerprints_2412
for k in 1 2 3; do
  f="$R/day1/exp2/N2922/clean_run_$k.bin"        # adjust exp/device
  python3 -c "from exp3_extract_frames import extract_frames_for_file; \
    print('run $k:', len(extract_frames_for_file('$f',floor_pct=20.0,thr_mult=2.0)[0]),'frames')"
done
```

Expect ~200–400 frames/run at −24 dB. A run at 0 frames or a level far from −24 dB
means reposition/retune and recapture that device.

---

## 8. Critical gotchas (these bit us)

1. **`--run-ids` is COMMA-separated** (`1,2,3`). Space-separated (`--run-ids 2 3`) makes
   argparse treat the extra as a stray positional and it silently errors out with no
   capture. A single `--run-ids 1` works, which masks the mistake.
2. **One session = one LO latch.** Always capture a device's 3 runs in **one** invocation
   (`--run-ids 1,2,3`). Never run three separate single-run calls — each re-inits the
   radio and re-latches the LO at a different phase, baking a fake per-run "fingerprint"
   into the data.
3. **Let each device's command fully finish** before starting the next. Back-to-back
   invocations race on the radio (the previous one hasn't released it) and the second
   produces nothing.
4. **Capture to the internal disk**, never a USB drive sharing the b200's bus → `BAD_PACKET`.
5. **b200 init > 60 s.** `--init-wait 40` is fine for the N-series/USRP2; bump to `70`
   if a b200 TX shows a weak/None first probe.
6. Qt runs offscreen automatically (`QT_QPA_PLATFORM=offscreen`). Headless is fine.

---

## 9. Then build the dataset / train

Once a day is captured (and again after Day 2), from `experiment_3`:

```bash
R=~/captures/fading_fingerprints_2412
# within-day sanity (train runs 1-2, test held-out run 3):
python3 ff_train.py --root $R --exp 2 --protocol within-day --day 1 --val-run 3 --epochs 40
# cross-day drift (train ALL Day-1, test ALL Day-2) — the headline:
python3 ff_train.py --root $R --exp 2 --protocol cross-day --train-day 1 --val-day 2 --no-aug --epochs 40
```

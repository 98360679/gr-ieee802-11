# Fading Fingerprints — Runbook

RF-fingerprint **separability + cross-day stability** study. Two experiments, the **same
receiver** throughout, run on **Day 1** and repeated on **Day 2**:

| Experiment | Devices | Question |
|---|---|---|
| **Exp 1** | same **model** (e.g. several B205minis) | Can the fingerprint separate nominally identical hardware? |
| **Exp 2** | different **models** (B-series **vs** USRP2/N-series) | How does separability change across architectures? |

Running both days lets us measure not just *separability* but *temporal stability* — whether a
model trained one day still holds the next. Working hypothesis (from the CFO work): **same-model
separates only weakly and fades day-to-day** (it's largely carrier-offset, which drifts); **cross-
architecture separates by real hardware and stays stable.**

Pipeline: **`ff_capture.py`** (collect) → **`ff_dataset.py`** (build splits) → **`ff_train.py`**
(train + drift/CFO analysis).

---

## Fixed parameters — the controls (do NOT change between days)

| Parameter | Value | Why |
|---|---|---|
| Frequency | **3.3 GHz** | only band all models reach (N-series caps at 4.4 GHz; 5.29 GHz is out) |
| Shared receiver | **b200 serial 325936A**, antenna **TX/RX** | same RX in both experiments removes the receiver as a variable |
| RX gain target | **−44.5 dB** received power | equalises SNR across devices & days → a Day-2 change is *drift*, not level |
| Capture | **3 runs × 30 s** per device | runs give the leakage-safe within-day split |
| Sample rate | 5 MHz | matches the frame/window geometry (WIN=1024, FRAME_LEN=15360) |

Keep positions, gains, and the −44.5 dB target **identical on both days** so Day 2 differences are
device drift, not geometry.

---

## One-time host setup

The USRP2/N-series stream over GigE and need large socket buffers, or you get `fifo-ctrl`
timeouts / `BAD_PACKET`:

```bash
sudo sysctl -w net.core.rmem_max=50000000 net.core.wmem_max=50000000
```

All capture scripts live in the flowgraph dir and import the WiFi TX flowgraph, so run them there:

```bash
cd /home/nghoselab/gr-ieee802-11-maint-3.10/examples
```

---

## Device table — fill in before Day 1

Assign a stable `device_id` to each radio and keep it fixed across both days. `--tx-antenna` is
`TX/RX` for B-series/b200 and `J1` (or `J2`) for the USRP2/N-series.

| device_id | model | serial / IP (`--tx-serial`) | antenna | Exp |
|---|---|---|---|---|
| 1 | B205mini | `325426C` *(example)* | TX/RX | 1 |
| 2 | B205mini | `______` | TX/RX | 1 |
| 3 | B205mini | `______` | TX/RX | 1 |
| … | | | | |
| — | USRP2 | `2192` (192.168.10.3) | J1 | 2 |
| — | N2922 | `192.168.10.5` | J1 | 2 |

> Exp 2 must cross a **real architecture line**: a B-series device (AD9364 RFIC) vs a
> USRP2/N-series (SBX/CBX daughterboard). B200mini vs B205mini share the AD9364 and behave like
> same-model, so that pairing is *not* a valid "different model."

---

## Stage 1 — Capture  (`ff_capture.py`)

Brings the device's TX up **resident**, tunes `tx_gain` LIVE until the received level locks at
−44.5 dB, then records the 30 s clean runs via the shared RX. One device at a time; you swap the
device between calls.

```bash
cd /home/nghoselab/gr-ieee802-11-maint-3.10/examples

# Exp 1 (same model), Day 1, one B205mini:
python3 ff_capture.py --day 1 --exp 1 --device-id 1 --tx-serial 325426C \
      --tx-antenna TX/RX --run-ids 1,2,3

# Exp 2 (different model), Day 1, the USRP2:
python3 ff_capture.py --day 1 --exp 2 --device-id 7 --tx-serial 2192 \
      --tx-antenna J1 --run-ids 1,2,3
```

Watch for the `RESULT ... clean_run=NN MB frames=NNN bad_packet=no` line per run. A good run is
~10–15 MB with a few hundred frames and `bad_packet=no`. Writes to
`~/captures/fading_fingerprints/day{D}/exp{E}/device_{id}/`.

Repeat for **every device** in the experiment. Then, on **Day 2**, rerun everything with `--day 2`
(same device_ids, same serials, same antenna).

Useful flags: `--target-db` (level, keep −44.5), `--fix-gain G` (skip the search if a device caps
out), `--capture-secs`, `--init-wait` (raise if a radio is slow to come up).

---

## Stage 2 — Build / inspect the dataset  (`ff_dataset.py`)  *(optional)*

`ff_train.py` builds the data itself, but you can inventory it first — frame counts per device,
label map, train/val sizes — without training:

```bash
cd /home/nghoselab/Experiments/experiment_3

# per-day separability view:
python3 ff_dataset.py --exp 1 --protocol within-day --day 1

# cross-day view:
python3 ff_dataset.py --exp 1 --protocol cross-day --train-day 1 --val-day 2
```

Devices are auto-discovered from the `device_*` folders (cross-day uses only devices present on
**both** days). Add `--devices 1,2,3` to pin an explicit set, `--rebuild` to ignore the cache.

---

## Stage 3 — Train + analyze  (`ff_train.py`)

Trains a CVNN sized to the experiment's device count. Prints per-device accuracy, a confusion
matrix, and the **headline Day→Day frame accuracy**; saves `fingerprint_<tag>.pt` + a `.json`
sidecar (devices, label map, protocol, score).

**Per-day separability** (is anything separable today?):
```bash
python3 ff_train.py --exp 1 --protocol within-day --day 1 --epochs 40
```

**Cross-day drift** — run all three to tell the full story for each experiment:
```bash
# 1) raw drift — no synthetic drift masking the real day-to-day change:
python3 ff_train.py --exp 1 --protocol cross-day --train-day 1 --val-day 2 --epochs 40 --no-aug

# 2) drift-augmented — does training-time augmentation recover the drop?
python3 ff_train.py --exp 1 --protocol cross-day --train-day 1 --val-day 2 --epochs 40

# 3) CFO-corrected — is the drift essentially just carrier offset?
python3 ff_train.py --exp 1 --protocol cross-day --train-day 1 --val-day 2 --epochs 40 --cfo-correct
```

Then repeat all of the above with `--exp 2`. The **contrast between Exp 1 and Exp 2** across those
three conditions is the result:

- **Exp 1 (same model):** expect a real cross-day drop in (1) that (3) largely fixes → the
  fingerprint was mostly CFO, and CFO drifts. That *is* the finding.
- **Exp 2 (different models):** expect high, stable accuracy in (1) with little to gain from (3) →
  genuine hardware fingerprint, robust across days.

---

## Directory layout

```
~/captures/fading_fingerprints/
  day1/
    exp1/device_1/clean_run_{1,2,3}.bin   ber_frames_run_*.jsonl   rx_frames_run_*.jsonl
    exp1/device_2/...
    exp2/device_7/...
    enroll_gains.csv                       # per-device locked tx_gain + level (audit trail)
  day2/
    exp1/...   exp2/...

Experiments/experiment_3/
  ff_capture.py   ff_dataset.py   ff_train.py   README_fading_fingerprints.md
  fingerprint_<tag>.pt / .json               # trained models + metadata
  .ff_cache/                                 # built-dataset npz cache
```

---

## Gotchas

- **Socket buffers** must be raised each boot (the sysctl above) before any GigE radio capture.
- **Capture to the internal disk** (default `~/captures/...`). Do NOT target the T9/T91 USB drives —
  they share the b200's USB bus and cause `BAD_PACKET`.
- **Radio init is slow** (USRP2/b200 > 60 s). `ff_capture.py` waits `--init-wait` (default 70 s) —
  don't kill it early.
- **Same target both days.** If a device can't reach −44.5 dB, capture it with `--fix-gain` at its
  cap and use that same fixed gain on Day 2 (note it in the device table).
- **Antenna per device:** B-series/b200 = `TX/RX`; USRP2/N-series = `J1` (some USRP2 units radiate
  stronger on `J2` — check the `signal dB` during the gain search and switch if it's weak).
- **The stock flowgraphs stay at 5.29 GHz** on disk; `ff_capture.py` stamps 3.3 GHz + the capture
  path into them at launch, so the parked attack setup remains revertible.
- **Cross-day needs matching device_ids** on both days, or a device is dropped from the intersection.

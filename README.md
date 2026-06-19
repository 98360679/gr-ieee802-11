# radio_adversarial_attacks

Software-defined-radio experiments around **IEEE 802.11 (Wi-Fi) signal capture**
and **RF device fingerprinting** with deep learning, built on top of GNU Radio
and Ettus USRP hardware (N210 / B200 / B205).

The work has two halves:

1. **Signal capture & link verification** — a headless GNU Radio flowgraph that
   receives and decodes 802.11 frames from a USRP, plus a tool to analyze the
   captured raw I/Q.
2. **Device fingerprinting** (`radio_fingerprint/`) — a PyTorch pipeline that
   identifies *which physical transmitter* sent a signal, purely from RF
   hardware impairments in the raw I/Q (time- and frequency-domain, real and
   complex-valued CNNs).

> **Hardware required.** The capture scripts talk to a real USRP over UHD. The
> fingerprinting pipeline runs on pre-captured `.bin` I/Q files and needs only a
> machine with PyTorch (GPU recommended).

---

## Repository layout

| Path | What it is |
|------|------------|
| `exp3_wifi_rx.py` | Headless 802.11 a/g receiver flowgraph (USRP → decode MAC), logs decoded frames and taps raw I/Q to `raw_iq.bin`. |
| `find_bursts.py` | Analyzes a `raw_iq.bin` capture: detects bursts and tests whether they form a periodic train at the expected TX cadence. |
| `radio_fingerprint/` | Deep-learning RF device-fingerprinting subproject (see its own [README](radio_fingerprint/README.md)). |
| `gr-ieee802-11-maint-3.10/` | Vendored copy of [gr-ieee802-11](https://github.com/bastibl/gr-ieee802-11) — the GNU Radio 802.11 a/g/p transceiver OOT module (maint-3.10 branch). |
| `gr-foo-maint-3.10/` | Vendored copy of [gr-foo](https://github.com/bastibl/gr-foo) — helper blocks that gr-ieee802-11 depends on. |

`*.bin`, `*.iq`, `*.npy`, `*.h5`, `*.pth`, `*.ckpt` and similar large/binary
artifacts are git-ignored (see `.gitignore`).

---

## 1. Signal capture & link verification

### Dependencies
- [GNU Radio](https://www.gnuradio.org/) 3.10
- [UHD](https://github.com/EttusResearch/uhd) drivers + a USRP (N210/B2xx)
- The `gr-foo` and `gr-ieee802-11` OOT modules **installed** so that
  `import ieee802_11` works (the vendored copies here are the sources; build &
  install them per their own READMEs).
- Python 3, NumPy

### Receive 802.11 frames — `exp3_wifi_rx.py`
A headless reimplementation of the stock `gr-ieee802-11` `wifi_rx.py` receive
chain (Schmidl-Cox short/long sync → FFT → frame equalizer → MAC decode). It:
- streams complex samples from the USRP at **5 MHz**,
- decodes 802.11 MAC frames,
- taps the raw I/Q to `<out>/raw_iq.bin`, and
- flags frames sent by a companion transmitter (recognized by a known MAC
  address `23:23:…`/`42:42:…` and an `'A'`-filled 500-byte payload carrying a
  4-character frame id), verifying received payloads **byte-for-byte**.

```bash
python3 exp3_wifi_rx.py --freq 2.45e9 --antenna RX2 --gain 1.0
python3 exp3_wifi_rx.py --device "serial=3256204" --freq 2.45e9 --antenna RX2
```

Key options: `--addr`/`--device` (UHD device), `--freq`, `--antenna`
(`RX2`/`J2`), `--gain` (normalized 0–1), `--duration` (seconds), `--out`
(capture dir), `--verbose`. At the end it prints how many frames decoded, how
many matched byte-for-byte, and how many had bit errors.

### Analyze a capture — `find_bursts.py`
Given a `raw_iq.bin`, detects energy bursts and checks whether they form a
periodic train at the expected transmit cadence (default 300 ms = 1.5 M samples
at 5 MHz) — useful for distinguishing *"frames arriving but not decoding"* from
*"frames never transmitted."*

```bash
python3 find_bursts.py ./capture/raw_iq.bin
```

---

## 2. RF device fingerprinting (`radio_fingerprint/`)

A PyTorch pipeline that fingerprints **4 transmitters** (`B200_1, B200_2,
B205_1, B205_2`) from raw I/Q using hardware-impairment cues. It evaluates two
signal representations × two model families:

|              | 1D-CNN (real, 2-ch) | CVNN (complex) |
|--------------|---------------------|----------------|
| **Time**     | `time_cnn1d`        | `time_cvnn`    |
| **Frequency**| `freq_cnn1d`        | `freq_cvnn`    |

Uses a **leakage-safe split** (train/val on `run_1+run_2` split *by burst*, test
on the unseen `run_3`) and reports both window-level and frame-level accuracy.

```bash
cd radio_fingerprint
python3 build_dataset.py      # Stage 1 → artifacts/windows_L512.npz
python3 plots_eda.py          # Stage 2 → figs/ EDA plots
python3 train.py --all        # Stage 3 → train 4 configs + confusion matrices
```

Data location, window size, splits and other knobs live in
`radio_fingerprint/config.py` (multi-day datasets selectable via the `FP_DAY`
env var). See the [subproject README](radio_fingerprint/README.md) and
`radio_fingerprint/REPORT.md` for full details and results.

---

## Credits & licensing

`gr-ieee802-11` and `gr-foo` are third-party GNU Radio modules by
[Bastian Bloessl](https://github.com/bastibl), vendored here for convenience and
distributed under their own licenses (see the `LICENSE` file inside each
directory). All other code in this repository is the authors' own experimental
work.

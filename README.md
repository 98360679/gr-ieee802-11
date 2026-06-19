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

## Installation

The two halves have independent requirements. You only need the GNU Radio stack
(below) for the **capture** scripts; the **fingerprinting** pipeline needs only
Python + PyTorch.

### Prerequisites
- **GNU Radio 3.10** and a C++ toolchain (`cmake`, `make`, a compiler) — required
  for the capture scripts.
- **UHD** drivers and a **USRP** (N210 / B200 / B205) — for live capture only.
- **Python 3** with `numpy`, `scipy`, `matplotlib`, `scikit-learn`, and
  **PyTorch** (`torch`) — for analysis and fingerprinting.

Install the Python packages (a virtualenv is recommended):

```bash
pip install numpy scipy matplotlib scikit-learn torch
# For GPU training, install the CUDA build of torch per https://pytorch.org/get-started/
```

### Build the GNU Radio OOT modules (for capture)

`exp3_wifi_rx.py` does `import ieee802_11`, which requires the `gr-foo` and
`gr-ieee802-11` modules to be **built and installed** into your GNU Radio
environment. Use the vendored copies in this repo (build `gr-foo` first — it's a
dependency of `gr-ieee802-11`):

```bash
# 1) gr-foo (dependency)
cd gr-foo-maint-3.10
mkdir -p build && cd build
cmake ..
make
sudo make install
sudo ldconfig
cd ../..

# 2) gr-ieee802-11
cd gr-ieee802-11-maint-3.10
mkdir -p build && cd build
cmake ..
make
sudo make install
sudo ldconfig
cd ../..
```

Then a couple of one-time post-install steps required by `gr-ieee802-11`:

```bash
# Build the OFDM PHY hierarchical block — open in GNU Radio Companion and
# generate it; this installs it under ~/.grc_gnuradio/ (where exp3_wifi_rx.py
# looks for it via GRC_HIER_PATH).
gnuradio-companion gr-ieee802-11-maint-3.10/examples/wifi_phy_hier.grc

# Tagged-stream blocks buffer a whole frame; raise max shared memory.
sudo sysctl -w kernel.shmmax=2147483648

# Optimize VOLK kernels for your CPU (recommended).
volk_profile
```

> If `import ieee802_11` fails afterward, your module was likely installed under
> a different `CMAKE_INSTALL_PREFIX` than GNU Radio uses — pass a matching
> `cmake -DCMAKE_INSTALL_PREFIX=…` and ensure `PYTHONPATH`/`LD_LIBRARY_PATH`
> cover it. See `gr-ieee802-11-maint-3.10/README.md` for full troubleshooting.

---

## 1. Signal capture & link verification

> Requires the GNU Radio OOT modules from [Installation](#installation) and a
> USRP reachable over UHD.

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

Needs only the Python packages from [Installation](#installation) (no GNU Radio
required) and pre-captured I/Q `.bin` files.

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

This project is licensed under the **GNU General Public License v3.0** — see
[`LICENSE`](LICENSE).

`gr-ieee802-11` and `gr-foo` are third-party GNU Radio modules by
[Bastian Bloessl](https://github.com/bastibl), vendored here for convenience.
They are themselves GPLv3 (see the `LICENSE` file inside each directory), which
is why the project as a whole is distributed under the same terms. All other
code in this repository is the authors' own experimental work.

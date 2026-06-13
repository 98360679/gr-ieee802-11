# Day1 Device Fingerprinting (mini-USRP)

Fingerprints the **4 Day1 transmitters** (`B200_1, B200_2, B205_1, B205_2`)
from raw I/Q, in **two representations** × **two model families**:

|              | 1D-CNN (real, 2-ch)   | CVNN (complex)        |
|--------------|-----------------------|-----------------------|
| **Time**     | `time_cnn1d`          | `time_cvnn`           |
| **Frequency**| `freq_cnn1d`          | `freq_cvnn`           |

`freq` = `fftshift(FFT(window))`. The CNN1D consumes `[real, imag]` as 2
channels; the CVNN consumes the complex window directly (complex conv / BN /
modReLU, magnitude before the classifier head).

## Data
- Source: `/media/nghoselab/T9/Data/mix_fingerprint/train/Day1/<device>/run_{1,2,3}.bin`
- Format: **complex64 (fc32)**, **Fs = 5 MHz**, ~35 s / run.
- **Leakage-safe split:** train/val on `run_1+run_2` (split *by burst*), test on the
  unseen `run_3`. Bursts (real data frames) are energy-detected; windows
  (`L=512`, stride 256) are power-normalized to unit average power so the model
  learns RF nuance, not gross amplitude.

## Run
```bash
python3 build_dataset.py      # Stage 1 -> artifacts/windows_L512.npz
python3 plots_eda.py          # Stage 2 -> figs/ EDA plots
python3 train.py --all        # Stage 3 -> train 4 configs, confusion matrices
```

## Outputs
- `figs/burst_detection.png`, `time_domain.png`, `constellation.png`,
  `psd_welch.png`, `avg_magnitude_fft.png`, `class_balance.png` — signal EDA.
- `figs/<config>_confusion.png` — window- & frame-level confusion matrices.
- `figs/<config>_training.png` — loss/accuracy curves.
- `figs/comparison.png` — 4-config accuracy comparison.
- `artifacts/<config>_metrics.json`, `summary.json`, `*_best.pt`.

Evaluation reports **window-level** accuracy and **frame-level** accuracy
(majority vote over windows sharing a burst), the more honest per-frame metric.

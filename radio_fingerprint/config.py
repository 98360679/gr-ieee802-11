"""Shared configuration for the device-fingerprinting pipeline.

Day is selected with the FP_DAY env var (default Day1):
    FP_DAY=Day2 python3 build_dataset.py
Device classes are auto-discovered from the day's sub-folders, and all
outputs are scoped to that day so multiple days coexist.
"""
import os

# ── Data ────────────────────────────────────────────────────────────────
DATA_BASE = "/media/nghoselab/T9/Data/mix_fingerprint/train"
DAY       = os.environ.get("FP_DAY", "Day1")
DATA_ROOT = os.path.join(DATA_BASE, DAY)
DEVICES   = sorted(d for d in os.listdir(DATA_ROOT)
                   if os.path.isdir(os.path.join(DATA_ROOT, d)))   # classes
RUNS      = [1, 2, 3]
TEST_RUN  = 3                      # held-out session (leakage-safe)
FS        = 5e6                    # sample rate (Hz)
DTYPE     = "complex64"            # GNU Radio fc32 file-sink, 8 bytes/sample

# ── Windowing / burst extraction ────────────────────────────────────────
WIN_LEN     = 512                  # samples per fingerprint window
WIN_STRIDE  = 256                  # hop within a burst (50% overlap)
ENV_WIN     = 200                  # envelope smoothing window (40 us)
MIN_BURST   = 4000                 # min active samples to count as a data frame
THR_FLOOR_K = 6.0                  # threshold = max(median*K, mean*2)
PRE_ROLL    = 0                    # samples kept before onset
EPS         = 1e-12

# Caps to keep training balanced & tractable (per device, per split)
MAX_TRAIN_PER_DEV = 25000
MAX_VAL_PER_DEV   = 4000
MAX_TEST_PER_DEV  = 8000
VAL_FRAC          = 0.15           # carved from run1+run2 by burst group
SEED              = 1234

# ── Paths (scoped per day) ──────────────────────────────────────────────
PROJ      = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(PROJ, "artifacts", DAY)
FIG_DIR   = os.path.join(PROJ, "figs", DAY)
DATASET   = os.path.join(OUT_DIR, f"windows_L{WIN_LEN}.npz")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

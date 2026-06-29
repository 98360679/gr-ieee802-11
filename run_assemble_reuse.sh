#!/usr/bin/env bash
# Assemble a new attack bundle REUSING the existing ch0 (eot_t4/adv_frame.bin +
# adv_index.csv) — only ch1 changes. No T91/TX recording needed.
# Usage: run_assemble_reuse.sh <pertdir_name> <out_subdir>
set -euo pipefail
OTA=/media/nghoselab/T9/Data/session13/ota_dev6
HERE=/home/nghoselab/Experiments/experiment_3
SRC="$OTA/eot_t4"                       # existing ch0 + index to reuse
PERTDIR="$1"; OUTSUB="$2"; BASE_PSR="${3:--20}"   # PSR the delta was crafted at
mkdir -p "$OTA/$OUTSUB"
cp "$SRC/adv_frame.bin" "$OTA/$OUTSUB/adv_frame.bin"   # ch0 identical

python3 "$HERE/exp3_assemble_pert.py" \
  --adv-frame "$SRC/adv_frame.bin" --adv-index "$SRC/adv_index.csv" \
  --pert-dir  "$OTA/$PERTDIR" --pert-glob '{id}.bin' \
  --out-pert  "$OTA/$OUTSUB/adv_perturbation.bin"

python3 "$HERE/exp3_psr_sweep.py" \
  --frame "$OTA/$OUTSUB/adv_frame.bin" \
  --pert  "$OTA/$OUTSUB/adv_perturbation.bin" \
  --base-psr "$BASE_PSR" --gap-ms 50 \
  --psr -40 -35 -30 -25 -20 -15 -10 -5 0 5 10 \
  --out   "$OTA/$OUTSUB/dac_safe_gapped"
echo "DONE -> $OTA/$OUTSUB/dac_safe_gapped  (ch0 adv_frame.bin, ch1 adv_perturbation_psr_*.bin)"

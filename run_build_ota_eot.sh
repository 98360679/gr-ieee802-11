#!/usr/bin/env bash
# Assemble an EOT per-frame OTA bundle, then DAC-safe PSR sweep.
# Usage: run_build_ota_eot.sh [pertdir_name] [out_subdir]   (defaults: targeted dev4)
#   pertdir_name : dir under ota_dev6/ with per-frame <id>.bin (default eot_t4_pertdir)
#   out_subdir   : output dir under ota_dev6/   (default eot_t4)
set -euo pipefail
TX=/media/nghoselab/T91/Data/session13/train/6_27_2026/device_6
OTA=/media/nghoselab/T9/Data/session13/ota_dev6
FORK=/home/nghoselab/gr-ieee802-11-exp3/examples
PERTDIR="${1:-eot_t4_pertdir}"
OUTSUB="${2:-eot_t4}"
mkdir -p "$OTA/$OUTSUB"

python3 "$FORK/build_adv_replay.py" \
  --frame-bin "$TX/frame_run_1.bin" \
  --index     "$TX/frame_index_run_1.csv" \
  --ids       "$OTA/eot_t4_ids.txt" \
  --pert-dir  "$OTA/$PERTDIR" --pert-glob '{id}.bin' --align data \
  --out-frame "$OTA/$OUTSUB/adv_frame.bin" \
  --out-pert  "$OTA/$OUTSUB/adv_perturbation.bin" \
  --out-index "$OTA/$OUTSUB/adv_index.csv"

# DAC-safe gapped PSR sweep (transmit these on the 2-ch MIMO)
python3 /home/nghoselab/Experiments/experiment_3/exp3_psr_sweep.py \
  --frame "$OTA/$OUTSUB/adv_frame.bin" \
  --pert  "$OTA/$OUTSUB/adv_perturbation.bin" \
  --base-psr -20 --gap-ms 50 \
  --out   "$OTA/$OUTSUB/dac_safe_gapped"
echo "DONE -> $OTA/$OUTSUB/dac_safe_gapped (ch0 adv_frame.bin, ch1 adv_perturbation_psr_*.bin)"

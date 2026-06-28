#!/usr/bin/env bash
# Assemble the EOT per-frame OTA bundle, then DAC-safe PSR sweep. Run after the
# per-frame deltas in eot_t4_pertdir/ are ready.
set -euo pipefail
TX=/media/nghoselab/T91/Data/session13/train/6_27_2026/device_6
OTA=/media/nghoselab/T9/Data/session13/ota_dev6
FORK=/home/nghoselab/gr-ieee802-11-exp3/examples

python3 "$FORK/build_adv_replay.py" \
  --frame-bin "$TX/frame_run_1.bin" \
  --index     "$TX/frame_index_run_1.csv" \
  --ids       "$OTA/eot_t4_ids.txt" \
  --pert-dir  "$OTA/eot_t4_pertdir" --pert-glob '{id}.bin' --align data \
  --out-frame "$OTA/eot_t4/adv_frame.bin" \
  --out-pert  "$OTA/eot_t4/adv_perturbation.bin" \
  --out-index "$OTA/eot_t4/adv_index.csv"

# DAC-safe gapped PSR sweep (transmit these on the 2-ch MIMO)
python3 /home/nghoselab/Experiments/experiment_3/exp3_psr_sweep.py \
  --frame "$OTA/eot_t4/adv_frame.bin" \
  --pert  "$OTA/eot_t4/adv_perturbation.bin" \
  --base-psr -20 --gap-ms 50 \
  --out   "$OTA/eot_t4/dac_safe_gapped"
echo "DONE -> $OTA/eot_t4/dac_safe_gapped (ch0 adv_frame.bin, ch1 adv_perturbation_psr_*.bin)"

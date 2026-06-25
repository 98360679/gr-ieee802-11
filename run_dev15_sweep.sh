#!/usr/bin/env bash
# Focused tuning sweep on the device_1 <-> device_5 confusion (4-device model).
# Each config is tagged so the canonical fingerprint_cnn_retrained_4dev.pt
# (committed-best, 94.3% frame) is never overwritten.
set -u
cd /home/nghoselab/Experiments/experiment_3
SUM=/tmp/exp3_sweep_summary.txt
: > "$SUM"

run() {
  local tag="$1"; shift
  echo "############ CONFIG: $tag ############" | tee -a "$SUM"
  python3 -u exp3_train_fingerprint.py --exclude 2 --aug --tag "$tag" "$@" \
      > "/tmp/exp3_sweep_${tag}.log" 2>&1
  echo "[$tag] $(grep '^BEST' /tmp/exp3_sweep_${tag}.log)" | tee -a "$SUM"
  # evaluate the saved tagged model, keep only the summary + dev1/dev5 rows
  python3 -u exp3_confusion.py --exclude 2 \
      --model "fingerprint_cnn_retrained_4dev_${tag}.pt" 2>/dev/null \
    | grep -E "val window acc|FRAME confusion|device_1 |device_5 |top confusions" \
    | sed "s/^/[$tag] /" | tee -a "$SUM"
  echo | tee -a "$SUM"
}

run cool      --lr 5e-4 --epochs 150 --wd 1e-4 --label-smooth 0.02 --snr-lo 12 --snr-hi 35
run mildaug   --lr 1e-3 --epochs 130 --wd 1e-4 --label-smooth 0.05 --snr-lo 20 --snr-hi 40
run strongreg --lr 1e-3 --epochs 150 --wd 3e-4 --label-smooth 0.05 --snr-lo 8  --snr-hi 32

echo "=========== SWEEP DONE ===========" | tee -a "$SUM"

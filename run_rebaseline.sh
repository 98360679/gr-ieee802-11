#!/usr/bin/env bash
# run_rebaseline.sh — re-validate the 6-class model on TODAY's fresh clean captures
# (tests the "channel drifted from yesterday" hypothesis before any fine-tune).
#
# Drop ONE clean capture per device into:
#   /media/nghoselab/T9/Data/session13/rebaseline_clean/device_<d>/<anything>.bin
# (raw complex64, 5 MHz, ~30 s, same TX cadence as training — like train/device_*/clean_run_*.bin)
# then run this. Partial is fine: it scores whatever devices have a .bin.
set -euo pipefail
cd "$(dirname "$0")"

ROOT=/media/nghoselab/T9/Data/session13/rebaseline_clean
MODEL=fingerprint_cnn_retrained.pt          # the canonical 6-class model (NOT the 4-dev default)
OUT=rebaseline_$(date +%Y%m%d).json

echo "readiness (.bin per device dir):"
for d in 1 2 3 4 5 6; do
  n=$(find "$ROOT/device_$d" -maxdepth 1 -name '*.bin' 2>/dev/null | wc -l)
  printf "  device_%s: %s file(s)\n" "$d" "$n"
done
echo

python3 exp3_rebaseline.py --root "$ROOT" --model "$MODEL" --out "$OUT"
echo
echo "wrote $OUT"
echo "VERDICT line: 'TRANSFERRED' -> model fine, the OTA failure is delta delivery (loopback test)."
echo "             'DEGRADED'   -> channel drift confirmed -> fine-tune/retrain on these fresh captures."

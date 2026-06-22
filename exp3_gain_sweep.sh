#!/usr/bin/env bash
# exp3_gain_sweep.sh — find the RX gain that maximizes error-free frame rate.
# Captures a short clip at each gain (exp3_wifi_rx.py file_sink), runs
# exp3_ber_eval.py, and tabulates error-free rate. Requires the TX on air.
#
# Usage: ./exp3_gain_sweep.sh [duration_s] [gain1 gain2 ...]
set -u
cd "$(dirname "$0")"
DEV="addr=192.168.10.4"
DUR="${1:-12}"; shift || true
GAINS=("$@"); [ ${#GAINS[@]} -eq 0 ] && GAINS=(0.01 0.015 0.02 0.03 0.05)
TMP=/tmp/gain_sweep; mkdir -p "$TMP"
echo "gain,err_free_pct,fer,ber,aligned_frames" > "$TMP/summary.csv"

for g in "${GAINS[@]}"; do
  echo "──────── gain $g ────────"
  out="$TMP/g_$g"; mkdir -p "$out"
  python3 exp3_wifi_rx.py --device "$DEV" --gain "$g" --duration "$DUR" --out "$out" \
      >/dev/null 2>&1
  python3 exp3_ber_eval.py --file "$out/raw_iq.bin" --tag "g$g" \
      --json "$TMP/ber_$g.json" 2>&1 | grep -E "error-free|>>> BER|>>> FER|ours, payload"
  python3 - "$g" "$TMP/ber_$g.json" "$TMP/summary.csv" <<'PY'
import json, sys
g, jf, csv = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(jf))
    F = d.get('frames_ours_aligned', 0); ef = d.get('frames_error_free', 0)
    pct = 100*ef/F if F else 0
    row = f"{g},{pct:.1f},{d.get('fer',0):.4f},{d.get('ber',0):.3e},{F}"
except Exception as e:
    row = f"{g},ERR,{e},,"
open(csv, 'a').write(row + "\n")
PY
  rm -f "$out/raw_iq.bin"          # reclaim space immediately
done

echo; echo "==================== SUMMARY ===================="
column -t -s, "$TMP/summary.csv"
echo "best by error-free rate:"
tail -n +2 "$TMP/summary.csv" | sort -t, -k2 -nr | head -1 | column -t -s,

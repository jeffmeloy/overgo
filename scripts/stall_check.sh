#!/usr/bin/env bash
# stall_check.sh <task-output-file> [stall-minutes]
# Verdict line for the loop's heartbeat wake: OK / STALL from GPU util plus
# transcript mtime age. Threshold default 15 min (doctrine: checkpoint-demand
# point). Deterministic watchdog per no-silent-hangs.
set -u
file="${1:-}"
threshold_min="${2:-15}"
util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)"
util="${util:-unknown}"
if [ -z "$file" ] || [ ! -f "$file" ]; then
  echo "STALL_CHECK file=absent gpu_util=${util}% verdict=NO_TRANSCRIPT"
  exit 0
fi
now=$(date +%s)
mtime=$(stat -c %Y "$file" 2>/dev/null || echo "$now")
age_min=$(( (now - mtime) / 60 ))
size=$(stat -c %s "$file" 2>/dev/null || echo 0)
verdict=OK
if [ "$age_min" -ge "$threshold_min" ] && [ "${util}" != "unknown" ] && [ "${util}" -lt 5 ]; then
  verdict=STALL
fi
echo "STALL_CHECK file=$file size=$size age_min=$age_min gpu_util=${util}% verdict=$verdict"

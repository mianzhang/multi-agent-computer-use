#!/usr/bin/env bash
# Wait for the replan=5 recovery run to finish, then score it.
set -uo pipefail
cd "$(dirname "$0")/../.."
RD=runs/odysseys45_replan5
# Wait until 45 final_results OR the recovery runner (odyfix) is gone.
for i in $(seq 1 240); do   # up to ~2h (30s ticks)
  n=$(ls $RD/*/final_results.json 2>/dev/null | wc -l)
  [ "$n" -ge 45 ] && { echo "replan5 reached $n/45"; break; }
  tmux has-session -t odyfix 2>/dev/null || { echo "odyfix gone; replan5 at $n/45"; break; }
  sleep 30
done
n=$(ls $RD/*/final_results.json 2>/dev/null | wc -l)
echo "Scoring $RD ($n/45 complete) at $(date +%H:%M:%S)"
bash scripts/lib/run_ody_eval.sh "$RD"

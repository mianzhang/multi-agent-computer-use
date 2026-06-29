#!/usr/bin/env bash
# Wait for an odysseys run to finish, then score it.
# Usage: auto_score.sh <run-dir> <tmux-session-to-watch>
# Finishes when the run dir has 45 final_results.json OR the watched session is gone.
set -uo pipefail
cd "$(dirname "$0")/../.."
RD="${1:?usage: auto_score.sh <run-dir> <watch-session>}"
WATCH="${2:?usage: auto_score.sh <run-dir> <watch-session>}"
TARGET="${TARGET:-45}"
for i in $(seq 1 480); do   # up to ~4h (30s ticks)
  n=$(ls $RD/*/final_results.json 2>/dev/null | wc -l)
  [ "$n" -ge "$TARGET" ] && { echo "$RD reached $n/$TARGET"; break; }
  tmux has-session -t "$WATCH" 2>/dev/null || { echo "session $WATCH gone; $RD at $n/$TARGET"; break; }
  sleep 30
done
n=$(ls $RD/*/final_results.json 2>/dev/null | wc -l)
echo "Scoring $RD ($n complete) at $(date +%H:%M:%S)"
bash scripts/lib/run_ody_eval.sh "$RD"
echo "Scored $RD at $(date +%H:%M:%S)"

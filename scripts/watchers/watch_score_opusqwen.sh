#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
RD=runs/odysseys45_opus48mgr_qwen37plus_replan5
echo "[scorer] $(date +%H:%M:%S) waiting for $RD to reach 45/45 (or odyopus to end)..."
for i in $(seq 1 480); do   # up to ~4h
  n=$(ls $RD/*/final_results.json 2>/dev/null | wc -l)
  [ "$n" -ge 45 ] && { echo "[scorer] done $n/45 @ $(date +%H:%M:%S)"; break; }
  tmux has-session -t odyopus 2>/dev/null || { echo "[scorer] odyopus ended (n=$n) @ $(date +%H:%M:%S)"; break; }
  sleep 30
done
echo "[scorer] scoring $RD with gemini-3.1-flash-lite @ $(date +%H:%M:%S)"
bash scripts/lib/run_ody_eval.sh "$RD" 2>&1 | grep -iE "Summary|Judge cost" || true
echo "[scorer] DONE @ $(date +%H:%M:%S)"

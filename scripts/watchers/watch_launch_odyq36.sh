#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
echo "[watcher] $(date +%H:%M:%S) waiting for OSWorld qwen36 replan sweep (oswq36r0/r1/r5) to finish..."
while tmux has-session -t oswq36r0 2>/dev/null || tmux has-session -t oswq36r1 2>/dev/null || tmux has-session -t oswq36r5 2>/dev/null; do
  sleep 120
done
echo "[watcher] OSWorld qwen36 sweep done; launching odyssey qwen36 sweep @ $(date +%H:%M:%S)"
tmux kill-session -t odyq36 2>/dev/null
tmux new-session -d -s odyq36 "MODEL_ID=fireworks_ai/deployed/t5cvm6h4 LABEL=qwen36 bash scripts/lib/run_odysseys_model_sweep.sh > logs/odysseys_qwen36_sweep.log 2>&1"
echo "[watcher] launched odyq36 (qwen36 odysseys sweep) @ $(date +%H:%M:%S)"

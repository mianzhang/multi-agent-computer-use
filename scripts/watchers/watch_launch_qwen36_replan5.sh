#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
ND=runs/test_small_qwen36_nomanager
echo "[watcher] $(date +%H:%M:%S) waiting for qwen36 nomanager to finish (39/39)..."
while true; do
  done=$(ls $ND/*/final_results.json 2>/dev/null | wc -l)
  if [ "$done" -ge 39 ]; then echo "[watcher] nomanager done ($done/39) @ $(date +%H:%M:%S)"; break; fi
  tmux has-session -t oswq36 2>/dev/null || { echo "[watcher] oswq36 session gone (done=$done) @ $(date +%H:%M:%S)"; break; }
  sleep 60
done
rm -rf runs/test_small_qwen36_replan5
tmux kill-session -t oswq36r5 2>/dev/null
tmux new-session -d -s oswq36r5 "bash scripts/experiments/run_qwen36_replan5.sh > logs/osworld_test_small_qwen36_replan5.log 2>&1"
echo "[watcher] launched oswq36r5 (replan5) @ $(date +%H:%M:%S)"

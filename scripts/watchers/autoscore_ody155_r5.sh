#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
D=runs/odysseys155_qwen37plus_replan5
while tmux has-session -t ody155_r5 2>/dev/null; do sleep 300; done
echo "[autoscore] $(date +%m-%d_%H:%M) ody155_r5 ended; one-shot scoring ($(ls $D/*/final_results.json 2>/dev/null|wc -l) tasks)"
bash scripts/lib/run_ody_eval.sh "$D" 2>&1 | grep -iE "Summary|Backend" || true
echo "[autoscore] done @ $(date +%H:%M)"

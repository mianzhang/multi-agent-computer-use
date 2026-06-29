#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
D=runs/odysseys45_qwen37plus_replan10
while tmux has-session -t ody45_r10 2>/dev/null; do sleep 300; done
echo "[autoscore] $(date +%m-%d_%H:%M) ody45_r10 ended; one-shot scoring ($(ls $D/*/final_results.json 2>/dev/null|wc -l) tasks)"
bash scripts/lib/run_ody_eval.sh "$D" 2>&1 | grep -iE "Summary|Backend" || true
echo "[autoscore] done @ $(date +%H:%M)"

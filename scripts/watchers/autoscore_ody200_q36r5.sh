#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
D=runs/odysseys200_qwen36_replan5
while tmux has-session -t ody200_q36r5 2>/dev/null; do sleep 600; done
echo "[autoscore] $(date +%m-%d_%H:%M) ody200_q36r5 ended; one-shot scoring ($(ls $D/*/final_results.json 2>/dev/null|wc -l) tasks)"
bash scripts/lib/run_ody_eval.sh "$D" 2>&1 | grep -iE "Summary|Backend" || true
echo "[autoscore] done @ $(date +%H:%M)"

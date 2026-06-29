#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
D=runs/odysseys200_qwen36_replan5
while tmux has-session -t q36_fin2 2>/dev/null; do sleep 180; done
echo "[finalize] $(date +%m-%d_%H:%M) ended; rescore ($(ls $D/*/final_results.json|wc -l) done)"
bash scripts/lib/run_ody_eval.sh "$D" 2>&1 | grep -iE "Summary|Backend" || true

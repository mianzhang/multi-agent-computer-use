#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
D=runs/odysseys45_qwen37plus_replan10
while tmux has-session -t ody45_r10b 2>/dev/null; do sleep 120; done
echo "[finalize_r10] $(date +%H:%M) rerun ended; full-45 one-shot rescore ($(ls $D/*/final_results.json|wc -l) done)"
bash scripts/lib/run_ody_eval.sh "$D" 2>&1 | grep -iE "Summary|Backend" || true

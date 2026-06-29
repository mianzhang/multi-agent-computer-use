#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
while tmux has-session -t q36_fin 2>/dev/null; do sleep 180; done
echo "[finalize] $(date +%m-%d_%H:%M) q36_fin ended; rescore (185 done)"
bash scripts/lib/run_ody_eval.sh "runs/odysseys200_qwen36_replan5" 2>&1 | grep -iE "Summary|Backend" || true

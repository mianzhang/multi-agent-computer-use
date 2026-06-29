#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
while tmux has-session -t r10_fin 2>/dev/null; do sleep 180; done
echo "[finalize] $(date +%m-%d_%H:%M) r10_fin ended; rescore (43 done)"
bash scripts/lib/run_ody_eval.sh "runs/odysseys45_qwen37plus_replan10" 2>&1 | grep -iE "Summary|Backend" || true

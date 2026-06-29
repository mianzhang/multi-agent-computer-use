#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
for pair in "ody155_q37:runs/odysseys155_qwen37plus_nomanager" "ody155_q36:runs/odysseys155_qwen36_nomanager"; do
  sess="${pair%%:*}"; D="${pair##*:}"
  while tmux has-session -t "$sess" 2>/dev/null; do sleep 120; done
  echo "[autoscore] $(date +%H:%M) $sess ended; one-shot scoring $D ($(ls $D/*/final_results.json 2>/dev/null|wc -l) tasks)"
  bash scripts/lib/run_ody_eval.sh "$D" 2>&1 | grep -iE "Summary|Backend" || true
done
echo "[autoscore] both scored @ $(date +%H:%M)"

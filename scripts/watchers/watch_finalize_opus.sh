#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
D=runs/odysseys45_opus48mgr_qwen37plus_replan5
echo "[finalize] $(date +%H:%M:%S) waiting for re-run (odyopusrerun) AND re-score (rescoreopus) to finish..."
for i in $(seq 1 360); do   # up to 3h
  rr=$(tmux has-session -t odyopusrerun 2>/dev/null && echo 1 || echo 0)
  rs=$(tmux has-session -t rescoreopus 2>/dev/null && echo 1 || echo 0)
  [ "$rr" = 0 ] && [ "$rs" = 0 ] && break
  sleep 30
done
n=$(ls $D/*/final_results.json 2>/dev/null | wc -l)
echo "[finalize] both done @ $(date +%H:%M:%S); $n/45 tasks have results. Incremental-scoring the new ones (skips already-scored)..."
bash scripts/lib/run_ody_eval.sh "$D" 2>&1 | grep -iE "Summary|Judge cost|Skipping|evaluating" || true
echo "[finalize] DONE @ $(date +%H:%M:%S)"

#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
D=runs/odysseys45_opus48mgr_qwen37plus_replan5
# 1) wait for current re-run (2 runners + 2 failed) to end
echo "[complete] $(date +%H:%M:%S) waiting for odyopusrerun to end..."
while tmux has-session -t odyopusrerun 2>/dev/null; do sleep 30; done
echo "[complete] re-run ended @ $(date +%H:%M:%S); done=$(ls $D/*/final_results.json 2>/dev/null|wc -l)/45"
# 2) retry the still-missing tasks ALONE (no re-score contention now)
for t in 39255449e341c41a589b8a4e17f073be3a4809c9 73c63095aeed43efb10a74eee7db7459c5ea9f84; do
  [ -f "$D/$t/final_results.json" ] || rm -rf "$D/$t"
done
missing=$(ls -d $D/*/ 2>/dev/null | wc -l)
echo "[complete] retrying still-missing tasks alone @ $(date +%H:%M:%S)"
NUM_WORKERS=2 bash scripts/experiments/run_odysseys_opus48_qwen37plus_replan5.sh > logs/odysseys_opus48_replan5_retry2.log 2>&1
echo "[complete] retry done @ $(date +%H:%M:%S); done=$(ls $D/*/final_results.json 2>/dev/null|wc -l)/45"
# 3) final incremental score (skips already-scored, adds the new)
bash scripts/lib/run_ody_eval.sh "$D" 2>&1 | grep -iE "Summary|Judge cost|evaluating" || true
echo "[complete] FINAL DONE @ $(date +%H:%M:%S)"

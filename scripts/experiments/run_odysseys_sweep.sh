#!/usr/bin/env bash
# Sequential replan-budget sweep over the 45-task odysseys subset.
# Runs budgets 0, 1, 5 one after another (each a full multi-worker run),
# then scores each with odysseys_eval.py (gemini-3.1-flash-lite judge via proxy).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

TASKS_FILE="data/odysseys/odysseys45.json"
export NUM_WORKERS="${NUM_WORKERS:-6}"
export MAX_PARALLELISM="${MAX_PARALLELISM:-3}"
export MAX_STEPS="${MAX_STEPS:-50}"
export TASKS_FILE

for BUDGET in 0 1 5; do
  RD="runs/odysseys45_replan${BUDGET}"
  echo "================ BUDGET=$BUDGET -> $RD ================"
  # clean stale apptainer overlays between budgets (fresh VM pool)
  rm -f apptainer/overlays/* 2>/dev/null
  rm -rf apptainer/run/* "$OSWORLD_ROOT/apptainer" 2>/dev/null
  MAX_REPLANS="$BUDGET" RESULT_DIR="$RD" bash scripts/lib/run_odysseys.sh
  echo "---- scoring $RD with gemini-3.1-flash-lite ----"
  .venv/bin/python evals/odysseys_eval.py \
    --runs-dir "$RD" \
    --task-source-json data/odysseys/odysseys.json \
    --output "$RD/odysseys_eval.json" \
    --num-workers 8 --max-images 40 2>&1 | grep -iE "Summary|Judge cost" || true
done
echo "================ SWEEP COMPLETE ================"

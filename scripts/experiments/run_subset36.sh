#!/usr/bin/env bash
# Run MACU over the 36-task stratified OSWorld subset.
#   CUA subagent : gpt-5.4-mini      (--cua-provider openai, via OPENAI_BASE_URL proxy)
#   Manager      : claude-opus-4-6   (--manager-provider openai, same proxy)
#   VM backend   : apptainer (QEMU/KVM)
#
# Cross-task parallelism is achieved by launching NUM_WORKERS worker processes
# that share one --result-dir and atomically claim tasks via .claimed files.
# Within each task, subtasks run with --max-parallelism. Each VM uses ~4 vCPU +
# 4 GB; peak VMs ~= NUM_WORKERS * (MAX_PARALLELISM + 1).
set -euo pipefail

cd "$(dirname "$0")/../.."

# ---- config (override via env) ------------------------------------------------
NUM_WORKERS="${NUM_WORKERS:-6}"        # parallel worker processes (tasks in flight)
MAX_PARALLELISM="${MAX_PARALLELISM:-3}" # parallel CUA subtasks within one task
MAX_STEPS="${MAX_STEPS:-50}"           # max CUA steps per subtask
MAX_REPLANS="${MAX_REPLANS:-10}"       # manager replanning budget per task
# Result dir defaults to include the replanning budget for easy comparison.
RESULT_DIR="${RESULT_DIR:-runs/subset36_replan${MAX_REPLANS}}"
TASKS_FILE="${TASKS_FILE:-data/osworld/subset36.json}"
MANAGER_MODEL="${MANAGER_MODEL:-claude-opus-4-6}"
CUA_MODEL="${CUA_MODEL:-gpt-5.4-mini}"
# ------------------------------------------------------------------------------

# Load API keys + OSWORLD_ROOT + absolute apptainer paths.
set -a; source .env; set +a

: "${OSWORLD_ROOT:?OSWORLD_ROOT must be set in .env}"

mkdir -p "$RESULT_DIR"
echo "Launching $NUM_WORKERS worker(s) over $TASKS_FILE -> $RESULT_DIR"
echo "  manager=$MANAGER_MODEL  cua=$CUA_MODEL  max_parallelism=$MAX_PARALLELISM  max_steps=$MAX_STEPS  max_replans=$MAX_REPLANS"

pids=()
for w in $(seq 1 "$NUM_WORKERS"); do
  log="$RESULT_DIR/worker_${w}.log"
  .venv/bin/python run_macu.py "$TASKS_FILE" \
    --result-dir "$RESULT_DIR" \
    --osworld-root "$OSWORLD_ROOT" \
    --osworld-data-dir "$OSWORLD_ROOT/evaluation_examples" \
    --manager-provider openai --manager-model "$MANAGER_MODEL" \
    --cua-provider openai \
    --max-parallelism "$MAX_PARALLELISM" \
    --max-replans "$MAX_REPLANS" \
    -- --headless --provider_name apptainer \
       --model "$CUA_MODEL" --max_steps "$MAX_STEPS" --sleep_after_execution 3.0 \
    > "$log" 2>&1 &
  pids+=($!)
  echo "  worker $w -> pid ${pids[-1]}  (log: $log)"
  sleep 2   # small stagger so workers don't race the same first task / ports
done

echo "Waiting for $NUM_WORKERS worker(s) to finish..."
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "worker pid $pid exited non-zero"; fail=1; }
done

echo
echo "===== SUBSET36 RESULTS ====="
.venv/bin/python - "$RESULT_DIR" <<'PY'
import json, sys, glob, os
rd = sys.argv[1]
scores = {}
for fr in glob.glob(os.path.join(rd, "*", "final_results.json")):
    tid = os.path.basename(os.path.dirname(fr))
    try:
        d = json.load(open(fr))
        scores[tid] = d.get("osworld_evaluator_score")
    except Exception as e:
        scores[tid] = f"ERR {e}"
done_n = sum(1 for v in scores.values() if isinstance(v, (int, float)))
succ = sum(1 for v in scores.values() if isinstance(v, (int, float)) and v >= 1.0)
print(f"tasks with score: {done_n}/36   success(>=1.0): {succ}   SR={succ/done_n*100:.1f}%" if done_n else "no scored tasks yet")
for tid in sorted(scores):
    print(f"  {scores[tid]}\t{tid}")
PY

exit $fail

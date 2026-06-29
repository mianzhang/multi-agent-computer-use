#!/usr/bin/env bash
# Run MACU over an odysseys task subset (web CUA tasks).
#   manager : claude-opus-4-8   (--manager-provider openai, via the LiteLLM proxy)
#   CUA     : gpt-5.4-mini       (--cua-provider openai)
#   VM      : apptainer (QEMU/KVM)
# Cross-task parallelism via NUM_WORKERS workers sharing one --result-dir
# (atomic .claimed). Result dir encodes the replan budget.
set -uo pipefail
cd "$(dirname "$0")/../.."

NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_PARALLELISM="${MAX_PARALLELISM:-3}"
MAX_STEPS="${MAX_STEPS:-50}"
MAX_REPLANS="${MAX_REPLANS:-1}"
TASKS_FILE="${TASKS_FILE:-data/odysseys/odysseys45.json}"
RESULT_DIR="${RESULT_DIR:-runs/odysseys45_replan${MAX_REPLANS}}"
MANAGER_MODEL="${MANAGER_MODEL:-claude-opus-4-8}"
CUA_MODEL="${CUA_MODEL:-gpt-5.4-mini}"
CUA_PROVIDER="${CUA_PROVIDER:-openai}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # extra run_macu.py args, e.g. "--no-manager"

set -a; source .env; set +a
: "${OSWORLD_ROOT:?OSWORLD_ROOT must be set in .env}"
mkdir -p "$RESULT_DIR"

echo "Launching $NUM_WORKERS worker(s) over $TASKS_FILE -> $RESULT_DIR"
echo "  manager=$MANAGER_MODEL  cua=$CUA_MODEL  parallelism=$MAX_PARALLELISM  max_steps=$MAX_STEPS  max_replans=$MAX_REPLANS"

pids=()
for w in $(seq 1 "$NUM_WORKERS"); do
  .venv/bin/python -u run_macu.py "$TASKS_FILE" \
    --result-dir "$RESULT_DIR" \
    --osworld-root "$OSWORLD_ROOT" \
    --manager-provider openai --manager-model "$MANAGER_MODEL" \
    --cua-provider "$CUA_PROVIDER" \
    --max-parallelism "$MAX_PARALLELISM" \
    --max-replans "$MAX_REPLANS" \
    --max_steps "$MAX_STEPS" \
    $EXTRA_ARGS \
    -- --headless --provider_name apptainer \
       --model "$CUA_MODEL" --max_steps "$MAX_STEPS" --sleep_after_execution 3.0 \
    > "$RESULT_DIR/worker_${w}.log" 2>&1 &
  pids+=($!)
  echo "  worker $w -> pid ${pids[-1]}"
  sleep 2
done

echo "Waiting for $NUM_WORKERS worker(s)..."
for pid in "${pids[@]}"; do wait "$pid" || echo "worker $pid exited non-zero"; done

done_n=$(ls "$RESULT_DIR"/*/final_results.json 2>/dev/null | wc -l)
echo "[$RESULT_DIR] complete: $done_n task(s) have final_results.json"

#!/usr/bin/env bash
# Run MACU over an OSWorld task set ({domain:[ids]} json, programmatic evaluators).
# Configurable via env: TASKS_FILE, RESULT_DIR, MANAGER_MODEL, CUA_MODEL,
# CUA_PROVIDER, MAX_STEPS, MAX_REPLANS, NUM_WORKERS, MAX_PARALLELISM.
set -uo pipefail
cd "$(dirname "$0")/../.."

NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_PARALLELISM="${MAX_PARALLELISM:-4}"
MAX_STEPS="${MAX_STEPS:-60}"
MAX_REPLANS="${MAX_REPLANS:-1}"
TASKS_FILE="${TASKS_FILE:?set TASKS_FILE to an OSWorld tasks json}"
RESULT_DIR="${RESULT_DIR:-runs/osworld_replan${MAX_REPLANS}}"
MANAGER_MODEL="${MANAGER_MODEL:-fireworks_ai/qwen3p7-plus}"
CUA_MODEL="${CUA_MODEL:-fireworks_ai/qwen3p7-plus}"
CUA_PROVIDER="${CUA_PROVIDER:-qwen}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # extra run_macu.py args, e.g. "--no-manager"

set -a; source .env; set +a
: "${OSWORLD_ROOT:?OSWORLD_ROOT must be set in .env}"
mkdir -p "$RESULT_DIR"

echo "Launching $NUM_WORKERS worker(s) over $TASKS_FILE -> $RESULT_DIR"
echo "  manager=$MANAGER_MODEL  cua=$CUA_MODEL ($CUA_PROVIDER)  parallelism=$MAX_PARALLELISM  max_steps=$MAX_STEPS  max_replans=$MAX_REPLANS"

pids=()
for w in $(seq 1 "$NUM_WORKERS"); do
  .venv/bin/python -u run_macu.py "$TASKS_FILE" \
    --result-dir "$RESULT_DIR" \
    --osworld-root "$OSWORLD_ROOT" \
    --osworld-data-dir "$OSWORLD_ROOT/evaluation_examples" \
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

echo "===== RESULTS: $RESULT_DIR ====="
.venv/bin/python - "$RESULT_DIR" <<'PY'
import json, sys, glob, os
rd = sys.argv[1]
scores = {}
for fr in glob.glob(os.path.join(rd, "*", "final_results.json")):
    tid = os.path.basename(os.path.dirname(fr))
    try:
        scores[tid] = json.load(open(fr)).get("osworld_evaluator_score")
    except Exception as e:
        scores[tid] = f"ERR {e}"
done_n = sum(1 for v in scores.values() if isinstance(v, (int, float)))
succ = sum(1 for v in scores.values() if isinstance(v, (int, float)) and v >= 1.0)
print(f"scored: {done_n}   success(>=1.0): {succ}   SR={succ/done_n*100:.1f}%" if done_n else "no scored tasks yet")
PY

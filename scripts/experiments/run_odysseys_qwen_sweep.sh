#!/usr/bin/env bash
# Sequential sweep over odysseys45 with Qwen3.7-Plus as BOTH manager and CUA.
#   manager = fireworks_ai/qwen3p7-plus
#   CUA     = fireworks_ai/qwen3p7-plus
#   max_steps (per subagent) = 100
#   configs: nomanager, replan0, replan1, replan5  (run one after another)
# Each run is scored with gemini-3.1-flash-lite (via proxy). Result dirs get a
# qwen37 prefix so the gpt-5.4-mini runs are untouched.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

export NUM_WORKERS="${NUM_WORKERS:-6}"
export MAX_PARALLELISM="${MAX_PARALLELISM:-4}"
export MAX_STEPS=100
export MANAGER_MODEL="fireworks_ai/qwen3p7-plus"
export CUA_MODEL="fireworks_ai/qwen3p7-plus"
export CUA_PROVIDER="qwen"   # qwen35vl_agent path (gpt54/openai computer_use tool is unsupported on fireworks)
export TASKS_FILE="data/odysseys/odysseys45.json"

# config name -> "MAX_REPLANS|EXTRA_ARGS"
run_one () {
  local name="$1" budget="$2" extra="$3"
  local RD="runs/odysseys45_qwen37_${name}"
  echo "================ $name (max_replans=$budget, extra='$extra') -> $RD ================"
  rm -rf "$RD"
  rm -f apptainer/overlays/* 2>/dev/null
  rm -rf apptainer/run/* "$OSWORLD_ROOT/apptainer" 2>/dev/null
  MAX_REPLANS="$budget" RESULT_DIR="$RD" EXTRA_ARGS="$extra" bash scripts/lib/run_odysseys.sh
  echo "---- scoring $RD with gemini-3.1-flash-lite ----"
  bash scripts/lib/run_ody_eval.sh "$RD" 2>&1 | grep -iE "Summary|Judge cost" || true
}

run_one nomanager 0 "--no-manager"
run_one replan0   0 ""
run_one replan1   1 ""
run_one replan5   5 ""
echo "================ QWEN SWEEP COMPLETE ================"

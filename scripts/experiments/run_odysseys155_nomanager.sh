#!/usr/bin/env bash
# Single-agent (no-manager) over the 155-task odysseys complement.
# Usage: LABEL=qwen37plus CUA_MODEL=fireworks_ai/qwen3p7-plus bash scripts/experiments/run_odysseys155_nomanager.sh
cd "$(dirname "$0")/../.."
LABEL="${LABEL:?set LABEL}"
NUM_WORKERS="${NUM_WORKERS:-10}" \
MAX_PARALLELISM=1 \
MAX_STEPS=100 \
MAX_REPLANS=0 \
TASKS_FILE=data/odysseys/odysseys155.json \
RESULT_DIR="runs/odysseys155_${LABEL}_nomanager" \
CUA_MODEL="${CUA_MODEL:?set CUA_MODEL}" \
CUA_PROVIDER="${CUA_PROVIDER:-qwen}" \
EXTRA_ARGS="--no-manager" \
  bash scripts/lib/run_odysseys.sh

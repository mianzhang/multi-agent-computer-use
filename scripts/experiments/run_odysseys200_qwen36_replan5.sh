#!/usr/bin/env bash
# MACU replan5, Qwen3.6-27B as BOTH manager and CUA, over ALL 200 odysseys tasks.
cd "$(dirname "$0")/../.."
NUM_WORKERS="${NUM_WORKERS:-8}" \
MAX_PARALLELISM=4 \
MAX_STEPS=60 \
MAX_REPLANS=5 \
TASKS_FILE=data/odysseys/odysseys.json \
RESULT_DIR=runs/odysseys200_qwen36_replan5 \
MANAGER_MODEL=fireworks_ai/deployed/t5cvm6h4 \
CUA_MODEL=fireworks_ai/deployed/t5cvm6h4 \
CUA_PROVIDER=qwen \
  bash scripts/lib/run_odysseys.sh

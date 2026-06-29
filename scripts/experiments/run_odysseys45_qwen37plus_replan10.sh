#!/usr/bin/env bash
# MACU replan=10, qwen3.7-plus as BOTH manager and CUA, over the 45-task odysseys subset.
cd "$(dirname "$0")/../.."
NUM_WORKERS="${NUM_WORKERS:-4}" \
MAX_PARALLELISM=4 \
MAX_STEPS=60 \
MAX_REPLANS=10 \
TASKS_FILE=data/odysseys/odysseys45.json \
RESULT_DIR=runs/odysseys45_qwen37plus_replan10 \
MANAGER_MODEL=fireworks_ai/qwen3p7-plus \
CUA_MODEL=fireworks_ai/qwen3p7-plus \
CUA_PROVIDER=qwen \
  bash scripts/lib/run_odysseys.sh

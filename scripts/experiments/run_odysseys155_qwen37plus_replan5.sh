#!/usr/bin/env bash
# MACU replan5, qwen3.7-plus as BOTH manager and CUA, over the 155-task odysseys complement.
# Matches the analyzed odysseys45 replan5 config (max_steps=60, 4 subagents).
cd "$(dirname "$0")/../.."
NUM_WORKERS="${NUM_WORKERS:-6}" \
MAX_PARALLELISM=4 \
MAX_STEPS=60 \
MAX_REPLANS=5 \
TASKS_FILE=data/odysseys/odysseys155.json \
RESULT_DIR=runs/odysseys155_qwen37plus_replan5 \
MANAGER_MODEL=fireworks_ai/qwen3p7-plus \
CUA_MODEL=fireworks_ai/qwen3p7-plus \
CUA_PROVIDER=qwen \
  bash scripts/lib/run_odysseys.sh

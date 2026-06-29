#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
export TASKS_FILE=data/odysseys/odysseys45.json
export RESULT_DIR=runs/odysseys45_opus48mgr_qwen37plus_replan5
export MANAGER_MODEL=claude-opus-4-8
export CUA_MODEL=fireworks_ai/qwen3p7-plus
export CUA_PROVIDER=qwen
export MAX_STEPS=60
export MAX_REPLANS=5
export NUM_WORKERS=6
export MAX_PARALLELISM=4
exec bash scripts/lib/run_odysseys.sh

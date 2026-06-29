#!/usr/bin/env bash
export TASKS_FILE=/home/ec2-user/macu/OSWorld/evaluation_examples/test_small.json
export RESULT_DIR=runs/test_small_qwen36_replan1
export MANAGER_MODEL=fireworks_ai/deployed/t5cvm6h4
export CUA_MODEL=fireworks_ai/deployed/t5cvm6h4
export CUA_PROVIDER=qwen
export MAX_STEPS=60
export MAX_REPLANS=1
export NUM_WORKERS=6
export MAX_PARALLELISM=4
exec bash scripts/lib/run_osworld.sh

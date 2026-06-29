#!/usr/bin/env bash
export TASKS_FILE=/home/ec2-user/macu/OSWorld/evaluation_examples/test_small.json
export RESULT_DIR=runs/test_small_qwen3vl_nomanager
export MANAGER_MODEL=fireworks_ai/qwen3p7-plus
export CUA_MODEL=bedrock/qwen.qwen3-vl-235b-a22b
export CUA_PROVIDER=qwen
export MAX_STEPS=100
export MAX_REPLANS=0
export EXTRA_ARGS=--no-manager
export NUM_WORKERS=6
export MAX_PARALLELISM=4
exec bash scripts/lib/run_osworld.sh

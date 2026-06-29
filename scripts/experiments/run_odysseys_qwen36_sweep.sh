#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
export MODEL_ID=fireworks_ai/deployed/t5cvm6h4
export LABEL=qwen36
exec bash scripts/lib/run_odysseys_model_sweep.sh

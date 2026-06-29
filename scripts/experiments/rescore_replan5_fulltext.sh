#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
set -a; source .env 2>/dev/null; set +a
D=runs/odysseys45_qwen37plus_replan5
.venv/bin/python evals/odysseys_eval.py \
  --runs-dir "$D" --task-source-json data/odysseys/odysseys.json \
  --output "$D/odysseys_eval_fulltext.json" \
  --num-workers 4 --max-concurrent-rubrics 2 --max-steps 0 --max-images 200

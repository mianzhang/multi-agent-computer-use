#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
set -a; source .env 2>/dev/null; set +a
D="$1"
.venv/bin/python evals/odysseys_eval.py \
  --runs-dir "$D" --task-source-json data/odysseys/odysseys.json \
  --output "$D/odysseys_eval_oneshot.json" \
  --oneshot --num-workers 6 --max-steps 0 --max-images 200

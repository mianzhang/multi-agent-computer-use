#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
set -a; source .env 2>/dev/null; set +a
cfg="$1"
[ "$cfg" = "nomanager" ] && D=runs/odysseys45_qwen37plus_nomanager || D=runs/odysseys45_qwen37plus_$cfg
.venv/bin/python evals/odysseys_eval.py \
  --runs-dir "$D" --task-source-json data/odysseys/odysseys.json \
  --output "$D/odysseys_eval_oneshot.json" \
  --oneshot --num-workers 6 --max-steps 0 --max-images 200

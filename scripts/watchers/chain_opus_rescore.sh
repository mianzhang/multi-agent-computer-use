#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
# wait for the qwen replan0/1 re-scores to free the proxy
while tmux has-session -t rescore0 2>/dev/null || tmux has-session -t rescore1 2>/dev/null; do sleep 30; done
tmux new-session -d -s rescoreopus "bash scripts/experiments/rescore_opus48_fulltext.sh > logs/rescore_opus48_fulltext.log 2>&1"

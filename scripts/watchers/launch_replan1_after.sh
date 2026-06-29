#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
while tmux has-session -t rescorenm 2>/dev/null; do sleep 30; done
tmux new-session -d -s rescore1 "bash scripts/experiments/rescore_replan1_fulltext.sh > logs/rescore_replan1_fulltext.log 2>&1"

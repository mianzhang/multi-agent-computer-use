#!/usr/bin/env bash
# Non-invasive disk-leak guard for apptainer runs.
#
# The framework deletes overlay .qcow2 files at task teardown but leaves their
# ~4GB .vmstate snapshots behind, so disk fills over a batch run. This reaper
# deletes ONLY orphan .vmstate files whose .qcow2 overlay is already gone.
# That is provably safe: if the overlay no longer exists, nothing can boot or
# inherit from it, so its vmstate is pure garbage. It never touches a .vmstate
# whose overlay is still alive (i.e. still inheritable).
#
# Usage: scripts/vmstate_reaper.sh [overlay_dir] [interval_seconds]
set -u
OVERLAY_DIR="${1:-apptainer/overlays}"
INTERVAL="${2:-30}"

echo "[reaper] watching $OVERLAY_DIR every ${INTERVAL}s (deletes orphan .vmstate/.conn.json)"
while true; do
  freed=0
  shopt -s nullglob
  for vs in "$OVERLAY_DIR"/*.vmstate "$OVERLAY_DIR"/*.conn.json; do
    overlay="${vs%.vmstate}"; overlay="${overlay%.conn.json}"
    # delete the sidecar only if its overlay qcow2 is gone
    if [ ! -e "$overlay" ]; then
      sz=$(du -m "$vs" 2>/dev/null | cut -f1)
      rm -f "$vs" 2>/dev/null && freed=$((freed + ${sz:-0}))
    fi
  done
  if [ "$freed" -gt 0 ]; then
    echo "[reaper $(date +%H:%M:%S)] reclaimed ${freed}MB of orphan sidecars; disk now $(df -h / | awk 'NR==2{print $5}')"
  fi
  sleep "$INTERVAL"
done

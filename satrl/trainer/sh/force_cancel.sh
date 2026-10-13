#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SATRL_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_ROOT="$(cd -- "$SATRL_ROOT/.." && pwd)"
PROJECT="${PROJECT:-$ARTIFACT_ROOT}"
STOP_DIR="${STOP_DIR:-$ARTIFACT_ROOT/output/requeue_state}"
mkdir -p "$STOP_DIR"
touch "$STOP_DIR/STOP_RESUBMIT"

# Cancel only jobs with "_l40s" in the job name
squeue -u "$USER" -h -o "%A %j" \
  | awk '$2 ~ /Qwen3.5/ {print $1}' \
  | xargs -r scancel

sleep 30
rm "$STOP_DIR/STOP_RESUBMIT"

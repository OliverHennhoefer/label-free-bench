#!/usr/bin/env bash
# Run N research sessions locally, each a fresh Claude Code session.
set -euo pipefail
cd "$(dirname "$0")/.."
for _ in $(seq "${1:-1}"); do
  if python -m benchmarks next | grep -q '^role: halt'; then
    python -m benchmarks next
    exit 0
  fi
  LAB_SESSION="$(python -c 'import uuid; print(uuid.uuid4())')" \
    claude -p "$(cat scripts/wake-prompt.md)"
done

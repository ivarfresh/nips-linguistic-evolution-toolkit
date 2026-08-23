#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export LLM_PROVIDER=direct
export GEMINI_THINKING_LEVEL=medium
export TRUST_BATCH_QUIET=1
export PYTHONPYCACHEPREFIX=/tmp/nlet-pycache
export PYTHONPATH=.

OUTPUT_SUBDIR="gemini37_flash_task_order_n3_20260823"
LOG_ROOT="/tmp/nlet-gemini37-task-order-n3"

for EXPERIMENT in \
  noise8_crossmodel_gemini37_flash_n3_game \
  noise8_crossmodel_gemini37_flash_n3_game_myth \
  noise8_crossmodel_gemini37_flash_n3_myth_game
do
  python3 scripts/run_noisy_missing.py \
    "$EXPERIMENT" \
    --workers 3 \
    --output-subdir "$OUTPUT_SUBDIR" \
    --log-dir "$LOG_ROOT"
done

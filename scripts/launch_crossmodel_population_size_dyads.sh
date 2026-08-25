#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

FAMILY="${1:-all}"
OUTPUT_SUBDIR="crossmodel_population_size_dyads_20260824"
LOG_ROOT="/tmp/nlet-crossmodel-population-size-dyads"

export LLM_PROVIDER=direct
export TRUST_BATCH_QUIET=1
export PYTHONPYCACHEPREFIX=/tmp/nlet-pycache
export PYTHONPATH=.

run_gpt() {
  unset OPENAI_REASONING_EFFORT || true
  for EXPERIMENT in \
    noise2_crossmodel_signed_gpt_n5_game \
    noise2_crossmodel_signed_gpt_n5_game_myth \
    noise2_crossmodel_signed_gpt_n5_myth_game
  do
    python3 scripts/run_noisy_missing.py \
      "$EXPERIMENT" \
      --workers 3 \
      --output-subdir "$OUTPUT_SUBDIR" \
      --log-dir "$LOG_ROOT/gpt"
  done
}

run_gemini() {
  export GEMINI_THINKING_LEVEL=medium
  export GEMINI_REQUEST_TIMEOUT_SECONDS=300
  for EXPERIMENT in \
    noise2_crossmodel_gemini37_flash_n3_game \
    noise2_crossmodel_gemini37_flash_n3_game_myth \
    noise2_crossmodel_gemini37_flash_n3_myth_game
  do
    python3 scripts/run_noisy_missing.py \
      "$EXPERIMENT" \
      --workers 3 \
      --output-subdir "$OUTPUT_SUBDIR" \
      --log-dir "$LOG_ROOT/gemini"
  done
}

case "$FAMILY" in
  gpt)
    run_gpt
    ;;
  gemini)
    run_gemini
    ;;
  all)
    run_gpt
    run_gemini
    ;;
  *)
    echo "Usage: $0 [gpt|gemini|all]" >&2
    exit 2
    ;;
esac

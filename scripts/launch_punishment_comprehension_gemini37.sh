#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export LLM_PROVIDER=direct
export GEMINI_THINKING_LEVEL=medium
export GEMINI_REQUEST_TIMEOUT_SECONDS=300
export PYTHONPYCACHEPREFIX=/tmp/nlet-pycache
export PYTHONPATH=.

python3 scripts/run_punishment_comprehension_calibration.py \
  --output data/json/noise_experiments/punishment_comprehension_gemini37_20260823/results.json \
  --workers 5 \
  --model google/gemini-3.7-flash \
  --variants current \
  --trials-per-cell 10 \
  --order-seed 202608237 \
  --return-ratios 0 .10 .25 .50 .75 \
  --protocol docs/punishment_comprehension_gemini37_protocol_2026-08-23.md

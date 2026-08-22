#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export TRUST_BATCH_QUIET=1
export PYTHONPYCACHEPREFIX=/tmp/nlet-pycache
export PYTHONPATH=.

exec python3 scripts/run_noisy_missing.py \
  noise8i_defector_punishment_gemini_availability_off_matched_n10 \
  --workers 5 \
  --output-subdir defector_punishment_gemini_availability_matched_n10_20260822 \
  --log-dir /tmp/nlet-gemini-punishment-availability

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export LLM_PROVIDER=direct
export GEMINI_THINKING_LEVEL=medium
export GEMINI_REQUEST_TIMEOUT_SECONDS=300
export TRUST_BATCH_QUIET=1
export PYTHONPYCACHEPREFIX=/tmp/nlet-pycache
export PYTHONPATH=.

python3 scripts/run_noisy_missing.py \
  noise8i_defector_punishment_gemini37_n3 \
  --workers 3 \
  --output-subdir defector_punishment_gemini37_n3_20260823 \
  --log-dir /tmp/nlet-gemini37-defector-punishment-n3

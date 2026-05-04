#!/usr/bin/env bash
# Phase 6 — A1 + targeted (cooperative) myth × bootstrap noise.
# Tests whether targeted myth content amplifies the A1 bootstrap-rescue
# effect already observed in Phase 1.
#
# Waits for Phase 5 to finish (master log shows "PHASE 5 exit:"), then runs.

set -uo pipefail
cd "$(dirname "$0")/../../../.."

LOG_MASTER="${LOG_MASTER:-/tmp/overnight_master.log}"
LOG_OUT="/tmp/overnight_phase6.log"

echo "[$(date '+%H:%M:%S')] Phase 6 waiter armed; watching $LOG_MASTER for 'PHASE 5 exit:'"

# Wait for Phase 5 exit line to appear
until grep -q "PHASE 5 exit:" "$LOG_MASTER" 2>/dev/null; do
  sleep 30
done

echo "[$(date '+%H:%M:%S')] Phase 5 done — launching Phase 6"

env PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct \
  /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py \
  gpt5nano_partner_myth_targeted_bootstrap \
  --workers 2 \
  --output-subdir v4_direct_provider_A1_targeted_bootstrap \
  > "$LOG_OUT" 2>&1
EXIT=$?

echo "[$(date '+%H:%M:%S')] Phase 6 exit: $EXIT (log: $LOG_OUT)"
echo "  Output: data/json/noise_experiments/v4_direct_provider_A1_targeted_bootstrap/"

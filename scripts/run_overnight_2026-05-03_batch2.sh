#!/usr/bin/env bash
# Batch 2 — Adversarial / targeted myth × non-bootstrap noise.
# Chains automatically from batch 1. ~5 sets × 60 runs + 1 × 30 = 330 runs.
set -uo pipefail

cd "$(dirname "$0")/.."

LOG_DIR=/tmp/nlet-runs/overnight_2026-05-03
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/_overnight_summary.log"

run_set() {
  local name="$1"
  local subdir="$2"
  echo "[$(date +%H:%M:%S)] B2 launching $name → $subdir" | tee -a "$SUMMARY"
  env PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct \
    /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py \
    "$name" --workers 4 --output-subdir "$subdir" --log-dir "$LOG_DIR" \
    >> "$LOG_DIR/$name.out" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] B2 finished $name (rc=$rc)" | tee -a "$SUMMARY"
}

run_set gpt5nano_partner_myth_adversarial_positive_5  v4_direct_provider_controls
run_set gpt5nano_partner_myth_adversarial_negative_5  v4_direct_provider_controls
run_set gpt5nano_partner_myth_adversarial_no_noise    v4_direct_provider_controls
run_set gpt5nano_partner_myth_targeted_positive_5     v4_direct_provider_controls
run_set gpt5nano_partner_myth_targeted_negative_5     v4_direct_provider_controls

echo "[$(date +%H:%M:%S)] ALL BATCHES FINISHED" | tee -a "$SUMMARY"

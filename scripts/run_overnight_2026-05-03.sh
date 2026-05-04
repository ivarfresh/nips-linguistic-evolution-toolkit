#!/usr/bin/env bash
# Overnight launcher 2026-05-03 — completes the control × noise matrix.
# Tier 1: negative_5 × {C1, C2, C3} (180 runs)
# Tier 2: positive_5 × {C1, C3} (120 runs; C2 already done)
# Tier 3: no_noise × {C1, C2, C3} (90 runs)
# Total: 390 runs at ~30-60s each → 2-4 hours with --workers 4.
set -uo pipefail

cd "$(dirname "$0")/.."

LOG_DIR=/tmp/nlet-runs/overnight_2026-05-03
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/_overnight_summary.log"
: > "$SUMMARY"

run_set() {
  local name="$1"
  local subdir="$2"
  echo "[$(date +%H:%M:%S)] launching $name → $subdir" | tee -a "$SUMMARY"
  env PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct \
    /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py \
    "$name" --workers 4 --output-subdir "$subdir" --log-dir "$LOG_DIR" \
    >> "$LOG_DIR/$name.out" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] finished $name (rc=$rc)" | tee -a "$SUMMARY"
}

# Tier 1 — negative_5 controls (most informative)
run_set gpt5nano_partner_myth_filler_negative_5    v4_direct_provider_controls
run_set gpt5nano_partner_myth_shuffled_negative_5  v4_direct_provider_controls
run_set gpt5nano_partner_myth_own_negative_5       v4_direct_provider_controls

# Tier 2 — positive_5 controls (boundary)
run_set gpt5nano_partner_myth_shuffled_positive_5  v4_direct_provider_controls
run_set gpt5nano_partner_myth_own_positive_5       v4_direct_provider_controls

# Tier 3 — no_noise controls (rules out general lift)
run_set gpt5nano_partner_myth_filler_no_noise      v4_direct_provider_controls
run_set gpt5nano_partner_myth_shuffled_no_noise    v4_direct_provider_controls
run_set gpt5nano_partner_myth_own_no_noise         v4_direct_provider_controls

# Backfill 1 missing run from earlier filler bootstrap batch
run_set gpt5nano_partner_myth_filler_bootstrap     v4_direct_provider_controls

echo "[$(date +%H:%M:%S)] BATCH 1 FINISHED — chaining batch 2" | tee -a "$SUMMARY"

# Batch 2 chains here so it runs without manual intervention
if [ -x scripts/run_overnight_2026-05-03_batch2.sh ]; then
  exec bash scripts/run_overnight_2026-05-03_batch2.sh
fi
echo "[$(date +%H:%M:%S)] ALL SETS FINISHED" | tee -a "$SUMMARY"

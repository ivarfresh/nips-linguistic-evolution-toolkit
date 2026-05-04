#!/usr/bin/env bash
# Overnight experiment queue, prioritized by (information value) / (cost).
#
# Total ~430 runs at 2 workers ≈ 6–9 hours wall time depending on rate
# limits and per-run latency. Logs to /tmp/overnight_<exp>.log so you
# can tail in the morning.
#
# Each phase exits independently — if one fails, the next still runs.
#
# Usage (kick off before bed):
#   nohup ./projects/neurips-2026-llm-ling-evo/analysis/A1_A3_package/overnight_queue.sh \
#         > /tmp/overnight_master.log 2>&1 &
#
# Then in the morning:
#   tail -50 /tmp/overnight_master.log
#   ls -la data/json/noise_experiments/v4_direct_provider_A*

set -uo pipefail   # NOT -e — we want to keep going if one phase fails

cd "$(dirname "$0")/../../../.."   # navigate to repo root
echo "Repo root: $(pwd)"

WORKERS="${WORKERS:-2}"
LOGDIR="${LOGDIR:-/tmp}"
START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "=== overnight_queue START: $START_TIME ==="
echo "WORKERS=$WORKERS LOGDIR=$LOGDIR"
echo

# ─────────────────────────────────────────────────────────────────────
# PHASE 1 — A1 partner-myth injection (the headline test for the
# linguistic-coupling reframe). 2 task orders × 4 noise conditions × 15
# seeds = ~120 runs. Highest-information-per-run experiment we have.
# ─────────────────────────────────────────────────────────────────────
echo "=== PHASE 1: A1 partner-myth injection ($(date +%H:%M)) ==="
env PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct \
  /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py \
  gpt5nano_partner_myth_injection \
  --workers "$WORKERS" \
  --output-subdir v4_direct_provider_A1_partner_myth \
  > "$LOGDIR/overnight_A1.log" 2>&1
P1_EXIT=$?
echo "  PHASE 1 exit: $P1_EXIT  (log: $LOGDIR/overnight_A1.log)"
echo

# ─────────────────────────────────────────────────────────────────────
# PHASE 2 — A3 forced reasoning prose. 3 task orders × 4 noise × 10 =
# ~120 runs. Closes the §4.5 methodological blind spot.
# ─────────────────────────────────────────────────────────────────────
echo "=== PHASE 2: A3 forced reasoning prose ($(date +%H:%M)) ==="
env PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct \
  /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py \
  gpt5nano_forced_reasoning \
  --workers "$WORKERS" \
  --output-subdir v4_direct_provider_A3_forced_reasoning \
  > "$LOGDIR/overnight_A3.log" 2>&1
P2_EXIT=$?
echo "  PHASE 2 exit: $P2_EXIT  (log: $LOGDIR/overnight_A3.log)"
echo

# ─────────────────────────────────────────────────────────────────────
# PHASE 3 — Top up incomplete cells from today's targeted-myth runs
# (some at N=5–10 of 15). Pure cleanup; lower information per run but
# tightens the existing tables.
# Uses the same experiment_set names — run_noisy_missing.py will only
# fill in missing seeds.
# ─────────────────────────────────────────────────────────────────────
echo "=== PHASE 3: Top-up incomplete targeted-myth cells ($(date +%H:%M)) ==="
for exp in targeted_myth_neutral_gpt5nano targeted_myth_k1_gpt5nano targeted_myth_k2_gpt5nano; do
  echo "  -- $exp --"
  env PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct \
  /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py \
    "$exp" \
    --workers "$WORKERS" \
    --output-subdir "v4_direct_provider_${exp//targeted_myth_/targeted_}" \
    >> "$LOGDIR/overnight_topup.log" 2>&1
  echo "    exit: $?"
done
echo "  PHASE 3 log: $LOGDIR/overnight_topup.log"
echo

# ─────────────────────────────────────────────────────────────────────
# PHASE 4 — A1 + A3 combined (only if previous phases finished early).
# The cleanest test: open channel + visible reasoning prose. 80 runs.
# ─────────────────────────────────────────────────────────────────────
echo "=== PHASE 4: A1+A3 combined ($(date +%H:%M)) ==="
env PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct \
  /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py \
  gpt5nano_partner_myth_plus_reasoning \
  --workers "$WORKERS" \
  --output-subdir v4_direct_provider_A1A3_combined \
  > "$LOGDIR/overnight_A1A3_combined.log" 2>&1
P4_EXIT=$?
echo "  PHASE 4 exit: $P4_EXIT  (log: $LOGDIR/overnight_A1A3_combined.log)"
echo

# ─────────────────────────────────────────────────────────────────────
# PHASE 5 — Targeted myth × bootstrap noise, GPT-5-Nano. The §5.2
# mechanism-discriminating test: does cooperative-themed myth content
# rescue or worsen the bootstrap-noise destabilization? 3 task orders ×
# 2 noise variants × 15 seeds ≈ 90 runs.
# ─────────────────────────────────────────────────────────────────────
echo "=== PHASE 5: Targeted myth × bootstrap (mechanism test) ($(date +%H:%M)) ==="
env PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct \
  /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py \
  targeted_myth_bootstrap_gpt5nano \
  --workers "$WORKERS" \
  --output-subdir v4_direct_provider_targeted_bootstrap \
  > "$LOGDIR/overnight_targeted_bootstrap.log" 2>&1
P5_EXIT=$?
echo "  PHASE 5 exit: $P5_EXIT  (log: $LOGDIR/overnight_targeted_bootstrap.log)"
echo

END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "=== overnight_queue END: $END_TIME ==="
echo
echo "Summary:"
echo "  A1 partner-myth:      data/json/noise_experiments/v4_direct_provider_A1_partner_myth/"
echo "  A3 forced reasoning:  data/json/noise_experiments/v4_direct_provider_A3_forced_reasoning/"
echo "  A1+A3 combined:       data/json/noise_experiments/v4_direct_provider_A1A3_combined/"
echo "  targeted top-ups:     data/json/noise_experiments/v4_direct_provider_targeted_*/"
echo
echo "Next morning steps:"
echo "  1. Verify prompts: python3 projects/neurips-2026-llm-ling-evo/analysis/A1_A3_package/verify_prompts.py"
echo "  2. Re-run analyses: see analysis_hook.md in the same folder"
echo "  3. Re-render manuscript + overleaf bundle:"
echo "       python3 projects/neurips-2026-llm-ling-evo/analysis/rebuild_overleaf.py"

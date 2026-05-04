#!/usr/bin/env bash
# GPT-5-Nano prompt-variant launcher for the NeurIPS sprint.
# Runs only missing final JSONs via scripts/run_noisy_missing.py.
set -uo pipefail

cd "$(dirname "$0")/.."

LOG_DIR=${LOG_DIR:-/tmp/nlet-runs/gpt5nano-prompt-variants-20260504}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-v4_direct_provider_prompt_variants}
WORKERS=${WORKERS:-4}
mkdir -p "$LOG_DIR"

SUMMARY="$LOG_DIR/_summary.log"
MARKER="$LOG_DIR/_start.marker"
: > "$SUMMARY"
touch "$MARKER"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

final_count() {
  local name="$1"
  local root="data/json/noise_experiments/$OUTPUT_SUBDIR/$name"
  if [ ! -d "$root" ]; then
    echo 0
    return
  fi
  find "$root" -type f -name "*.json" \
    ! -name "*results*" \
    ! -name "*checkpoint*" \
    ! -name "*.error.json" | wc -l | awk '{print $1}'
}

new_final_count() {
  local name="$1"
  local root="data/json/noise_experiments/$OUTPUT_SUBDIR/$name"
  if [ ! -d "$root" ]; then
    echo 0
    return
  fi
  find "$root" -type f -name "*.json" \
    ! -name "*results*" \
    ! -name "*checkpoint*" \
    ! -name "*.error.json" \
    -newer "$MARKER" | wc -l | awk '{print $1}'
}

run_set() {
  local name="$1"
  local failed="$LOG_DIR/$name/failed.json"
  mkdir -p "$LOG_DIR/$name"
  rm -f "$failed"

  local before
  before=$(final_count "$name")
  echo "[$(timestamp)] launching $name -> $OUTPUT_SUBDIR (finals_before=$before)" | tee -a "$SUMMARY"

  env PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct \
    /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py \
    "$name" --workers "$WORKERS" --output-subdir "$OUTPUT_SUBDIR" --log-dir "$LOG_DIR" \
    >> "$LOG_DIR/$name.out" 2>&1
  rc=$?

  local after new
  after=$(final_count "$name")
  new=$(new_final_count "$name")
  echo "[$(timestamp)] finished $name (rc=$rc finals_after=$after new_since_start=$new)" | tee -a "$SUMMARY"
}

echo "[$(timestamp)] START GPT-5-Nano prompt variants; marker=$MARKER" | tee -a "$SUMMARY"

# Priority 1: lower-risk prompt-only variant.
run_set gpt5nano_prompt_unconstrained_bootstrap

# Priority 2: myth-first blind system-prompt path.
run_set gpt5nano_myth_first_blind_bootstrap

# Priority 3: combined blind + unconstrained prompt.
run_set gpt5nano_myth_first_blind_unconstrained_bootstrap

echo "[$(timestamp)] ALL DONE" | tee -a "$SUMMARY"

#!/usr/bin/env bash
# Overnight launcher 2026-05-03 - Ivar missing-condition model sweep, no Claude.
# Runs only missing final JSONs via scripts/run_noisy_missing.py.
set -uo pipefail

cd "$(dirname "$0")/.."

LOG_DIR=${LOG_DIR:-/tmp/nlet-runs/missing-conditions}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-v4_direct_provider}
mkdir -p "$LOG_DIR"

SUMMARY="$LOG_DIR/_summary.log"
MARKER="$LOG_DIR/_start.marker"
STOP_FLAG="$LOG_DIR/_stop_due_openai_quota"

if [ -f "$STOP_FLAG" ]; then
  echo "[$(date "+%Y-%m-%d %H:%M:%S")] STOP flag present at $STOP_FLAG; exiting without launching sets" >> "$SUMMARY"
  exit 0
fi

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

  env PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct OPENAI_REASONING_EFFORT=low \
    /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py \
    "$name" --workers 4 --output-subdir "$OUTPUT_SUBDIR" --log-dir "$LOG_DIR" \
    >> "$LOG_DIR/$name.out" 2>&1
  rc=$?

  local after new
  after=$(final_count "$name")
  new=$(new_final_count "$name")
  echo "[$(timestamp)] finished $name (rc=$rc finals_after=$after new_since_start=$new)" | tee -a "$SUMMARY"
}

echo "[$(timestamp)] START missing-condition model sweep; marker=$MARKER" | tee -a "$SUMMARY"

# Gemini first: resumes the existing partial positive set, then fills the two
# missing noise families.
run_set noise_positive_mem3_gemini_3_1_pro
run_set noise_negative_mem3_gemini_3_1_pro
run_set noise_bootstrap_mem3_gemini_3_1_pro

# GPT-5.5 full v4-main sweep.
run_set noise_positive_mem3_gpt5_5
run_set noise_negative_mem3_gpt5_5
run_set noise_bootstrap_mem3_gpt5_5

echo "[$(timestamp)] ALL DONE" | tee -a "$SUMMARY"

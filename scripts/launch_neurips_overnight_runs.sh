#!/usr/bin/env bash
set -euo pipefail

# Resumable overnight runner for the NeurIPS rough-draft experiment queue.
# Defaults run only the mandatory queue: matched baseline + neutral framing pilot.
# Set RUN_K1=1 to append the 360-job k=1 sensitivity run after those complete.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ "${USE_CAFFEINATE:-1}" = "1" ] && command -v caffeinate >/dev/null 2>&1 && [ "${NLET_CAFFEINATED:-0}" != "1" ]; then
  export NLET_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi

export PYENV_VERSION="${PYENV_VERSION:-3.10.14}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/nlet-pycache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/nlet-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/nlet-cache}"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Force direct provider routing for this queue, even if .env contains an older
# default such as LLM_PROVIDER=auto.
export LLM_PROVIDER="${NLET_LLM_PROVIDER:-direct}"

run_python() {
  if [ -x /opt/homebrew/bin/pyenv ]; then
    /opt/homebrew/bin/pyenv exec python "$@"
  else
    python3 "$@"
  fi
}

WORKERS="${WORKERS:-2}"
LOG_DIR="${LOG_DIR:-/tmp/nlet-runs}"
RUN_K1="${RUN_K1:-0}"
RUN_FULL_NEUTRAL="${RUN_FULL_NEUTRAL:-0}"
RENDER_HTML="${RENDER_HTML:-1}"
RENDER_PDF="${RENDER_PDF:-0}"

run_status() {
  run_python scripts/noisy_run_status.py "$1" --output-subdir "$2"
}

run_missing() {
  local experiment="$1"
  local subdir="$2"
  echo
  echo "=============================================================================="
  echo "Status before: ${experiment} -> ${subdir}"
  echo "=============================================================================="
  run_status "$experiment" "$subdir"

  echo
  echo "Running missing jobs: ${experiment}"
  run_python scripts/run_noisy_missing.py "$experiment" \
    --workers "$WORKERS" \
    --output-subdir "$subdir" \
    --log-dir "$LOG_DIR"

  echo
  echo "Status after: ${experiment} -> ${subdir}"
  run_status "$experiment" "$subdir"
}

echo "NeurIPS overnight run queue"
echo "Project: $PROJECT_ROOT"
echo "Workers: $WORKERS"
echo "Provider: $LLM_PROVIDER"
echo "Log dir: $LOG_DIR"
echo "RUN_K1=$RUN_K1 RUN_FULL_NEUTRAL=$RUN_FULL_NEUTRAL"

run_missing baseline_v4_mem3_direct v4_direct_provider_baseline
run_missing neutral_framing_v4_pilot v4_direct_provider_neutral

if [ "$RUN_FULL_NEUTRAL" = "1" ]; then
  run_missing neutral_framing_v4_mem3 v4_direct_provider_neutral
fi

if [ "$RUN_K1" = "1" ]; then
  run_missing noise_directional_k1_mem3 v4_direct_provider_k1
fi

echo
echo "Regenerating draft artifacts..."
run_python scripts/build_neurips_draft_artifacts.py

if [ "$RENDER_HTML" = "1" ]; then
  echo
  echo "Rendering manuscript HTML..."
  (cd projects/neurips-2026-llm-ling-evo/manuscript && HOME=/tmp quarto render index.qmd --to html --no-cache)
fi

if [ "$RENDER_PDF" = "1" ]; then
  echo
  echo "Rendering manuscript PDF..."
  (cd projects/neurips-2026-llm-ling-evo/manuscript && quarto render index.qmd --to pdf)
fi

echo
echo "Overnight queue complete."

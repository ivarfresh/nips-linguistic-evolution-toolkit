#!/usr/bin/env bash
# Runner for A1 (partner-myth injection) and A3 (forced reasoning prose)
# experiments. GPT-5-Nano only.
#
# Prerequisites (apply once before running):
#   1. Apply patch:
#        patch -p1 < projects/neurips-2026-llm-ling-evo/analysis/A1_A3_package/trust_game_noisy.partner_myth.patch
#
#   2. Append YAML:
#        cat projects/neurips-2026-llm-ling-evo/analysis/A1_A3_package/experiments_noisy.append.yaml \
#            >> config/experiments_noisy.yaml
#        (then manually move sections into the right parent — see file comments)
#
# Then run from repo root:
#   ./projects/neurips-2026-llm-ling-evo/analysis/A1_A3_package/run_A1_A3.sh

set -euo pipefail

cd "$(dirname "$0")/../../../.."   # navigate to repo root

PYENV_PREFIX=(
  PYENV_VERSION=3.10.14
  PYTHONPYCACHEPREFIX=/tmp/nlet-pycache
  LLM_PROVIDER=direct
  /opt/homebrew/bin/pyenv exec python
)

WORKERS="${WORKERS:-2}"

echo "=== A3: Forced reasoning prose (~120 runs) ==="
"${PYENV_PREFIX[@]}" scripts/run_noisy_missing.py \
  gpt5nano_forced_reasoning \
  --workers "$WORKERS" \
  --output-subdir v4_direct_provider_A3_forced_reasoning

echo
echo "=== A1: Partner-myth injection (~120 runs) ==="
"${PYENV_PREFIX[@]}" scripts/run_noisy_missing.py \
  gpt5nano_partner_myth_injection \
  --workers "$WORKERS" \
  --output-subdir v4_direct_provider_A1_partner_myth

# Optional: combined A1+A3 (~80 runs).
# Uncomment if you want the cleanest "open channel + visible reasoning" test.
#
# echo
# echo "=== A1+A3 combined (~80 runs) ==="
# "${PYENV_PREFIX[@]}" scripts/run_noisy_missing.py \
#   gpt5nano_partner_myth_plus_reasoning \
#   --workers "$WORKERS" \
#   --output-subdir v4_direct_provider_A1A3_combined

echo
echo "Done. Outputs:"
echo "  data/json/noise_experiments/v4_direct_provider_A3_forced_reasoning/"
echo "  data/json/noise_experiments/v4_direct_provider_A1_partner_myth/"
echo
echo "Next: rerun analyses to fold in the new cells:"
echo "  python3 projects/neurips-2026-llm-ling-evo/analysis/build_cell_summary.py"
echo "  python3 projects/neurips-2026-llm-ling-evo/analysis/build_lag_and_lexicon.py"
echo "  python3 projects/neurips-2026-llm-ling-evo/analysis/build_reason_coding.py"
echo "  # ...etc"
echo "(Edit INCLUDE_VERSIONS in each build_*.py to include the new subdirs.)"

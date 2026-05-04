#!/usr/bin/env bash
# Sequential runner for #2 (A1 × no-noise) and #3 (A1 × adversarial × bootstrap).
# 90 runs total, ~45 min wall.

set -uo pipefail
cd "$(dirname "$0")/../../../.."

WORKERS="${WORKERS:-2}"
echo "=== Run 2+3 START: $(date '+%Y-%m-%d %H:%M:%S') ==="

# #2 — A1 × no-noise (30 runs)
echo
echo "=== #2: A1 × no-noise (gpt5nano_partner_myth_no_noise) ==="
env PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct \
  /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py \
  gpt5nano_partner_myth_no_noise \
  --workers "$WORKERS" \
  --output-subdir v4_direct_provider_A1_no_noise \
  > /tmp/run2_no_noise.log 2>&1
echo "  exit: $?  (log: /tmp/run2_no_noise.log)"

# #3 — A1 × adversarial × bootstrap (60 runs)
echo
echo "=== #3: A1 × adversarial-myth × bootstrap (gpt5nano_partner_myth_adversarial_bootstrap) ==="
env PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct \
  /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py \
  gpt5nano_partner_myth_adversarial_bootstrap \
  --workers "$WORKERS" \
  --output-subdir v4_direct_provider_A1_adversarial_bootstrap \
  > /tmp/run3_adversarial.log 2>&1
echo "  exit: $?  (log: /tmp/run3_adversarial.log)"

echo
echo "=== Run 2+3 END: $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "Outputs:"
echo "  data/json/noise_experiments/v4_direct_provider_A1_no_noise/"
echo "  data/json/noise_experiments/v4_direct_provider_A1_adversarial_bootstrap/"

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to launch confirmatory runs from a dirty worktree." >&2
  exit 1
fi

output_subdir="${1:-corrected_v2_confirmatory_20260812}"
workers="${WORKERS:-2}"
export LLM_PROVIDER="anthropic"
export ANTHROPIC_MAX_TOKENS="4096"
export TRUST_BATCH_QUIET="1"

experiment_sets=(
  noise2i_memprimary_v2_game
  noise2i_memprimary_v2_game_myth
  noise2i_memprimary_v2_myth_game
  noise8i_memprimary_v2_game
  noise8i_memprimary_v2_game_myth
  noise8i_memprimary_v2_myth_game
)

for experiment_set in "${experiment_sets[@]}"; do
  python3 experiments/run_noisy_batch.py \
    "$experiment_set" \
    --workers "$workers" \
    --output-subdir "$output_subdir"

  experiment_dir="data/json/noise_experiments/$output_subdir/$experiment_set"
  completed_runs="$({
    find "$experiment_dir" -type f -name '*.json' \
      ! -name '*.results.json' \
      ! -name '*.checkpoint.json' \
      ! -name '*.error.json'
  } | wc -l | tr -d ' ')"
  if [[ "$completed_runs" != "10" ]]; then
    echo "$experiment_set produced $completed_runs final runs; expected 10." >&2
    exit 1
  fi
  python3 scripts/audit_v2_protocol.py "$experiment_dir"
done

python3 scripts/audit_v2_protocol.py \
  "data/json/noise_experiments/$output_subdir"

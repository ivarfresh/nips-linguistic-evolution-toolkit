#!/usr/bin/env bash
# Morning-after status check. Quick visual on what landed overnight.
#
# Usage:
#   ./projects/neurips-2026-llm-ling-evo/analysis/A1_A3_package/morning_check.sh

set -uo pipefail

cd "$(dirname "$0")/../../../.."

echo "=== Overnight queue master log (last 30 lines) ==="
tail -30 /tmp/overnight_master.log 2>/dev/null || echo "  (no master log)"
echo

echo "=== Phase exit codes ==="
for phase in A1 A3 topup A1A3_combined targeted_bootstrap; do
  log="/tmp/overnight_${phase}.log"
  if [ -f "$log" ]; then
    last=$(tail -5 "$log" | grep -E "Done|Error|exit" | tail -1)
    echo "  $phase: $last"
  fi
done
echo

echo "=== Run counts per new experiment dir ==="
for d in v4_direct_provider_A1_partner_myth v4_direct_provider_A3_forced_reasoning v4_direct_provider_A1A3_combined; do
  full="data/json/noise_experiments/$d"
  if [ -d "$full" ]; then
    n=$(find "$full" -name "*.json" ! -name "*.results.json" ! -name "*.checkpoint*" ! -name "*.error*" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $d: $n runs"
  else
    echo "  $d: not present"
  fi
done
echo

echo "=== Today's targeted-myth top-up status (incomplete cells, target=15) ==="
for d in v4_direct_provider_targeted_gpt5nano v4_direct_provider_targeted_k1_gpt5nano v4_direct_provider_targeted_k2_gpt5nano; do
  for sub in data/json/noise_experiments/$d/*/*/*/*/; do
    [ -d "$sub" ] || continue
    n=$(find "$sub" -maxdepth 1 -name "*.json" ! -name "*.results.json" ! -name "*.checkpoint*" ! -name "*.error*" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -lt 15 ]; then
      echo "  $sub: $n / 15"
    fi
  done
done
echo

echo "=== Errored runs (.error.json files) ==="
err_count=$(find data/json/noise_experiments -name "*.error.json" -newermt "$(date -v-1d +%Y-%m-%d)" 2>/dev/null | wc -l | tr -d ' ')
echo "  $err_count error files since yesterday"
if [ "$err_count" -gt 0 ]; then
  find data/json/noise_experiments -name "*.error.json" -newermt "$(date -v-1d +%Y-%m-%d)" 2>/dev/null | head -5
fi
echo

echo "=== Quick smoke test on first A1 run ==="
A1_RUN=$(find data/json/noise_experiments/v4_direct_provider_A1_partner_myth -name "*.json" ! -name "*.results.json" ! -name "*.checkpoint*" ! -name "*.error*" 2>/dev/null | head -1)
if [ -n "$A1_RUN" ]; then
  echo "  Verifying $A1_RUN ..."
  python3 projects/neurips-2026-llm-ling-evo/analysis/A1_A3_package/verify_prompts.py "$A1_RUN" 2>&1 | tail -10
else
  echo "  No A1 runs found yet."
fi
echo

echo "=== Suggested next steps ==="
echo "  1. If A1/A3 produced results — run analyses with new INCLUDE_VERSIONS:"
echo "     edit build_*.py and add the new subdirs (see analysis_hook.md)"
echo "  2. Then regenerate headline tables + figures"
echo "  3. Then python3 projects/neurips-2026-llm-ling-evo/analysis/rebuild_overleaf.py"

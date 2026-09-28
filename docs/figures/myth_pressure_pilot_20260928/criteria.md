# Myth-pressure pilot: criteria

Runs: 20; myths: 400; council messages: 360

| arm             | resources     | mean_send    | late_ppl         | late_delivered   | fit_rate_r4plus   | planned_rate_r4plus   | shared_invented   | council_flag_rate   | cost         |
|:----------------|:--------------|:-------------|:-----------------|:-----------------|:------------------|:----------------------|:------------------|:--------------------|:-------------|
| loose_nocouncil | 57.30 (±5.45) | 0.65 (±0.11) | 46.79 (±16.82)   | 200.00 (±0.00)   | —                 | —                     | 1.00 (±0.71)      | —                   | 0.89 (±0.07) |
| loose_council   | 57.76 (±9.71) | 0.66 (±0.19) | 57.26 (±11.80)   | 200.00 (±0.00)   | —                 | —                     | 1.20 (±1.30)      | 0.03 (±0.06)        | 2.06 (±0.08) |
| tight_nocouncil | 59.76 (±4.36) | 0.70 (±0.09) | 453.44 (±114.00) | 20.00 (±0.00)    | 0.00 (±0.00)      | 0.20 (±0.12)          | 0.00 (±0.00)      | —                   | 0.75 (±0.03) |
| tight_council   | 61.14 (±8.25) | 0.72 (±0.17) | 382.90 (±72.52)  | 19.94 (±0.13)    | 0.09 (±0.12)      | 0.17 (±0.16)          | 0.00 (±0.00)      | 0.17 (±0.27)        | 1.71 (±0.13) |

1. Tight arms, rounds 4+: fit before truncation 4%; planned for the cut 19% (criterion: planned ≥ 80%) → FAIL
2a. Perplexity ratio tight/council ÷ tight/no-council, rounds 6–10: 0.84 (criterion ≥ 2) → FAIL
2b. Shared invented tokens, tight/council: 0.00 (±0.00); runs with ≥ 3: 0/5
3. Council messages with amount/strategy talk: 10% (criterion ≤ 25%) → PASS

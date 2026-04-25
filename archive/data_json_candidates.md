# data/json/ archive candidates

Survey only — nothing has been moved. Confirm per row and I'll run `mv` on the ones you mark.

Cross-referenced `data/json/*/` against experiment names defined in `config/experiments.yaml` and `noise/experiments_noisy.yaml`.

| Folder | Size | Matches a live experiment set? | Archive candidate? |
|---|---|---|---|
| `10_runs_gpt5_myth_topics/` | 43M | Not an `experiment_sets` entry, but **used as the gpt-5-nano baseline** by `analyses/noise_balance_comparison.py` (see the `--baseline gpt-5-nano=...` example in its `--help`) | **No — keep** |
| `10runs_model_comparison/` | 88M | Yes (`10runs_model_comparison`) | No — live |
| `10runs_non_coop_model_comparison/` | 14M | Yes | No — live |
| `deepseek_selfish/` | 348K | Yes | No |
| `demo/` | 296K | Yes | Small, keep where it is |
| `game_prompt_bug_fix/` | 336K | Yes | No |
| `main_loop/` | 256K | No | Likely — outputs from manual `run_trust_game.py` runs |
| `model_comparison_cooperation/` | 824K | Yes | No |
| `model_comparison_task_order/` | 180K | Yes | No |
| `model_comparison/` | 804K | Yes | No |
| `multiplier_increasing/` | 124K | Yes | No |
| `myth_topics/` | 14M | No (not a current config name) | Possibly — older iteration? |
| `no_defection_good/` | 1.4M | Yes | No |
| `noise_experiments/` | 380M | Yes (umbrella for all noise configs) | No |
| `pilot_other_models/` | 46M | Yes | No |
| `pilot/` | 14M | Not in config (`pilot` is referenced in `run_trust_game_batch.py` default, but the set isn't defined in `config/experiments.yaml`) | Check — either code is broken or the set existed earlier |

## Suggested default moves (pending your sign-off)

Revised after verifying against `analyses/noise_balance_comparison.py`:

- `data/json/main_loop/` (256K) → `archive/data/json/main_loop/` — outputs of manual single-run tests, no config reference
- `data/json/myth_topics/` (14M) → `archive/data/json/myth_topics/` — not a current config name and not referenced by any analysis script

**Do NOT archive** `10_runs_gpt5_myth_topics/` — it's the gpt-5-nano baseline for the balance-comparison plots.

## What I won't auto-move

- `pilot/` — even though "pilot" isn't in the current config, it's referenced by `run_trust_game_batch.py` as a default, and may correspond to your earliest experiments (foundational results).
- Anything that matches a current experiment name exactly.
- `noise_experiments/` — huge, but fully live.

# archive/

Holding area for files that were removed from the active codebase but **not deleted** — either because they still have reference value, might be needed again, or their role is uncertain. Contents here are not imported or executed by the main framework.

If you find yourself reaching into `archive/` to wire something back into the main code, that's a signal the file shouldn't have been archived. Move it back out.

## Inventory (2026-04-24)

### `games/trust_game_old.py`
Older implementation of the trust game, predating the current `games/trust_game.py`. Missing `multiplier_distribution` support. A codebase-wide grep found **zero import references**, so moving it out of the live tree does not break anything.

### `noise/tmpb6g_ehc3.yaml`
Scratch YAML written by `noise/run_noisy_batch.py` (see the `tmp*.yaml` entry in `.gitignore`). Kept in case the exact config it captured is ever needed for reproducibility.

### `credentials.json`
Google API credentials (untracked, gitignored). Moved here so it no longer sits in the repo root. Delete from this directory when you're sure you don't need it — it contains secrets, even if not committed.

### `token.pickle`
OAuth token cache corresponding to `credentials.json`. Same guidance.

## data/plots/ reorganization (2026-04-24)

`data/plots/` was restructured to mirror `data/json/`'s `baseline / myth_topics / noise_experiments` layout. The following plot folders had no parallel under `data/json/` (their source JSONs were dropped from the active tree when the 3-model baseline was curated) and were moved here instead of deleted:

### `data/plots/10runs_model_comparison/`
Plots for three models that did not make the final baseline: `gemini-3-flash-preview/`, `grok-4.1-fast/`, `mistral-small-creative/`. The plots for `claude-sonnet-4.5` and `gemini-3-pro-preview` from the same parent folder were promoted into `data/plots/baseline/` since those models are live baselines.

### `data/plots/10runs_non_coop_model_comparison/`
Plots for the "non-cooperative prompt" condition experiment (3 models: claude-sonnet-4.5, gemini-3-pro-preview, gpt-5-nano). The underlying JSONs are no longer in the active tree.

## `noise/` directory split (2026-04-24)

The former `noise/` directory was dispersed into the normal top-level buckets so there's a single canonical home for each file type:

| Was at | Now at |
|---|---|
| `noise/trust_game_noisy.py` | `games/trust_game_noisy.py` |
| `noise/public_goods_game.py` | `games/public_goods_game.py` |
| `noise/experiments_noisy.yaml` | `config/experiments_noisy.yaml` |
| `noise/run_noisy_batch.py` | `experiments/run_noisy_batch.py` |
| `noise/analyze_conditions.py` | `analyses/analyze_conditions.py` |
| `noise/__init__.py` | `archive/noise/__init__.py` (re-exports that are no longer needed) |

The `noise/` directory itself is gone. All imports and path references have been updated; run the runner as `python experiments/run_noisy_batch.py ...`.

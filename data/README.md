# Data

Experiment outputs are written here by the runners in `experiments/`:

```
data/json/    full run state, per-turn logs, lightweight .results.json
data/plots/   analysis figures
data/share/   curated bundles for sharing
```

`data/json/`, `data/plots/`, and `data/share/` are gitignored — they run to
multiple GB and are reproducible from `config/experiments.yaml` plus the
runners. The full dataset for the paper will be archived separately
(Zenodo link TBD).

`data/sample/` contains two small tracked runs so the analysis scripts can be
tried without running an experiment:

- `myth_topics_000_claude-sonnet-4.5_neutral_game.json` — trust game only
- `myth_causal_confirm_..._myth_game_directive_anything.json` — myth writing
  followed by trust game

```bash
./scripts/run_all_analyses.sh -a trajectory data/sample/myth_topics_000_claude-sonnet-4.5_neutral_game.json data/plots/sample
```

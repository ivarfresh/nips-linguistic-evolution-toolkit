# Folding A1 + A3 into the existing analysis pipeline

After the runs complete, the existing `build_*.py` scripts can absorb
the new cells with a one-line edit each.

## 1. Add new subdirs to the include list

Edit `INCLUDE_VERSIONS` in each generator:

```python
# In build_cell_summary.py, build_lag_and_lexicon.py,
# build_neologism_analysis.py, build_reason_coding.py,
# build_embedding_convergence.py:

INCLUDE_VERSIONS = {
    "v4_direct_provider",
    "v4_direct_provider_A1_partner_myth",      # NEW
    "v4_direct_provider_A3_forced_reasoning",  # NEW
}
```

(And in `build_headline_tables.py`, similarly extend `HEADLINE_VERSIONS`.)

## 2. Add NOISE_LABEL_FROM_EXPERIMENT mappings

Each generator script has a small dict mapping experiment-set name to
a short noise label. Add the new ones:

```python
NOISE_LABEL_FROM_EXPERIMENT = {
    ...existing entries...
    "gpt5nano_forced_reasoning": "varied",       # contains all 4 noise types
    "gpt5nano_partner_myth_injection": "varied", # contains all 4 noise types
}
```

But — because these new experiment_sets contain MULTIPLE noise types in
their `game_params_list`, the generators currently group all runs under
a single noise label. Two options:

**Option A (cleaner):** modify the generator to extract noise from the
`run_metadata.game_params_name` field of each JSON, instead of from the
experiment-set name. The directory layout under v4 already separates by
`{noise_condition}` so this is mostly a path-parsing change.

**Option B (faster):** split the experiment_sets in the YAML into one
per noise condition (8 sets instead of 2). Same total runs, just more
granular bookkeeping. Recommended for the deadline.

Example for option B — replace the single `gpt5nano_partner_myth_injection`
with four sets named `gpt5nano_partner_myth_positive`,
`gpt5nano_partner_myth_positive_informed`, `gpt5nano_partner_myth_negative_5`,
`gpt5nano_partner_myth_bootstrap`, each with one entry in
`game_params_list`. Then the existing path-parsing code finds them
automatically.

## 3. Re-run the analysis

```bash
cd /Users/aron/nips-linguistic-evolution-toolkit
python3 projects/neurips-2026-llm-ling-evo/analysis/build_cell_summary.py
python3 projects/neurips-2026-llm-ling-evo/analysis/build_lag_and_lexicon.py
python3 projects/neurips-2026-llm-ling-evo/analysis/build_neologism_analysis.py
python3 projects/neurips-2026-llm-ling-evo/analysis/build_reason_coding.py
python3 projects/neurips-2026-llm-ling-evo/analysis/build_embedding_convergence.py  # ~25 min
python3 projects/neurips-2026-llm-ling-evo/analysis/build_headline_tables.py
python3 projects/neurips-2026-llm-ling-evo/analysis/plot_decision_table.py
python3 projects/neurips-2026-llm-ling-evo/analysis/plot_trajectories.py
python3 projects/neurips-2026-llm-ling-evo/analysis/plot_variance_summary.py
python3 projects/neurips-2026-llm-ling-evo/analysis/plot_linguistic_dynamics.py
python3 projects/neurips-2026-llm-ling-evo/analysis/plot_reason_coding.py
```

## 4. Compare A1 / A3 cells against baseline

A new analysis worth adding (`build_A1A3_comparison.py`):

For each (noise_condition × task_order) pair, compare the
*partner-myth-injection* cell against the matched *unmodified* v4 cell
on three axes:

- **Cooperation effect:** Δmean(A1 cell) vs Δmean(matched baseline cell).
  If A1 lifts cooperation MORE than the baseline did, that's evidence
  the linguistic channel is causally active.
- **Variance ratio:** does the consolidation pattern get sharper?
- **Reason-coding rate** (A3 only): does forced reasoning produce
  Claude-like ~80% own-myth-vocab references in GPT-5-Nano, or does it
  stay low?

Headline cells for the comparison:

| Cell | A1/A3 finding sought |
|---|---|
| gpt5-nano × positive (informed) × game_myth | Compare Δmean A1 vs baseline (baseline was lift+consolidation Δmean +2.23) |
| gpt5-nano × positive × myth_game | Same comparison (baseline was lift+consolidation Δmean +2.08) |
| gpt5-nano × bootstrap × game_myth | Critical: does opening the channel rescue the destabilization (baseline harmful Δmean -7.73) or worsen it? |
| gpt5-nano × bootstrap × myth_game | Same (baseline harmful Δmean -7.60) |

## 5. Manuscript-side updates if A1 / A3 produce findings

Add a §3.6 to `_3-results.b_draft.md` titled "Tightening the cross-task
channel" (~150–200 words). Include:

- Brief description of the partner-myth-injection variant
- The 2×2 comparison table (cell × {baseline, A1}) for the headline cells
- One paragraph on what changed
- Reference back to §4.5 (reason-coding) and §4.4 (lag-1, embedding)

Add to §5.4 (cross-task influence): one paragraph on how A1's results
affect the constrained-channel reading.

If A3 produces Claude-like reason-vocabulary rates in GPT-5-Nano, fold
into §3.5 — the "undecidable for GPT-5-Nano" caveat partly resolves.
If it doesn't, the cross-model variation finding (§5.3) gets stronger.

# Reproducible conditions and explicit comparisons

The original bug was hidden variation: the same named experiment could send
different API settings on different machines. Matching parameter labels across
vendors does not solve this. A comparison must declare what it varies and keep
the remaining inputs fixed.

## New runs

Pin `llm_settings.provider`, `reasoning`, and `temperature` in the experiment set.
An optional positive `max_output_tokens` pins the output cap; pinned Anthropic
calls otherwise use 4096, while other routes omit the cap. Environment overrides
of provider, reasoning, and temperature remain explicit in metadata. Legacy
vendor-specific reasoning knobs and token-cap environment variables do not
override a pinned request. The resolved request plan is retained by the client
for the duration of the run.

`run_metadata.experiment_condition` records the resolved request, actual prompt
templates, game and memory policies, task order, retry policy, seeds, replicate
identity, and hashes of simulation source files. `condition_sha256` binds that
record. Each response records `usage.request_settings` and `finish_reason`.
An omitted parameter means vendor default, not a measured effective value;
unreported reasoning-token counts remain unknown, not zero.

Resume requires exactly the same condition and preserves the original metadata.
Changed settings, prompts, seeds, or source code require a distinct run rather
than silently relabelling an old checkpoint. Pre-condition checkpoints cannot
be automatically resumed with a claim of equivalence.

## Comparisons

Top-level `comparison_sets` declarations list experiment sets and an
`allowed_differences` mapping from individual expanded configuration fields to
written reasons. CI and runners expand the configurations, check replicate
alignment, and reject any undeclared difference. The controlled format study
in `config/experiments_noisy.yaml` allows only `game_params.decision_format`:
all three new arms use identical game-directed, memory-primary myth prompts.
These new `fmt_controlled_v2_*` sets are configured, not yet run.

`analyses._shared.load_simulation_runs` checks complete saved conditions across
all inputs, not just provider settings within each model. Explicitly allow
replicate differences when pooling repeats; allow model, provider, or reasoning
differences only when these are part of the declared comparison. Equal names
such as `low` do not establish equal computation across vendors.

The cooperation-ratio and resources-max plotting entrypoints validate inputs
before writing figures. `--comparison-spec` accepts a JSON object mapping saved
condition fields to reasons, for example
`{"replicate": "Independent repeats", "protocol.game.decision_format": "Format intervention"}`.
Historical runs additionally require `--legacy-reason`; unknown values remain
unknown. Existing single-file readers remain available for historical tools;
not every older analysis entrypoint has been migrated.

Output `provenance.json` files contain input paths and hashes, complete recorded
conditions, declared and independently recomputed differences, and output
hashes. Regenerate them with `scripts/write_provenance.py`, specifying each
`--allow-difference FIELD=REASON`. CI rejects changed or new output directories
without valid manifests. Legacy exemptions are checked against a committed,
checksum-locked snapshot of historical definitions and file identities. They
cannot grow; modified legacy sets must be pinned. The check also works after
squash merges and in shallow CI checkouts.

## Completion and historical data

Only the final full-state JSON proves completion. Upload eligibility is a
separate check: automatic HF sync requires complete condition and per-call
provenance. It does not upload orphan sidecars, and an upload failure does not
invalidate a completed run. The explicit `--allow-legacy-provenance` migration
option accepts historical records, but not corrupt modern condition records.

Historical experiment sets need `--allow-legacy-settings` to use their old
environment-dependent behavior. This is an acknowledged legacy path, not a
matched-condition guarantee. The missing-run wrapper verifies existing finals
and their recorded expanded configuration instead of trusting file existence.

The September 4 format results remain exploratory. See the correction in
`data/analysis/decision_format_confound_2026_09_04/README.md`; neither a pure
format effect nor a prose-in-memory mechanism was identified by those runs.

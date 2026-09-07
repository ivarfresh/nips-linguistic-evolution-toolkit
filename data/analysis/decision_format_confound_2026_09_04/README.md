# September 4 decision-format comparison: exploratory only

Correction recorded September 6, 2026. The numerical summaries are reproducible,
but these runs do **not** isolate the effect of output format.

- The reference uses `myth_writing_default_game_directive` and
  `myth_writing_later_rounds_directive_memory_primary`.
- Both new format arms use `myth_writing_default` and
  `myth_writing_later_rounds`: the myth instructions and self-context change too.
- The JSON-only reruns also use a changed corrective retry policy. Different
  source revisions and incomplete historical settings are recorded in the
  provenance manifest rather than treated as equivalent.

The observed sending difference therefore cannot be assigned to forbidding
prose, nor does it establish that prose in memory causes Claude's behavior.
Raw runs and the historical research log remain unchanged.

Reproduce the descriptive summary and checked provenance from local raw runs:

```sh
python3 analyses/decision_format_confound.py --exploratory
```

The manifest records every known differing field and why it is included in
this exploratory comparison. Unknown historical fields are not inferred as
matched. Corrected `fmt_controlled_v2_*` experiment sets use the same directed
myth prompts and retry implementation in every arm; they have not been run.

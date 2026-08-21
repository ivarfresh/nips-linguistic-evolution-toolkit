# GPT-5 Nano replication of the corrected 8-agent task-order effect

## Design

This exploratory replication used GPT-5 Nano in the corrected 8-agent rotating-
population protocol. It crossed the same three task orders used in the Sonnet
confirmatory batch: game only, game→myth, and myth→game. Each cell contains five
independent eight-agent runs of ten rounds.

The design holds the pairing schedule and signed communication-noise draws fixed
within each replicate across task orders. The five pairing/noise seed pairs are
202608200–202608204. Agents receive their own recent interactions through
private chat memory and a three-game prompt dossier about the current co-player;
co-player names are hidden. Communication noise is explicitly configured as
uniform U(−1,+1), clipped to the feasible action range and applied after both
sending and returning decisions.

Thirteen selected runs use executable snapshot `3a957f41`; exact-seed
replacements for Game→Myth replicate 1 and Myth→Game replicate 0 use
`0846a69f`. All selected runs used the direct OpenAI route and record a clean
worktree in their metadata.

The two replacements were necessary because the original files contained an
accepted decision-as-myth response such as `{"send": 5}`. The first audit
checked prompt, transfer, memory, and call integrity but did not yet apply a
semantic myth/task-boundary check. Those original files are preserved for
diagnosis and excluded from every result below.

## Protocol audit

All 15/15 selected runs passed the expanded `scripts/audit_v2_protocol.py`
jointly:

- 150 complete population-rounds and 600 completed dyads;
- 2,000 accepted LLM task responses and zero recovered or unrecovered retries;
- 1,200 numerical transfer-noise checks and zero bound violations;
- correct myth-decision instruction placement, accepted-myth task boundaries,
  and memory behavior; and
- identical realized pairing schedules across task orders for each matched seed.

## Results

Values below are run-level means with 95% t intervals across the five independent
populations.

| Condition | Final balance per agent | Proportion sent | Proportion of tripled amount returned | Dollars returned / dollars sent |
|---|---:|---:|---:|---:|
| Game only | 62.64 [61.10, 64.17] | 0.753 [0.722, 0.783] | 0.387 [0.339, 0.435] | 1.168 [1.020, 1.315] |
| Game→Myth | 62.34 [60.88, 63.80] | 0.747 [0.718, 0.776] | 0.362 [0.344, 0.379] | 1.085 [1.026, 1.144] |
| Myth→Game | 66.26 [64.99, 67.53] | 0.825 [0.800, 0.851] | 0.405 [0.339, 0.471] | 1.207 [1.003, 1.411] |

Paired final-balance contrasts:

| Contrast | Difference | 95% paired CI | Raw p | Holm p (three contrasts) |
|---|---:|---:|---:|---:|
| Game→Myth − Game only | −0.30 | [−3.05, +2.46] | .779 | .779 |
| Myth→Game − Game only | +3.63 | [+2.45, +4.80] | .001 | .003 |
| Myth→Game − Game→Myth | +3.92 | [+1.38, +6.47] | .013 | .026 |

Final balance is a welfare measure and is algebraically determined by sending in
this game: returns redistribute resources within a dyad but do not change the
dyad's joint balance. The final-balance effect therefore corresponds exactly to
a 7.25 percentage-point increase in the proportion sent for myth→game versus
game only. No return-ratio contrast was detected at n=5.

## Interpretation

The cheap GPT model reproduces the central Sonnet ordering: myth→game is higher
than both game only and game→myth, whereas game→myth is essentially identical to
game only. The five paired differences for myth→game versus game only are all
positive. This is useful cross-model evidence that the ordering is not unique to
Claude Sonnet 4.5. These corrected values supersede the initial same-day
estimates that included the two task-boundary failures.

The result remains exploratory because n=5 is small and GPT-5 Nano is a single
additional model. The p values quantify this planned paired screen; they should
not be treated as definitive estimates or pooled with the Sonnet batch without
a model-level analysis.

The preceding Gemini 3.1 Flash-Lite smoke reached the sending ceiling in every
condition, making it uninformative about task-order differences under this
protocol.

## Cost and reproducibility

The 15 selected GPT runs used 4,575,142 input tokens and 238,722 output tokens,
with no reported reasoning tokens. At the recorded list-price rates, the
estimated cost was $0.324.

Reproduce the tables and figures with:

```bash
python3 scripts/analyze_crossmodel_signed_gpt_n5.py
```

Outputs are in `docs/figures/crossmodel_signed_gpt_n5_20260821/`.

![Final balance](figures/crossmodel_signed_gpt_n5_20260821/final_balance_gpt_n5.png)

![Behavior metrics](figures/crossmodel_signed_gpt_n5_20260821/behavior_metrics_gpt_n5.png)

![Trust trajectories](figures/crossmodel_signed_gpt_n5_20260821/trust_trajectories_gpt_n5.png)

## Recommended follow-up

The next causal experiment should manipulate information visibility in the
8-agent population while holding model, pairings, noise, identity cues, and
memory horizon fixed. Start with game-only own-history versus current-partner
dossier as a supported two-arm gate, then add a population-wide public ledger
arm after specifying how stable pseudonyms and noisy third-party observations
are represented. This separates direct partner reputation from genuinely
population-wide third-party reputation without conflating the two.

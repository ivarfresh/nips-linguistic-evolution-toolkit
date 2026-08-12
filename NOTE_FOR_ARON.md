# Note for Aron — noise buffering is a population-size effect

Hi Aron — dropping the state of a small investigation on this branch
(`informed-dyad-triplet`), plus one experiment you could run to finish it.

## What we found

Under **informed bidirectional noise** (memory-primary, Sonnet 4.5, game-only,
n=5/cell), cooperation survives in 8-agent populations but collapses in 2-agent
dyads — and the cause is **`num_agents` itself**, not any of the prompt/memory
features we suspected. Final cumulative balance (mean per agent):

| 8-agent (HOLD) | | 2-agent (COLLAPSE) | |
|---|---|---|---|
| rotating | 60.6 (±1.6) | plain dyad | 46.3 (±2.6) |
| fixed pairs | 59.4 (±2.6) | dyad + cap 6 | 43.4 (±2.6) |
| no co-player block | 58.0 (±3.2) | dyad + anon | 42.9 (±2.2) |
| | | dyad, **full 8-agent config** | 44.6 (±1.8) |

Every specific channel is ruled out: **rotation** (fixed pairs still hold),
**co-player history block** (removing it still holds), **memory depth**,
**anonymity alone**, and the **full config bundle** — giving a 2-agent dyad an
exact copy of the 8-agent config (`noise2i_full8match_game`) does *not* rescue
it. Perfect separation by agent count regardless of config. Details +
per-cell configs in `researchlog.md` (2026-08-11 entries) and
`config/experiments_noisy.yaml` (the `noise2i_*` / `noise8i_*` sets).

## The open question — and the test you could run

**Why does population size buffer noise?** Leading hypothesis: *anonymity only
bites in a population* — with 8 anonymized agents you can't tell whether "another
agent" is the same partner, so targeted retaliation can't build; in a 2-agent
dyad anonymity is vacuous (only one possible partner), so the spiral proceeds.

That predicts a clean test: **named fixed-pairs 8-agent should collapse.** It's
configured and ready to run (one cell, ~$10):

```bash
LLM_PROVIDER=openrouter \
python experiments/run_noisy_batch.py noise8i_fixed_named_memprimary_game --workers 4
```

- **Collapses toward ~46** → anonymity-in-a-population is the mechanism.
- **Holds at ~59** → it's something else about having more agents; the mechanism
  is still open.

(`pairing_mode: fixed` was added this session — `games/dyadic_pairing.py`
`_get_fixed_multi_agent_pairings`; wired through `trust_game_noisy.py` and
`run_noisy_batch.py`. The named cell is `noise8i_fixed_named_memprimary_game` in
`config/experiments_noisy.yaml`.)

— Ivar (with Claude)

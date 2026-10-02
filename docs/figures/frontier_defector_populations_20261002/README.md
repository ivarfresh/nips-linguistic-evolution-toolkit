# Frontier populations with defectors, 2026-10-01/02: Opus 5 and GPT-5.6 Sol

**Headline.** With two defectors in the group, myths raise cooperation among frontier
agents again: by about 8–9 points in all-Opus groups and by 5–8 in the Opus + Sol mix.
They do not reliably help all-Sol groups. The mix ends close to the average of its two
single-model groups (within 2 points; every interval includes zero), so the mid-tier result
that mixed groups beat their parts does not appear here. Myths also make agents easier to exploit: they roughly double what agents
send to a defector.

## Why this run

Without defectors, frontier groups sit near the ceiling of 75 and myths have no room to
act (the 2 Gemini / 3 Opus / 3 Sol population: 72.2 in game only, 74.7 with
myth → game). The paper's Limitations also note that no frontier group contains a model
that withholds, the role GPT-5 Nano plays in the mid-tier runs. Two scripted defectors
fill that role and lower the game-only level.

## What was run

- **Groups:** 8 agents, balanced rotating pairs, hidden names.
  - 4 Opus 5 (Agent_1–4) + 4 GPT-5.6 Sol (Agent_5–8);
  - 8 Opus 5;
  - 8 Sol.
- **Defectors:** Agent_4 and Agent_8 in every run (one per family in the mix). They
  always send and return $0 without a model call, still write myths, and are not told
  they are defectors. Settings are the September `defectors25` cell apart from the fixed
  seats; September drew the seats per replicate.
- **Models:** each at its 2026-09-18 frontier request profile (D011). Gemini 3.1 Pro is
  left out because it ends at the ceiling in every frontier condition.
- **Design:** game only, game → myth, myth → game × 5 replicates = 45 runs, informed
  negative-only noise, 10 rounds. Replicates share noise and pairing seeds across groups
  and task orders, so differences are seed-paired.
- **Cost:** $106.95 at standard rates, from recorded token usage. All 45 finals passed
  the launcher's audit (PR #20, independently reviewed).

Launcher: `scripts/run_frontier_defector_populations.py`.
Analysis: `scripts/analyze_frontier_defector_populations_20261002.py`.

## Results

`resources_boxplots.png`: final resources of the six ordinary agents, one dot per run.

| Group | Game only | Game → Myth | Myth → Game |
|---|---|---|---|
| 4 Opus 5 + 4 Sol | 49.1 (±6.3) | 54.6 (±1.6) | 56.8 (±2.1) |
| 8 Opus 5 | 48.4 (±2.3) | 56.5 (±1.6) | 57.3 (±1.2) |
| 8 Sol | 53.4 (±5.5) | 50.5 (±5.1) | 55.9 (±2.1) |

Mean (±sd) over 5 runs of the ordinary agents' final resources.

**Myths help Opus 5 and the mix, not Sol** (`myth_effects.csv`; seed-paired myth minus
game):

| Group | Game → Myth | Myth → Game |
|---|---|---|
| 4 Opus 5 + 4 Sol | +5.5, 5 of 5 pairs up, Welch p 0.12 (paired 0.085) | +7.6, 5 of 5 up, p 0.052 (paired 0.026) |
| 8 Opus 5 | +8.1, 5 of 5 up, p < 0.001 (paired 0.002) | +8.9, 5 of 5 up, p < 0.001 (paired 0.001) |
| 8 Sol | −2.9, 2 of 5 up, p 0.41 (paired 0.54) | +2.4, 3 of 5 up, p 0.39 (paired 0.31) |

Welch p as in the paper's Table 1; paired t-test p in brackets, since the runs are seed-paired.

**The mix equals its parts** (`mix_vs_parts.csv`; mix minus ½ · 8 Opus + ½ · 8 Sol,
bootstrap 95% interval): game −1.8 [−7.8, +3.2]; game → myth +1.1 [−1.4, +3.6];
myth → game +0.2 [−1.7, +2.1]. In the mid-tier runs the corresponding mixed groups earned
up to 17 points more than their parts with myths (paper Table 1).

**Myths make agents easier to exploit** (`sends.csv`; mean send per decision, of $5):

| Group | Task order | To an ordinary agent | To a defector |
|---|---|---|---|
| 4 Opus 5 + 4 Sol | game | 3.54 | 0.67 |
| | game → myth | 4.47 | 1.35 |
| | myth → game | 4.80 | 1.52 |
| 8 Opus 5 | game | 3.49 | 0.88 |
| | game → myth | 4.84 | 1.86 |
| | myth → game | 4.94 | 1.79 |
| 8 Sol | game | 4.15 | 0.59 |
| | game → myth | 3.82 | 1.02 |
| | myth → game | 4.65 | 1.34 |

Agents with myths keep sending to a partner who never returns anything, about twice as
much as without myths, in every group. With myths, ordinary agents also send almost the
full $5 to each other (4.8–4.9 in Opus groups and the mix under myth → game). The
remaining distance to 75 comes from the defectors: they return nothing, and ordinary
agents keep sending them $1.0–1.9 per decision.

## Myth board pilot (same mix, 2026-10-02)

Branch `feature/myth-board-20261002`. From round 2 every agent read every myth written
so far (a shared, persistent board without author names) instead of its last partner's
myth. Three myth → game runs, $22.76, matched by seed to the runs above:

| Replicate | Partner's myth | Shared board |
|---|---|---|
| 0 | 56.00 | 55.42 |
| 1 | 58.83 | 58.33 |
| 2 | 54.75 | 55.25 |

No difference, and no sign that myths converged (word-overlap similarity within a round
stays at 0.31–0.40 with and without the board; a quick TF-IDF check, script not committed). With partner myths, ordinary agents
already send $4.81 of $5 to each other, so the board has no room to act. The remaining
seven board runs were not run (paused 2026-10-02).

## Caveats

- Five runs per cell (three for the board): exploratory.
- Part of the drop in resources is mechanical: a defector partner returns nothing.
- In the single-model groups both defectors belong to that model.
- No no-defector Opus + Sol control; the comparisons above are all within the defector
  design.
- Defector seats are fixed, so the Sonnet comparison in the September defector cells is
  not like-for-like.

## Draft text for the paper

**Section 4.3, after the mixed-population paragraph:**

> Frontier groups leave little room for myths to act, so we added two scripted
> defectors, agents that always send and return nothing, to eight-agent groups of Claude
> Opus 5, GPT-5.6 Sol, and a 4 + 4 mix (Gemini 3.1 Pro, already at the ceiling, was left
> out). Defectors lower game-only resources to about 49 for Opus and the mix and 53 for
> Sol. Myths then raise them again: by 8–9 points in Opus groups (all ten seed-paired
> runs higher, Welch p < 0.001) and by 5.5–7.6 in the mix, though not reliably in Sol
> groups. Unlike the mid-tier groups, the mix earns no more than the average of its
> parts (−1.8 to +1.1 across task orders, all intervals including zero). None of these
> frontier models withholds as GPT-5 Nano does (Sol only partly collapses, in 2 of 10
> game-only runs), which may be why composition matters less; we did not test this
> directly.

**Section 5.1, after "withdrawing their own generosity":**

> Myths make this weakness worse. In frontier groups with scripted defectors, agents
> with myths sent a defector about twice as much as agents without them ($1.0–1.9 against
> $0.6–0.9 per decision), while sending $4.6–4.9 of $5 to each other under myth → game. The
> channel that raised cooperation also made agents easier to exploit.

**Supplementary material, frontier models:**

> A shared board on which every agent read every earlier myth, rather than only its last
> partner's, did not change cooperation in three seed-paired runs of the defector mix
> (−0.6, −0.5, +0.5) and did not make myths more alike. Ordinary agents there already
> sent $4.8 of $5 to each other, so the board had no room to act.

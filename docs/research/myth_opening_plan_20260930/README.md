# Why the myth task raises cooperation: the myth is the agent's opening plan

*2026-09-30. A synthesis of one day's analyses (PRs #11, #12, #15, #16), with the evidence
trail and the cheapest test that would settle it.*

## The answer

The myth task raises cooperation because it makes each agent write down a plan before it
plays. The agent then follows that plan on its first move, and a generous opening sustains
itself through ordinary reciprocity. What matters in the myth is the send amount or rule the
agent writes into its own myth. Its moral and whether partners share norms show no detectable
effect on play in our data; the drift toward "consistency" shows none on cooperation levels,
with one unconfirmed Sonnet hint (section 6).

This is the explanation most consistent with the evidence, not a measured chain. Section 6
says what is not shown, and section 7 gives a cheap test.

## Update 2026-10-01: the replay probe

The cheap test in section 7 ran (PR #19, `docs/research/myth_replay_probe_20261001/README.md`).
Adding one rule sentence to an agent's own myth, *"whoever holds five should send X"*, moves its
next send $0.93 per $1 in its first myth and $0.68 in a myth written after round 1 (Sonnet and
GPT; Gemini sends $5 regardless). So the first link holds causally, for an explicitly stated rule;
amounts told inside the story move sends less. Two claims below change:

- **A read myth is not inert.** The same rule in a partner's myth moves the reader's send $0.23
  per $1, a lower bound (section 6 said no detectable effect).
- **The opening explains at most part of the task-order gap.** Holding the round-1 send fixed
  shrinks the story-first advantage in single-model populations but not in mixed or pooled runs.

## 1. The question

Myth→game runs cooperate more than game-first runs. We had
labelled every myth's moral ("be generous / be fair / be cautious") and plotted cooperation by
moral. The moral barely predicted play. Ivar asked two things: is that because the LLM judges
are bad, or because norms in myths don't drive cooperation? And what in the myths does?

## 2. What we did, in order

1. **Moral by behaviour, split by family, setting and task order** (PR #11). Agents that write
   or read generous myths cooperate more on average, but we found no detectable shift within
   an agent when its moral turns generous. Review fixed two problems before merge: in mixed
   dyads the gap partly measured the partner's family (Sonnet 2-agent mixed game→myth, shown
   myth: +0.21 → +0.08 once the partner's family is controlled), and 5-run cells needed
   small-sample (t) inference.
2. **Paper figures for how morals move through populations** (PR #12). Three designs were
   prototyped in parallel; two were kept (`docs/figures/linguistic_analysis_20260923/`
   `moral_composition_by_round_*.png`, `moral_lineage_network.png`). Within a family a moral
   carries weakly from the myth an agent read into the myth it writes next; across families it
   does not. The homogeneous +4.8-point figure later failed a placebo (+3.9, p 0.046; linguistic README).
3. **Six parallel lenses on what in the myths predicts cooperation** (PR #15, all 156
   September myth runs, 8,519 myths): judge quality, norm alignment between partners, stated
   send amounts, the consistency drift, an open search over ~150 myth features, and the
   myth-transplant experiments. Each result is graded by how clean its evidence is (round 1
   before play; myths read from a partner versus a comparable unread myth; placebo myths; the
   transplants). An independent check of the synthesis against the lens tables and raw runs
   led to corrections before it merged.
   Spend: $0.83.
4. **The same analyses on the frontier runs** (PR #16: Claude Opus 5, Gemini 3.1 Pro, GPT-5.6
   Sol; 106 myth runs, 4,520 myths). The code rebuilds the September data byte-identically, and
   every lens reproduced its September headline before touching frontier data. A verified
   scorecard puts September beside frontier. Spend: $10.17 (judging). This rerun surfaced five
   corrections to the September write-ups, all applied.

Details and every number's source: `docs/research/myth_predictors_20260930/README.md`
(September), `docs/research/frontier_myth_predictors_20260930/README.md` and
`scorecard/scorecard.png` (frontier beside September).

## 3. An agent does what its own first myth says

In myth→game runs the agent writes a myth, then makes its first send. Among Sonnet and GPT
senders whose round-1 myth names an amount, the send follows it: **$0.67 per $1 stated**
(95% CI 0.55–0.79; 62 senders, 37 runs), and **71% send exactly the stated amount** (Sonnet
80%, $0.85 per $; GPT 55%, $0.48 per $). Filling myths without a number from their send-rule
band gives $0.58 (150 senders). The stated amount and send rule explain 25% of held-out
round-1 send variance. Gemini's round-1 myths all say $5, and it sends $5 (50 of 50).

**Why this is the myth, not a mood.** Within an experimental cell every agent gets the same
system prompt and instructions, the model keeps no state between calls, and only message
role and content reach the API. We checked the raw runs: in all 78 September myth→game runs
the round-1 send call differs between agents only by the agent's own myth. So the text is the
channel. Which feature of the text carries it is not identified by this alone, and it covers
senders only (a receiver also sees the partner's send). It is also an instructed plan: the
myth prompt asks how the game should be played, and the decision prompt says to take the
myths into account.

**The frontier models follow their plan even more tightly:** 50 of 52 senders send exactly
the named amount ($0.85 per $; Sol 31 of 31, Opus 19 of 21; Gemini 3.1 Pro always $5). The
Opus slope ($0.61) is suggestive only: just 5 Opus senders sent under $5.

## 4. The opening is where the task-order effect lives

Round 1 is where task order first changes an agent's input: in myth→game both players have
their own myth in context; in game→myth the round-1 send comes before any myth, as in
game-only runs. September round-1 sends, mean over runs (± sd):

| Family | game→myth (no myth yet) | myth→game (own myth first) |
|---|---|---|
| GPT-5 Nano | $0.18 (±0.66) | $3.10 (±1.10) |
| Sonnet 4.5 | $3.02 (±0.21) | $3.92 (±0.73) |
| Gemini 3.7 Flash | $5.00 (±0) | $5.00 (±0) |

Source: `data/analysis/linguistic_20260923/decisions.csv`, investors in round 1, averaged per
run (42 / 31 / 27 runs per cell, homogeneous and mixed together).

The round-1 myths mostly name high amounts, which fits a generous opening. In round-1
myth→game myths the judge extracts "send all" for 100% of Gemini myths, 51% of Sonnet's
(12% "most", 37% "moderate") and 22% of GPT's, whose most common rule is "moderate" (56%;
"little" 8%, "unspecified" 13%) (`myth_rules_september_z-ai__glm-5.2.csv`, all 426 round-1
myth→game myths).

Two earlier results fit this:

- **GPT's zero-lock.** In game-only runs GPT-5 Nano opened with $0 in every run at both sizes
  and never recovered; a myth task loosens the lock, fully in dyads and partly in populations
  (zero-receipt rate 0.12 / 0.08 in dyads, 0.59 / 0.40 at 8 agents for game→myth / myth→game;
  researchlog 2026-09-10). The opening explains the myth→game half: with its own myth first,
  GPT opens at $3.10. It does not explain the game→myth half, where GPT also opens at about
  $0 and escapes later, after its first myth.
- **The "founding myth" account.** Myth-first populations open generous and reach the $5
  ceiling by about round 4; game-first populations cold-open near $3 and climb slowly
  (`docs/architecture/findings-taskorder-myth.md`; the round-4 figure is from the 2026-08-19
  20-round washout, Sonnet on an older apparatus, n = 5 per arm, directional).

After the opening, the plan's direct pull fades: the round-1 send rule predicts later sends
only weakly over rounds 2–10 (+0.018 send fraction per rule level, p 0.023) and not over
rounds 6–10 (+0.011, p 0.27), against about +0.14 at round 1. What keeps the level up appears to be play
itself: later myths mostly record the game just played (a $1 higher send → $0.66 higher stated
amount in Sonnet's next myth).

## 5. The transplants agree, and the moral doesn't

In the slide-678 transplant reruns a donor myth was placed in the host's own myth slot, and
hosts treat it as their own ("the myth I just wrote…"). The donor's stated amount orders the
outcome: +$0.41 sent per $1 stated in the 8-agent reruns (Holm-significant; +$0.29 without the
one "send nothing" donor; +$0.15 on the original June–July apparatus; the dyad +$0.34 is not
Holm-significant). Across the 23 donors that state an amount, rank correlation 0.69 (Holm p
0.031). The moral label cannot order it: the judge calls 20 of 30 donors "be fair", including
the top and bottom cells. Caveat: old apparatus, Sonnet hosts only.

## 6. Everything else came up empty, and what is not shown

No detectable effect on play from:

- **A myth read from a partner.** Its stated amount moves the reader's next send by −$0.00
  (−0.10 to 0.10; 45 runs). It may carry into the reader's next myth (September +$0.07 per $,
  raw p 0.001 but not Holm-significant; frontier +$0.13, Holm-significant). Myths copy words
  and plans from each other without that detectably reaching behaviour.
- **Partners sharing norms** (label, moral summary, rule fields, giving score). No evidence
  alignment adds beyond each player's own level; one borderline September cell runs the
  opposite way.
- **The consistency drift.** Real in wording (Sonnet 8-agent homogeneous myth→game: keyword in
  5% of round-1 myths, 86% by rounds 8–10), mostly agents copying their own previous myth; no
  detectable effect on send or return levels within agents. One unconfirmed hint depends on how
  tests are counted: Sonnet's round-1 consistency wording goes with more cooperation over
  rounds 2–10 (+0.095; +0.110 in 8-agent mixed). If real, it would compete with "a generous
  opening sustains itself" as the second link.
- **About 150 myth features together.** After round 1 they add +0.000 held-out R² beyond past
  moves; no single feature adds 0.01 R² or more.
- **The judges are not the bottleneck.** They agree closely on amounts (send rule κ 0.91,
  stated amount r 0.997); the 3-way moral label (κ 0.54) mostly scores reciprocity, and sharper
  measures find the round-1 link it misses but are null exactly where it is null.

Not shown:

- **The chain across task orders.** No analysis tested "own myth → better first move → higher
  cooperation over the run" directly. Section 4 shows the first link and the opening gap; the
  second link rests on earlier results.
- **Game→myth versus game-only.** Game→myth does not beat game-only in general: the earlier
  data show no advantage in either regime (`findings-taskorder-myth.md`), and in the September
  runs it helps GPT (the zero-lock) and Sonnet dyads, not Sonnet populations or Gemini. Where
  it helps, the opening account does not cover it, since the first myth comes after round 1.
  A later-round version (the myth as a plan for the next round) is the guess, but within an
  agent the stated amount adds only $0.14 per $ beyond the last send (−0.03 to 0.31).
- **Which feature of the text.** The stated amount travels with the rest of a generous plan;
  the transplants suggest a rule-stated amount binds more than one that merely happens in a
  story, but rule wording and Sonnet authorship are confounded there.

Limits throughout: LLM judges without human validation; Gemini sends $5 almost always, so it
cannot show send effects; round-1 samples are small (62 senders).

## 7. The cheap test: edit-and-replay probes

The full seeding experiment on the September protocol (161 runs) would cost about $217. Most of
that pays for rounds 2–10. The mechanism claim concerns single decisions, and the model keeps
no state between calls, so a decision can be replayed exactly from the logged messages.

**Design.** Take the logged messages of real September calls, change one sentence, resend
with the run's pinned request settings, and record the send:

1. **Own myth, round 1** (myth→game): rewrite the amount in the agent's own myth to $5 / $3 /
   $1 / $0, or insert one. Tests whether the stated amount causes the first send.
2. **Read myth** (8-agent myth→game, round 2): rewrite the amount in the myth the agent was
   shown, leaving its own myth and history untouched. Tests the partner channel that
   observation could not.
3. **Own myth, later round** (game→myth): rewrite the amount in the myth written after round
   r, then replay the round r+1 decision. Tests the "plan for the next round" account of how
   GPT escapes the zero-lock in game→myth runs.
4. **Rule versus story:** the same amount as "send $X" versus "the traveller gave $X".

**Size and cost.** Sonnet round-1 sender calls in the September runs use about 840 input and
540 output tokens (thinking on); at $3 / $15 per million tokens (`analyses/_llm_judge.py`)
that is about $0.011 per call, a round-2 call about $0.019, and later-round calls $0.020–0.025.
Forty base contexts × 4 amounts × 3 slots × 2 samples ≈ 960 Sonnet calls ≈ $17; the
rule-versus-story arm, GPT-5 Nano and Gemini 3.7 Flash add a few dollars more. Budget about
$25, about one ninth of the full experiment.

**Free companion.** Decompose the existing task-order gap: within each composition, does
run-level cooperation still differ by task order once the round-1 send is held fixed? That
tests the second link of the chain on data we already have.

**What it cannot show.** Probes measure one decision, not how a changed opening plays out over
ten rounds; the free decomposition and the earlier zero-lock and founding-myth results cover
that part. Probe the exact runtime messages for refusals first: Sonnet has refused edited
text in its own-myth slot before.

# What in the myths predicts cooperation? Synthesis of six lenses, 2026-09-30

**Answer.** The judges are not the problem. We find no evidence that the norms in myths
drive cooperation. A myth detectably moves play in one place: an agent's *own* myth,
written before it has played, acts as its plan, and it sends what that myth says. A myth
read from a partner shows up in what the reader writes next, but has no detectable effect
on what it does. After round 1, no measured myth feature adds detectably to predicting
cooperation beyond the agent's past moves.

Scope: Sonnet 4.5 and GPT-5 Nano carry the send results; Gemini 3.7 Flash sends $5 almost
always, so it cannot show send effects.

**Data.** Five lenses: the 156 September informed negative-only myth runs (8,519 myths):
2-agent homogeneous 30 and mixed 36, 8-agent homogeneous 30 and mixed 60, both task orders.
Game-only, defector and frontier runs excluded. Causal lens: the slide-678 transplant
reruns (Sept 16–17, old apparatus: uninformed −$5 noise, myth-only memory, Sonnet hosts),
the same donors' June–July runs, and the May prompt-arm runs. **Spend:** $0.82 for a new
0–10 giving score on all myths by two judges (summed from the cost columns of
`judges/giving_scores_*.csv`) plus ~$0.01 for donor labels (lens report). No new runs.

## 1. Are the judges the problem? No; the 3-way label is the weak part (lens: judges)

- The two judges agree closely on amounts: send rule quadratic κ 0.91, stated amount
  r 0.997, the new 0–10 send score r 0.89. The moral label agrees at κ 0.54 (0.30 on
  Gemini). The lens's blind coding of 148 myths (itself an LLM, so a sanity check) matches
  the judges on send rule (κ 0.94) and send score (r 0.92), less on label (κ 0.43).
- The label mostly scores reciprocity: "send all five, return exactly half" is called
  "be fair". 99.5% of Gemini myths say "send all" (1,930 of 1,940), yet GLM splits those
  1,015 generous / 915 fair.
- Decisive test: the sharper measures find the one real link the label misses (round-1
  myth → round-1 send: send score +0.145 per SD, label −0.007), but they are null exactly
  where the label is null (within agent, and from the shown myth).
- Limit: within-agent reliability of the send score is 0.70; a within-agent effect up to
  0.075 send fraction per SD (≈ $0.38) is not excluded (`judges/noise_ceiling.csv`).
- The judge's "consistency" flag is as noisy as the label (κ 0.54; GLM 76% True vs
  DeepSeek 58%).

## 2. The one real link: the agent's own myth as its plan (lenses: search, amount, judges, causal)

- **Round 1, myth→game, before any play (R4), senders.** Sonnet and GPT senders whose myth
  names an amount send $0.67 per $1 stated (95% CI 0.55–0.79; 62 senders, 37 runs) and match
  it exactly 71% of the time (Sonnet 80%, $0.85 per $; GPT 55%, $0.48). Filling unnamed
  myths with their send-rule band gives $0.58 (150 senders). Stated amount and send rule
  add +0.25 held-out R² for round-1 sends. Gemini says $5 and sends $5 (50 of 50).
- **Why this is the text.** Verified on all 78 September myth→game runs (24 cells, 213
  round-1 senders): within a cell the system, myth and decision prompts are byte-identical,
  request settings are fixed per model, no seed is set, and only role + content reach the
  API. The model keeps no state between calls, so the agent's own myth is the only input
  to the send call that differs. The text is the channel; which feature of it is not
  identified. It is an instructed plan: the myth prompt asks how the game should be played,
  and the decision prompt says to take the myths into account. (Round-1 receivers also see
  the partner's noisy send, so this argument covers senders only.)
- **It fades.** The round-1 send rule predicts later sends weakly (+0.018 per level over
  rounds 2–10, p 0.023) and not at all over rounds 6–10 (+0.011, p 0.27), against ~+0.14 at
  round 1. In later rounds the stated amount equals the agent's last send 71% of the time;
  within an agent it adds $0.14 per $ (−0.03 to 0.31).
- **Causal check (transplants, R5; old apparatus, Sonnet hosts).** The donor text sat in
  the host's own myth slot; round-1 host replies mention "myth" in 80–100% of myth-seeded
  cells (0–10% in filler). The donor's stated amount orders the ladder: +$0.41 per $ in
  8-agent reruns (Holm-significant; +$0.29 without the send-nothing donor), +$0.15 on the
  original June–July apparatus; the dyad +$0.34 is not Holm-significant and falls to +$0.19
  without the send-nothing donor. Across the 23 donors that state an amount ρ 0.69 (Holm
  p 0.031); "send all" across all 30 donors ρ 0.68 (Holm 0.004). Within cells ρ 0.76 does not
  survive Holm. The moral label cannot order it: GLM calls 20 of 30 donors "be fair",
  including top and bottom cells. A rule-stated $5 binds far more than a $5 that merely
  happens in a GPT story; rule wording and Sonnet authorship are confounded.

## 3. The PI's three hypotheses

**H1 — aligned norms between partners → more cooperation: no evidence that alignment adds
beyond each player's own level** (lens: alignment). Four norm measures (same label,
moral-summary cosine, rule-field agreement, send-score agreement) plus whole-myth
similarity, fitted side by side with each player's own norm level and lagged moves, run ×
pair-family and round fixed effects. 30 primary tests; return proportion null in all 10.
Weak dyad hints: rule alignment → giving gap (Holm 0.044) halves to +0.009 (p 0.027) once
own send is controlled; same label → send +0.011 (p 0.007, Holm 0.19) vanishes with DeepSeek
labels. Dyads share history, so these cannot be separated from it. In 8-agent runs,
send-score alignment goes with *less* sending (−0.025), strongest in first meetings where
neither player had read the other's myth, and driven by the investor's score exceeding the
trustee's: a level effect, not similarity; it disappears with agent fixed effects.
Round 1 (before play): 0 of 60 survive Holm. Reverse direction, wording only: in 2-agent
game→myth, a full send makes the next two myths more alike in text (+0.076 cosine, Holm
0.025; 1 of 60 reverse tests), not in norms.

**H2 — shown myths naming a higher send → more cooperation: no detectable effect on play;
a suggestive carry-over into myths** (lenses: amount, search). 8-agent myth→game, reader's
next send per $1 named in the shown myth: +$0.01 (−0.09 to 0.11; 45 runs); averaging the
three shown myths in context +0.02; author ≠ current partner +0.04 (−0.09 to 0.17). The
upper bound (+0.11 per $) is about 0.17× the own-myth effect ($0.67). The search lens's
frozen candidates at the same rung: 15 of 15 null (all Holm 1.0). Into the reader's next
myth: +$0.07 per $ (0.03–0.12; raw p 0.001, not Holm-significant over the lens's tests;
unseen-myth placebo −0.03). A future-myth-controlled version (+0.14) is inflated by the
run × round fixed effect, which pushes any same-round myth negative, and is not used.

**H3 — the drift toward consistency: real; no detectable effect on cooperation levels;
stability hints unconfirmed** (lens: consistency). All three measures rise (keyword,
judge, judge-free embedding): Sonnet 8-agent homogeneous myth→game keyword share 0.05
(±0.07) → 0.86 (±0.14). GPT drifts too, more beside Sonnet (homogeneous 8-agent GPT 0.03 →
0.21). Mostly self-copying (a Sonnet agent uses the word 72% of the time if its previous
myth did, 34% if not; the prompt says to use the previous myth as inspiration), partly a
description of steady play. Within agents, no detectable effect on send or return levels
(CIs within ±0.03 for Sonnet and receivers; up to −0.08 to +0.03 for GPT senders).
Unconfirmed hints, none surviving Holm over ~540 tests: Sonnet judge-coded consistency →
smaller send changes (p 0.0002, judge measure only); keyword consistency pooled over
families → fewer cuts after a letdown (raw p 0.001, Holm 0.13); round-1 keyword consistency →
mean cooperation over rounds 2–10 (+0.062, raw p 0.002, Holm 0.15). Its spread fails the
future-myth control, as in August.

## 4. How much can myths add? (lenses: search, amount)

After round 1, adding ~150 myth features (rule fields, both labels, lexicons, sentiment,
style, embeddings, change vs previous myth), own and shown, gives no held-out gain over
past moves (CV grouped by run; the ridge penalty shrank them to zero, and gradient boosting
did no better than a noise block). What this bounds: the single-feature screen finds no
individual feature adding more than +0.005 R²; the full model can miss a single feature
worth 0.02 R², so a small signal is not excluded. A different design in the amount lens
shows small gains (own amount +0.016, shown amount +0.029), but an unseen placebo myth
gains +0.047 there, so that design picks up run-level information, not myth content.

## 5. What this means

Myths behave like an agent's own notes, not like norms that travel. Written before play,
an agent's myth is its plan and it follows it. After play starts, Sonnet's myths record what
just happened (a $1 higher send → $0.66 higher stated amount in its next myth; GPT +0.02,
null), and read myths are echoed in writing (words, perhaps stated amounts, the consistency
theme) without a detectable effect on behaviour. This fits the earlier findings: no moral
carryover within agents, and morals follow play.

## 6. What would change it, and the next experiment

- A seeding experiment (causal lens design, nothing launched): edit only the amount
  sentence ($5/3/1/0) in 7 fixed donor texts and place it in the agent's own slot or as the
  partner's myth; add rule-vs-story and a bare-rule control. 161 dyad runs, about $39 on the
  old apparatus (from the reruns' $8.49 per 35 runs); about $217 on the September protocol
  (from the pressure pilot's $27.07 per 20 runs), over the $200 gate. Probe the exact runtime
  messages for refusals first.
- A human pass on the 90-item coding sheet (settles "send all, return half").
- About twice the Sonnet runs to detect a within-agent effect of that size.

## Caveats

LLM judges, no human validation; Gemini at the send ceiling; frontier runs excluded; R4
samples small (62 senders, 37 runs); transplant evidence is old-apparatus Sonnet; in 8-agent
myth→game the shown myth is always last round's partner's, so "never played the reader" is
approximated by "had not played before writing" and by controlling that author's last move.
One lens file (`judges/r4_plan_check.csv`) has no generating script and a duplicated
subset; nothing here rests on it.

## Files

`BRIEF.md` is the shared brief every lens followed. Each lens folder holds its scripts and
the CSVs/PNGs under 2 MB; larger intermediates (decision tables, feature tables, pickles,
embeddings, judge caches) are in the gitignored `data/analysis/myth_predictors_20260930/`.
The lenses ran from a scratch directory and read `data/analysis/linguistic_20260923/`;
some scripts hard-code their scratch output path, so point their folder constants at this
directory before rerunning. The lens reports went to the lead by message (subagents cannot
write report files); this README is the checked synthesis of them.
Independent verification of this synthesis: 2026-09-30 (subagent; scripts verify_fp.py,
verify_msgs.py).

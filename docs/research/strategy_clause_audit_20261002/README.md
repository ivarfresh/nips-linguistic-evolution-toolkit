# Strategy clauses evolve, and some appear to travel

2026-10-02. **Exploratory machine-coded audit; human validation pending.**

The richer reading changes the narrow interpretation of the existing data:
“agents do not transmit rules” is too strong. We found concrete candidate
examples of peer-to-peer advice uptake and evolving conditional rules.
However, this audit does **not** establish that those clauses changed play,
spread through multiple network hops, or explain the cooperation effect.

## What was actually done

The [plan](PLAN.md) fixed an outcome-independent sample before reading its
trajectories: one complete agent trajectory per tier × family × size × mixing
× task-order cell. The original September and main-frontier corpora contribute
24 trajectories each: **48 trajectories, 480 myths, 45 unique completed runs**.
The later September extension, frontier-update models, injection experiments,
and message-board pilot are outside this audit. Mixed compositions are not
equally represented within the single mixed cell; this is a balanced diagnostic
sample, not a population prevalence estimate.

Three GPT-6-Sol readers each read 16 complete trajectories, then consulted
actual exposure, one matched unseen myth, and game context for their selected
events. The lead separately read five complete trajectories (T17, T24, T37,
T40, T48), checked selected source comparisons, and requested corrections.
This is not independent double-coding of all 48 trajectories.

Each reader selected at most two salient changes per trajectory. **44 selected
events are examples, not an exhaustive event census.** The scripts verify all
480 own texts against final full-state JSONs and all 432 peer exposures against
the accepted myth responses' saved prompts. All 432 unseen controls are from
different runs and match tier, author family, round, size, mixing, order and
composition. All 176 recorded quotations match their source strings. One
round contains an additional malformed logged attempt; the accepted response
matches the final myth. These checks establish provenance and quotation
accuracy, not the validity of the interpretations.

## What the texts show

### 1. Some specific advice appears to be taken up from peers

**T40, frontier Opus, mixed eight-agent population, myth before game:**

- Exposed peer, round 6: “And if one comes to you already cheated, overpay.
  That is how the bridge repairs itself.”
- Reader's new myth, round 7: “And when one arrives already cheated, overpay.
  That is how bridges repair themselves.”

The compensation clause is absent from this reader's own rounds 1–6 and the
matched unseen round-6 myth. It persists in the reader's rounds 8 and 9,
then is not restated in round 10. This is a strong **trace-level candidate
for textual transmission of a conditional norm**, not merely a shared
general topic. A single unseen comparator cannot eliminate common model
priors or shared game experience. Persistence in the same author is not a
second network hop.

**T34, frontier Opus, homogeneous dyad, myth before game:** the exposed
round-5 text says “you may draw back, slowly, and sorrowfully, and with the
door left open”; the reader's round-6 text adds “Even then, when you draw
back, draw back slowly, sorrowfully, and leave the door open”. Earlier own
myths advise judging a season but do not explicitly prescribe withdrawal;
the matched unseen myth lacks this combination. This is another candidate
for uptake of an actionable clause, though the duration and amount remain
unspecified.

These examples suggest why broad moral categories and exact numerical
endpoints can miss meaningful textual changes. They do not invalidate earlier
tests: those tests asked narrower questions.

### 2. Strategies change in more than one direction

**T23, September Sonnet, mixed population, game before myth:** a general
recommendation to reward courage develops into a return schedule. Round 8
contrasts half returned for a three-unit send with about sixty percent for
near-five sends. Round 10 fills in more levels: 30%, 40%, 50%, 55%, 57%,
and at least 60%. The broad principle already existed in its own stories;
the exposed peer supplies related graded advice. The exact rates cannot be
uniquely attributed to that peer. The percentages are prescriptions in text,
not observed behavioral estimates, and their accompanying fairness arithmetic
should not automatically be accepted as correct.

**T37, frontier Opus, homogeneous population, game before myth:** round 8
adds “If a hand is dry three times running, give less — and grieve. And open
again the moment it opens.” The exposed peer contains reduce-and-reopen
advice, but the unseen comparator also says “Forgive twice, not thrice —
then pour less, never nothing.” This illustrates an identifiable textual
change without an identifiable source. The numerical trigger is not restated
in round 10: complexity does not simply accumulate.

Counterexamples matter. T17 revisits different return amounts rather than
building a consistent conditional policy. T24 moves toward a fixed personal
commitment despite changing partners. T48 already has cautious withdrawal and
reconciliation in round 1, so later recovery imagery is not a newly invented
forgiveness strategy. Several Gemini trajectories retain essentially the same
full-send, half-return advice throughout.

For transparency, the provisional single-label trajectory summaries are:

| Machine label | September (24) | Main frontier (24) |
|---|---:|---:|
| Elaboration | 5 | 14 |
| Revision | 5 | 4 |
| Simplification | 0 | 1 |
| Stable | 13 | 5 |
| Unclear | 1 | 0 |

These are **unvalidated reader judgments**, not effect sizes or a tested
frontier-versus-September result. The categories compress mixed paths:
“elaboration” can include a temporary addition that is later dropped.
Different model profiles, text styles, and a one-trajectory-per-cell sample
preclude a reasoning-capability conclusion. No pooled significance test was run.

### 3. Written contingencies are not evidence of enacted strategies

The plan allowed only two prospective checks: reduced sending after repeated
poor reciprocation, and continued sending after a single explicitly
noise-attributed shortfall. Inspection did not yield a defensible pooled
behavioral test in this bounded audit. **This is an eligibility assessment of
selected examples, not a complete count of eligible events across the corpus.**

- T37's round-8 three-dry rule has just one remaining sender opportunity,
  round 9. The current partner's three displayed returns are $8.50, $7.53
  and $7.10, not empty returns. It sends $5, but that does not exercise the
  new punishment branch; “dry” also lacks an exact numeric definition.
- T32's round-8 story describes reducing five to one after repeated clear
  hoarding. Its decision prompt instead shows returns of $7.50, $7.38 and
  $6.92. Those observations do not instantiate the story's clear-hoarder
  premise. Its exact reduced amount is nested inside a narrative and still
  requires human judgment about endorsement.
- T34 leaves “a season” undefined. Converting it to three logged rounds
  would add an analyst-created rule. Rotating partners further complicate
  what should count as a repeated interaction history.
- Forgiveness language often mentions generic fog or wind, not an attribution
  to one specific observed shortfall. A generic forgiveness clause followed by
  another generous send is not a clean test of a newly adopted mechanism.
- T40's compensation clause concerns receiver behavior, outside the two
  predefined checks. Its next receiver turn returns $7.30 from a displayed
  $13.47 receipt. That is compatible with ordinary above-half returning;
  it does not identify additional compensation for prior cheating.

Game-before-myth outcomes cannot be consequences of that same round's new
myth. Myth-before-game writing can precede the same round's choice. The
generated context packets preserve this timing and the actual decision
prompts. Undistorted transfers must not replace what an agent could see.

## What this means for the paper

A defensible **provisional** interpretation is:

> Agents sometimes incorporate specific advice from peers into their evolving
> narratives. Such uptake can involve conditions, thresholds and recovery
> rules rather than literal copying of a fixed norm. Whether these textual
> changes alter behavior or account for the cooperation effect remains open.

Do not yet turn the audit counts into a paper result. If human readings
confirm the examples, a small qualitative panel could distinguish **textual
uptake**, **retention by the reader**, and **behavioral enactment**. This is
more accurate than either “norms do not transmit” or “moral norms diffuse and
cause cooperation.” A new design is not required to see textual changes;
stronger causal claims still need appropriate interventions and controls.

## Validation, limitations and next gate

- [HUMAN_VALIDATION.md](HUMAN_VALIDATION.md) contains 12 complete trajectories
  selected by hash, one per tier × family × size, without machine labels,
  model identifiers or outcomes. Textual style can still reveal the model.
  It includes stable cases; selection ignores the coding. The separate
  [key](human_validation_key.json) should remain unopened while coding.
- This packet tests extraction and trajectory interpretation only. Attribution
  would additionally need blinded, shuffled own/exposed/unseen comparisons;
  one selected unseen myth is not a randomized exposure control or a calibrated
  semantic baseline. Human checks of the highlighted examples are also needed
  before publication.
- Reconcile disagreements about endorsement, novelty, metaphor and category
  boundaries before scaling. Three disjoint model readers do not establish
  inter-rater reliability; exact-string validation is not human agreement.
- This audit does not quantify transmission frequency, compare intervention
  effects, trace complete network lineages, or adjudicate the pilot's registered
  null. It neither demonstrates nor rules out behavioral cultural transmission.

Stop here pending human validation. No experiment, prompt, runtime code,
existing source data, or paid API batch was changed or launched.

## Reproduction and artifacts

From the repository root:

```sh
python3 analyses/strategy_clause_audit.py
python3 analyses/strategy_clause_audit.py --finalize
python3 analyses/strategy_clause_audit.py --context T40 7
```

The first command rebuilds local full-text/context packets under
`data/analysis/strategy_clause_audit_20261002/` and the deterministic manifest.
The second verifies source hashes, quotations, exposure identities and accepted
saved prompts, then exports the summaries and human packet. It does not
recreate the interpretive coding, which is preserved in
[coding_1.json](coding_1.json), [coding_2.json](coding_2.json) and
[coding_3.json](coding_3.json).

See [manifest.json](manifest.json) for exact final-state paths, hashes and
sampling strata; [validation.json](validation.json) for checks/counts; and
[TRAJECTORIES.md](TRAJECTORIES.md) for all 48 assessments, including stable
and unclear cases. All reported counts are descriptive counts, not means.

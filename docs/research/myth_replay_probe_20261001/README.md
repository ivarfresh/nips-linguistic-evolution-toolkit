# Edit-and-replay probe: a send rule in an agent's own myth sets its next send

*2026-10-01. Tests the claim in `docs/research/myth_opening_plan_20260930/README.md` (PR #17)
that an agent's own myth works as its plan.*

## The answer

When we add one sentence to an agent's own myth, *"whoever holds five should send X"*, its next
send follows X almost one for one: **$0.93 per $1** in its first myth and **$0.68** in a myth it
wrote after round 1 (Sonnet and GPT pooled). An amount told inside the story moves sends less
(Sonnet: $0.57 and $0.24 per $1). The same rule sentence in a partner's myth the agent has read
moves its send **$0.23 per $1**, a lower bound for that channel (see "Arm B" below). So a stated
plan in the agent's own myth drives its next decision; these are single decisions, not whole runs.

## How it works

Every September run saved the exact text each model saw before each decision. The models keep
no memory between calls, so sending that same text again recreates the decision: a **replay**.
We replayed real decisions after changing one thing, the amount in a myth the agent could see.
If the send moves with the amount, the change caused it, because nothing else differed.

**The edit.** The main edit appends one fixed sentence to the myth, changing only the number:
*"And so the elders taught: whoever holds five should send {one / two / three / all five} of
the five."* A secondary edit (Sonnet only, myths that already name an amount) rewrites that
amount inside the story ("gave five coins" → "gave one coin") with an editor model, checked to
change little else. Each context was also replayed unedited.

**Where the edited myth sits.**

| Arm | Myth | Decision replayed |
|---|---|---|
| A | the agent's own myth, written just before its first send | round-1 send, myth→game |
| B | the myth the agent read from its previous partner | round-2/3 send, 8-agent myth→game |
| C | the agent's own myth written after the previous game | round-2/3 send, game→myth |

**Contexts and samples.** The plan drew 40 contexts per model and arm from the 156 no-defector
September myth runs (single-model and mixed; each agent replayed with its own recorded request
settings and the same model snapshot), with a fixed seed. Samples per version: Sonnet 2, GPT 4,
Gemini 1. Calls ran in a seeded shuffled order. We stopped the run at 1,359 of 4,720 planned
replays (29%) after looking at interim results, because arms A and C were already far beyond
their thresholds; the completed rows are the first 1,359 of the shuffled order, so every cell is
26–34% complete. Analysed contexts: Sonnet 36–38 and GPT 40 per arm, Gemini 24–29.

**Analysis.** Intent to treat: every edit counts, whatever a judge model later read in it. For
each arm, the slope of the replayed send on the edited amount, comparing versions of the same
context only (context fixed effects), SEs clustered by run, t inference. Primary test per arm:
rule edits, Sonnet and GPT pooled; Holm over the three arms. The decision rules (A and C confirmed
if the slope is at least $0.30 with the CI above 0; B "no effect" only if its CI stays below
$0.20) were set in the design discussion before the pilots and committed to the repo at launch.

**Checks.** Four pilots ($6.75) fixed the design: replay noise on identical input (Sonnet sd ≈
$0.4, GPT ≈ $1.1, Gemini 0), refusals (none in 780 edited replays), the edit pipeline (a
free-form editor was unreliable, so the fixed rule sentence became primary, and keeping only
edits a judge reads as intended would bias the comparison, so every edit counts), and the
placement of edits in arms B and C. Main run: 0 errors, 0 parse failures, 0 refusals; a judge
reads the intended amount in 95–100% of Sonnet and GPT rule edits. Unedited replays land close to
the logged sends (Sonnet arm A $4.07 vs $3.89; GPT arm A $3.00 vs $2.83); the largest gap, GPT arm
C ($1.89 vs $2.13), is within GPT's replay noise (`unedited_control.csv`).

## Results

Send per $1 stated (`slopes.csv`; 95% CI; Sonnet + GPT, rule edits):

| Arm | Pooled | Sonnet | GPT | Verdict |
|---|---|---|---|---|
| A own first myth | **0.93** (0.88–0.99) | 0.87 | 0.96 | confirmed |
| B partner's myth | **0.23** (0.12–0.34) | 0.13 | 0.28 | inconclusive (real, small) |
| C own later myth | **0.68** (0.55–0.80) | 0.60 | 0.70 | confirmed |

All three primary slopes stay above zero after Holm. Restricting to edits a judge read as
intended changes nothing. Two robustness checks (`robustness_8agent_and_weighting.csv`): on
8-agent contexts only, A 0.91 and C 0.61; weighting each context × amount version equally
(GPT has twice Sonnet's samples), A 0.92, B 0.20 (0.10–0.30), C 0.68. Verdicts unchanged.

**Arm B is a lower bound, not a clean contrast with A.** In arm B the agent's own myth, written
right after it read the partner's original myth, stays in context unedited, so the replay
measures only the direct path from the read myth to the send. B also differs from A and C in
where the myth sits (a user message, two messages before the decision) and uses only 8-agent
runs. What B shows is that reading a myth with a stated rule moves the next send a little; our
earlier conclusion that read myths do not move play is revised to "move it a little, at least".

**Rule versus story.** For Sonnet, amounts rewritten inside the story move sends $0.57 per $ on
the first myth (rule $0.87), $0.24 on later myths (rule $0.60; difference −0.47, CI −0.71 to
−0.22) and $0.06 in arm B (CI −0.03 to 0.15) (`rule_vs_natural.csv`). Most September myths state
their amount the story way, which may be why the observational effects were weaker.

**Gemini 3.7 Flash** sends $5 in nearly every replay whatever its myth says (lowest cell mean
$4.43, arm A at "three", n = 7); its slopes are not informative and are not shown as CIs.

## Does a good opening explain the task-order gap? (free check, existing runs)

The September story-first runs send more in rounds 2–10 than game-first runs. Holding each run's
round-1 send fixed (`opening_mediation.csv`, composition × size fixed effects):

| Runs | Story-first advantage | With round-1 send held fixed |
|---|---|---|
| single-model (60) | +$0.42 (0.06–0.77) | +$0.17 (−0.14 to 0.48) |
| mixed-model (96) | +$0.49 (0.22–0.75) | +$0.66 (0.32–1.00) |
| all (156) | +$0.46 (0.25–0.67) | +$0.49 (0.24–0.73) |

In single-model populations the advantage shrinks when the opening is held fixed, which fits a
good opening carrying forward, but the drop is not tested and the intervals overlap. In mixed and
pooled runs it does not shrink. Holding the round-1 send fixed is also an imperfect test, because
task order affects the round-1 send itself. So the opening may explain part of the gap in
single-model populations; what drives it in mixed populations is open.

## What this does not show

- Probes measure single decisions, not whole runs.
- The headline effect is for an explicit rule sentence; amounts told inside the story move sends
  less.
- Arm B holds the agent's own reaction myth fixed, so it understates the read-myth channel.
- Gemini cannot show an effect on sending.
- The stop was decided after seeing interim results; this does not affect A and C, but B's
  interval would narrow with the remaining replays (`python3 analyses/myth_replay_probe.py
  --stage main` resumes the run).

## Cost

Estimated from token counts at repo list prices (Gemini 3.7 Flash has no repo price; Gemini 2.5
Flash's is used): pilots $6.75, main run $8.99, total about $15.75 (planned $35.78 for the full
main run). Billed amounts appear in the grant budget tool.

## Files

`analyses/myth_replay_probe.py` (pilots and main run), `analyses/myth_replay_analysis.py`
(this analysis). Here: `slopes.csv`, `verdicts.csv`, `rule_vs_natural.csv`,
`robustness_8agent_and_weighting.csv`, `mean_send_by_amount.csv`, `manipulation_check.csv`,
`unedited_control.csv`, `opening_mediation.csv`, `replay_send_by_amount.png`, `replay_boxes_by_amount.png`, `replay_rule_vs_story.png`, `opening_runs.png` (`analyses/myth_replay_plots.py`). Raw replays
(gitignored): `data/analysis/myth_replay_probe_20261001/`. The edited text of natural edits in
the completed main run was not stored; rows written by a resumed run store it.

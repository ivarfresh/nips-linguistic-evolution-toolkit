# Edit-and-replay probe: the amount a myth states causes the next send

*2026-10-01. Tests the claim in `docs/research/myth_opening_plan_20260930/README.md` (PR #17)
that an agent's own myth works as its plan.*

## The answer

When we change only the amount an agent's own myth says to send, the agent's next send follows
it almost one for one: **$0.93 per $1 stated** in its own first myth and **$0.68** in a myth
it wrote after round 1. A partner's myth that the agent reads moves its send far less, **$0.23
per $1**. So the myth causes the send, and mostly when it is the agent's own.

## How it works

Every September run saved the exact text each model saw before each decision. The models keep
no memory between calls, so sending that same text again recreates the decision: a **replay**.
We replayed real decisions after changing one thing: the amount in a myth the agent could see.
If the send moves with the amount, the amount caused it, because nothing else differed.

**The edit.** The main edit adds one fixed sentence to the end of the myth, changing only the
number: *"And so the elders taught: whoever holds five should send {one / two / three / all
five} of the five."* A secondary edit (Sonnet only) rewrites the amount the myth already names
("gave five coins" → "gave one coin") with an editor model, checked to change no more than the
amount. Each context was also replayed unedited.

**Where the edited myth sits.**

| Arm | Myth | Decision replayed |
|---|---|---|
| A | the agent's own myth, written just before its first send | round-1 send, myth→game |
| B | the myth the agent read from its previous partner | round-2/3 send, 8-agent myth→game |
| C | the agent's own myth written after the previous game | round-2/3 send, game→myth |

**Contexts and samples.** 40 contexts per model and arm from the 156 no-defector September myth
runs (single-model and mixed; each agent replayed with its own recorded request settings),
sampled with a fixed seed. Planned samples per version: Sonnet 2, GPT 4, Gemini 1. Calls ran in
shuffled order. The run was stopped at 1,359 of 4,720 planned replays (29%) once the primary
answers were clear; because of the shuffle, the completed part is a random subset of every cell.

**Analysis.** Intent to treat: every edit counts, whatever a judge model later read in it. For
each arm, the slope of the replayed send on the edited amount, comparing versions of the same
context only (context fixed effects), with SEs clustered by run (t inference). Primary test per
arm: rule edits, Sonnet and GPT pooled; Holm over the three arms. Decision rules were fixed
before the run.

**Checks.** Four pilots ($6.80) fixed the design before the main run: replay noise on identical
input (Sonnet sd ≈ $0.4, GPT ≈ $1.1, Gemini 0), refusals (none in about 730 edited replays), the
edit pipeline (the free editor was unreliable, so the fixed rule sentence became primary; keeping
only edits that pass the judge would bias the comparison, so every edit counts), and the
placement of edits in arms B and C. In the main run: 0 errors, 0 parse failures, 0 refusals; a
judge reads the intended amount in 95–100% of Sonnet and GPT rule edits; the unedited replays
reproduce the logged sends (e.g. GPT arm A $3.00 replayed vs $2.83 logged, Sonnet $4.07 vs
$3.89).

## Results

Send per $1 stated, rule edits (`slopes.csv`; 95% CI):

| Arm | Sonnet + GPT (primary) | Sonnet | GPT | Gemini | Verdict |
|---|---|---|---|---|---|
| A own first myth | **0.93** (0.88–0.99) | 0.87 | 0.96 | 0.00 (always $5) | confirmed |
| B partner's myth | **0.23** (0.12–0.34) | 0.13 | 0.28 | 0.02 | inconclusive |
| C own later myth | **0.68** (0.55–0.80) | 0.60 | 0.70 | 0.18 (−0.60 to 0.96) | confirmed |

All three primary slopes are above zero after Holm. Arm B's rule was "no effect confirmed" only if
its CI stayed below $0.20; it does not, so reading a partner's myth has a real effect, about a
quarter of the own-myth effect. Restricting to edits the judge read as intended changes nothing.

**A stated rule moves sends more than an amount told inside the story.** For Sonnet, natural
edits give $0.57 per $ on the first myth (rule $0.87) and $0.24 on later myths (rule $0.60); the
later-myth difference is clear (natural minus rule −0.47, CI −0.71 to −0.22; `rule_vs_natural.csv`).

**Gemini 3.7 Flash sends $5 whatever its myth says**, including "send one"; the judge often still
reads such myths as "give all five", since the rest of the myth preaches it.

## Does a good opening explain the task-order gap? (free check, existing runs)

The September story-first runs send more in rounds 2–10 than game-first runs. Holding each
run's round-1 send fixed (`opening_mediation.csv`, composition × size fixed effects):

| Runs | Story-first advantage | With round-1 send held fixed |
|---|---|---|
| single-model (60) | +$0.42 (0.06–0.77) | +$0.17 (−0.14 to 0.48) |
| mixed-model (96) | +$0.49 (0.22–0.75) | +$0.66 (0.32–1.00) |

In single-model populations about 60% of the gap goes with the opening. In mixed populations
none of it does, so a good opening is part of the explanation, not all of it. What else carries
the gap in mixed populations is open. Arm C shows that later myths also act as plans, but agents
write a myth before every game after round 1 in both task orders, so that alone does not explain
why story-first runs stay ahead. These are 60 and 96 runs, and the comparison is observational.

## What this does not show

- Probes measure single decisions, not whole runs.
- The rule sentence is explicit by design; most September myths state their amount less
  directly, which may be why natural edits move sends less.
- Gemini cannot show an effect on sending.
- The run was stopped early; arm B's interval would narrow with the remaining replays
  (`python3 analyses/myth_replay_probe.py --stage main` resumes it).

## Cost

Estimated from token counts at repo list prices: pilots $6.80, main run $8.99, total about
$15.80 (planned $35.78 for the full main run).

## Files

`analyses/myth_replay_probe.py` (pilots and main run), `analyses/myth_replay_analysis.py`
(this analysis). Here: `slopes.csv`, `verdicts.csv`, `rule_vs_natural.csv`,
`mean_send_by_amount.csv`, `manipulation_check.csv`, `unedited_control.csv`,
`opening_mediation.csv`, `replay_send_by_amount.png`. Raw replays (gitignored):
`data/analysis/myth_replay_probe_20261001/`.

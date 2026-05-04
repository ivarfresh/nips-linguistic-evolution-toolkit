# A1 + A3 experiment package — GPT-5-Nano only

Two manipulations to test whether tighter prompt-level coupling between
the myth channel and the game channel translates the linguistic-coupling
findings (§3.4–3.5) into stronger behavioural effects.

Scope: **GPT-5-Nano only** (focus per Aron, May 1).

## What's in this folder

| File | Purpose |
|---|---|
| `README.md` | This file |
| `trust_game_noisy.partner_myth.patch` | Unified diff for `games/trust_game_noisy.py` — adds `{other_agent_last_myth}` placeholder support |
| `experiments_noisy.append.yaml` | YAML snippet to paste into `config/experiments_noisy.yaml` (new prompt templates + new experiment_sets) |
| `run_A1_A3.sh` | Runner commands |
| `analysis_hook.md` | How to fold the results into the existing analysis pipeline once runs complete |

## What A1 and A3 are

### A1 — Direct partner-myth injection
The current paradigm keeps the cross-task channel thin (per §3.3): the
partner's myth reaches the game-side computation only via the agent's
own prior myth-writing prompt. **A1 makes the channel structurally
open** by quoting the partner's most recent myth verbatim inside the
trust-game prompt. If language is load-bearing on cooperation, this
intervention should produce a noticeably stronger effect than the
implicit-channel baseline.

### A3 — Forced reasoning prose
GPT-5-Nano currently emits only the JSON action (e.g. `{"send": 3}`)
with no surrounding prose, which makes the §4.5 reason-coding analysis
methodologically undecidable for that model. **A3 requires GPT-5-Nano
to emit 2–3 sentences of reasoning before the action.** This both
(a) closes the §4.5 blind spot and (b) tests whether the
own-myth-vocabulary carryover finding (78–82% in Claude, 0% in
nano-as-currently-configured) is a Claude-specific phenomenon or
a configuration artefact.

## Headline cells (60 + 120 = 180 runs total)

| Manipulation | Cells | Seeds | Total runs |
|---|---|---|---|
| A1 partner-myth injection | gpt5-nano × {positive, positive_informed, negative_5, bootstrap} × {game_myth, myth_game} | 15 | ~120 |
| A3 forced reasoning prose | gpt5-nano × {positive, positive_informed, negative_5, bootstrap} × {game, game_myth, myth_game} | 10 | ~120 |

Both can run overnight Friday → Saturday on the same direct-provider
configuration as the existing v4 corpus.

## Discriminating predictions

For A1:

- **If injection lifts cooperation (Δmean > existing variance ratio),**
  the linguistic channel was bottlenecked by structural opacity, not by
  weak content. Strong evidence for the load-bearing-language hypothesis.
- **If injection doesn't lift cooperation,** the linguistic-coupling
  findings (§3.4–3.5) really are an isolated linguistic side-channel
  with no behavioural transmission. Cleaner negative result; pivots the
  paper firmly toward the linguistic-coupling story.
- **If injection destabilises (variance up, mean flat or down),** the
  partner's myth introduces a competing prior similar to the bootstrap-
  noise destabilisation pattern, supporting the §5.2 mechanism story.

For A3:

- **If forced reasoning makes nano emit own-myth-vocabulary references
  at Claude-like rates (~80%),** the §4.5 finding was real but invisible
  in the JSON-only configuration. The cross-task linguistic carryover
  generalises across models.
- **If forced reasoning produces low rates (<30%) even though prose is
  present,** the carryover IS Claude-specific — interesting in its own
  right and a substantive cross-model variation finding to add to §5.3.

Both outcomes are reportable, neither is a failure mode for the paper.

## What's deferred

- A1 for Claude (would amplify the lift+consolidation cells; ~60 runs).
  Skip for now per Aron.
- A2 self-myth recall, A4 targeted-myth × Claude.
- All Tier-B variants (myth-as-norm, scaffolding, adversarial content,
  bidirectional visibility) — EMNLP follow-up.

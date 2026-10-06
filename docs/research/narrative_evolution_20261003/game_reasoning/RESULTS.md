# Claude game reasoning: which myth patterns exist without myths?

2026-10-05. 10 Claude reader agents; no API spend.

## What was read

- **Scope.** The written game reasoning of 1,333 Claude agents, informed negative noise only:
  - every game-only run with a Claude agent
  - every Claude agent in the 70% myth-run sample
  - scripted defectors excluded (their moves are forced and carry no text)
- **Blinding.** Units were shuffled under random IDs, so readers did not know whether a unit came from a game-only run or a myth run.
- **Quotes.** All 3,634 quotes match the source response word for word.
- **Code.** Preparation: `analyses/narrative_game_reasoning_prepare.py`. Tally: `analyses/narrative_game_reasoning_tally.py`, which writes `summary.json`.

**Text availability:**

| Model | Game-only | Game→Myth | Myth→Game |
|---|---|---|---|
| Sonnet 4.5 | ~100% | ~100% | ~100% |
| Opus 5.5 | 100% | 100% | 100% |
| Opus 5 | 18% | 21% | 83% |

Opus 5 mostly answers with bare JSON, so its comparison is weak.

**Not like-for-like.** Game-only and myth runs differ in more than the myth text:
- Myth-run game prompts add "take any myths into account".
- Game-only populations cooperate less, so agents have more low senders to react to.

## Themes that appear only, or mostly, when the agent writes myths

| Theme in game reasoning | Sonnet, game-only (8-agent, no defection) | Sonnet, myth runs | Opus 5.5, game-only | Opus 5.5, myth runs |
|---|---|---|---|---|
| Reputation ("the network remembers") | 3% | 49–60% | 2% | 2% |
| The sender deserves more / return more than half | 0% | 24–60% | 5% | 89–91% |
| Someone must go first | 3% (all) | 27–30% (all) | 0% | 10–29% |
| Word–deed gap | 0% (all) | 8–13% (all) | 0% | 0–2% |
| Counted forgiveness | 0% (all) | 5% (all) | 0% | 2–4% |
| Luck vs intent | 10% | 23–29% | 62% | 93–100% |

- **Created by myth writing:** reputation, the sender deserves more, someone must go first, word–deed gap and counted forgiveness are essentially absent from game-only reasoning. They appear once the agent writes myths.
- **Amplified, not created:** luck vs intent is already Opus 5.5's default (62% game-only) and rises to 93–100% with myths.
- **Agents cite their myths:** "references a myth" appears in 98–100% of Sonnet and Opus 5.5 myth-run reasoning, and in 71% of Opus 5 Myth→Game reasoning.

## Punishment exists without myths, but changes form

| Sonnet 4.5 game reasoning | Game-only | Game→Myth | Myth→Game |
|---|---|---|---|
| States a sanction, 8-agent, no defection | 48% | 21% | 2% |
| States a sanction, 8-agent, scripted defectors | 96% | 100% | 100% |
| Total exclusion, scripted defectors | 12% | 40% | 43% |
| Graded, never to zero, scripted defectors | 0% | 27% | 10% |
| Sanction in dyads | 40% | 19% | 12% |

**Sonnet** already reasons about punishing in game-only play, and does so *more* often there than in myth runs. With myths, two things change:
- Without defection, sanctioning in its moves drops, most of all when the myth comes first.
- Against defectors, the form changes: the graded "never zero" wording appears (0% → 10–27%), and so does total exclusion (12% → 40–43%). Both mirror Sonnet's myths.

Within agents, the myth's sanction shows up in the same agent's game reasoning:

| Sanction in the agent's myths | Game reasoning also states a sanction |
|---|---|
| Total exclusion | 35/49 |
| Measure for measure | 18/26 |
| Withdraw (floor unspecified) | 35/107 |
| Graded, never to zero | 22/54 |
| None | 6/254 |

**Opus 5.5** almost never states a sanction in game reasoning, in any condition (0–6%). The withdrawal its myths prescribe (31%) does not appear in its game text (0/32 agents).

**Opus 5:** the graded "send less, never nothing" rule in its myths (76%) almost never appears in game text (4/145 agents). This is because Opus 5 rarely writes reasoning. The behaviour check (`analyses/narrative_defector_play_check.py`) does show the rule in play: with myths, Opus 5 sends a known defector $0 in 1–2% of decisions, vs 50% game-only.

## Bottom line

- **Without myths, Claude models already:** punish low cooperators (Sonnet, often bluntly) and blame noise (Opus 5.5).
- **Only with myths:** a reputation framing, a norm that the sender deserves more than half, a someone-must-go-first ethic, counted forgiveness, and graded or exclusionary *rules*.
- **These myth patterns enter the agents' game reasoning,** with agents citing their own myths.

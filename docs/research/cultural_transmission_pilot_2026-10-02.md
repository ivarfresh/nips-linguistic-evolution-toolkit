# Cultural transmission pilot (plan, 2026-10-02)

Status: **planted arm run (n=3, $18.42): the rule did not spread by the registered tests; control
arm built but not run.** Rule: the Velmar Rule, "whoever holds five should send two of the five".
Decision rules, results and the independent review are in
`docs/research/cultural_transmission_pilot_20261002/` (`PREREGISTRATION.md`, `README.md`,
`ASTRA_REVIEW.md`). This file is the plan as written before the run.

## Question

Can we establish that a rule one agent writes into a myth spreads to other
agents, so that it shows up in their myths, their play, and their retellings?
This pilot only asks whether spread can happen. It does not ask which design
feature causes it, so every lever is set to favour spread.

## Why a new design

Earlier work found that stories steer the writer strongly and the reader a little,
and that nothing is passed on beyond the reader:

- An agent's own myth sets its next send ($0.93 per $1 in its own first myth); a
  read partner myth moves it $0.23 (`docs/research/myth_replay_probe_20261001/README.md`).
- No observational norm transmission survives a placebo or rewiring null
  (`researchlog.md`, 2026-08-28 and 2026-09-30 entries).
- The frontier board and saboteur pilots (2026-10-02) found nothing, but the frontier
  models already send about $4.81 of $5 to each other, so there was no room to move.

The main feature of the current design that works against spread is that each myth
reaches only one partner. (Self-copying is the strongest effect we have measured, but the
"use your previous myth as inspiration" instruction does not cause it; the own myth in chat
memory is the likelier source, untested. See the self-anchor replay.)

## Design

- **Base:** September Sonnet 4.5 8-agent population, myth → game, informed
  negative-only noise, pinned `llm_settings` (as in
  `negative_only_reasoning_rerun_population_myth_game_claude_n5`, `config/experiments_noisy.yaml:5326`). Sonnet opens at
  about $3–4, so it has room to move either way; Opus 5 sits near the $5 ceiling
  and costs more ($4.01 vs $3.41 per 8-agent myth → game run).
- **Shared board on:** every agent reads every earlier myth while writing its myth
  (`myth_board: persistent`, already on main). After the myth call the board is replaced in
  chat memory by a one-line note, so agents do not see it when they decide a send.
- **Prompt wording kept as in September.** Removing "Use the myth you wrote in the previous
  round as inspiration, but adapt it in your own way" was planned, but the 2026-10-02 replay
  showed it does not reduce self-copying and slightly lowers borrowing from other myths
  (−2.0 pts, `docs/research/self_anchor_replay_20261002/README.md`), so the line stays.
- **Plant:** Agent_1's round-1 myth is fixed text, not generated: a real September
  Sonnet round-1 myth plus one named rule sentence that Sonnet rarely follows on its
  own: "And so the elders taught the Velmar Rule: whoever holds five should send two of
  the five." Sonnet never sent $2 in the September 8-agent runs (lowest $2.50), and
  "Velmar" / "two of five" appear in 0 of 800 September myths ("stone" appears in 129, so
  the earlier "Two Stones" draft was dropped).
- **Control:** the same myth with a named teaching of the same form and no amount
  ("the Velmar Song: whoever comes to the spring should sing while they pour"). Runs after
  the planted arm; without it, the newcomer test compares round-1 and round-10 boards
  within each run.
- **New code:** let one agent's round-1 myth be fixed text (`scripted_response` in
  `src/agents.py` already records non-generated responses).

## Measures (fixed before any run)

1. **Text:** share of the other 7 agents' myths naming the rule or the amount,
   round by round, planted vs control.
2. **Play:** the other agents' rate of $2 sends and their mean send, planted vs
   control.
3. **Newcomer test (main test):** after round 10, show a fresh Sonnet agent the
   board with all of Agent_1's myths removed and ask what it would send. If planted
   runs' newcomers lean towards the rule, the rule survived through other agents'
   retellings, not only through the source. A replay of saved runs, no new
   simulations.

On the board every agent reads the planted myth directly, so measures 1 and 2 show
copying from the source; only measure 3 shows retelling.

## Cost

| Item | Estimate |
|---|---|
| 6 board runs (3 planted, 3 control) | 6 × about $6 ≈ $36 |
| Smoke run + Sonnet refusal check on the exact runtime messages | ≈ $3 |
| Newcomer probe | ≈ $2 |
| **Total (n=3 per arm)** | **≈ $40** |
| n=5 per arm | ≈ $65 |

About $6 per run = $3.41 (Sonnet 8-agent myth → game,
`data/json/noise_experiments/table1_n10_extension_20261001/completion_receipt.json`)
plus about $2.60 of board input (8 agents × ~108k extra input tokens at Sonnet's
$3 per million input tokens, `analyses/myth_replay_probe.py` `PRICE`). The smoke run measures the real cost. Grant left on 2026-10-02: €661.25
(`scripts/grant_budget.py`; OpenRouter after 2026-10-01 not yet counted).

## What n=3 can show

Text spread should be clear if it happens at all (the rule name should almost
never appear in controls). Play only shows a large effect. The newcomer test is
suggestive at n=3.

## Left for later (only if spread is established)

Which lever matters (board vs partner channel), selection (stories of successful agents retold more), a newcomer who must
learn from myths alone as a design element, Park-style memory retrieval, and an
Opus arm.

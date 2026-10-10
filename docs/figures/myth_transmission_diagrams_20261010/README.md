# How myths travel through the 8-agent population (TikZ diagrams, 2026-10-10)

Three standalone TikZ figures that show the myth channel of the current
8-agent population design. Compile each with `pdflatex <file>.tex`, or paste
the `tikzpicture` into Overleaf (needs `arrows.meta`, `positioning`, `calc`).
The PNGs are 130 dpi previews rendered with Ghostscript.

| File | What it shows |
|---|---|
| `fig1_one_round_pipeline.tex` | One agent, one round: what goes into its myth call and its game call, and where its myth goes. |
| `fig2_schedule_myth_hops.tex` | The real pairing schedule of one run, rounds 1 to 5, with every myth hop drawn, and the spread of one agent's founding myth highlighted. |
| `fig3_ancestry_tree.tex` | The ancestry of one myth (Agent 1, round 4) three rounds back: six of the eight founding myths. |

## The mechanism, as the code implements it

Verified on 2026-10-10 against the code and against two completed runs (see
"Provenance"). Setting: `noisy8_crossmodel_negative_twotask_r3` in
`config/experiments_noisy.yaml` (8 agents, 10 rounds, memory-primary, informed
negative-only noise, names hidden), the parameter block used by the September
population sets and by the frontier population sets.

1. **Pairing.** Each round the 8 agents are split into four sender/receiver
   pairs by a seeded balanced schedule (`games/dyadic_pairing.py`,
   `_build_balanced_round_pairings`): sender roles are balanced to 5 per agent
   over 10 rounds, and each sender is matched to the receiver it has met least
   often. Over 10 rounds a run covers 25 to 28 of the 28 possible pairs, and no
   pair meets more than 3 times (five Sonnet 4.5 reps). Because pairing seeds
   are paired (`paired_protocol_seeds: true`), the Sonnet, GPT-5 Nano and
   Gemini 3.7 Flash runs with the same rep index use the same schedule, and so
   do the game-then-myth and myth-then-game orders.
2. **Order within a round.** With task order `["myth", "game"]` every agent
   first writes a myth, then plays one trust game with its partner for that
   round (`src/simulation.py`, task loop).
3. **What a myth call sees.** From round 2 on, the prompt contains exactly one
   other myth: the myth written in the previous round by the agent's
   previous-round game partner (`src/myth_writer.py`,
   `get_myth_prompt_round_later` via `_get_paired_opponent_id` on the previous
   round's pairings). The agent's own previous myth is not in the prompt; it
   sits in chat memory as the agent's own earlier reply
   (template `myth_writing_later_rounds_directive_memory_primary`). Round 1
   myths are written from scratch with the game directive
   (`myth_writing_default_game_directive`).
4. **What a game call sees.** The current co-player's last 3 games (noisy
   visible amounts, no names), the agent's visible earnings, and the line
   "Take any myths written in this session into account when making your
   decision" (`games/trust_game_noisy.py`, `_format_multi_agent_history` and
   `_finalize_game_prompt`; `myth_decision_link` in the config).
5. **Memory.** Chat memory keeps the system prompt plus the last 6
   prompt/reply pairs, which is 3 rounds of myth + game (`src/agents.py`,
   `_truncate_messages`; `memory_capacity: 6`). Nothing in memory is visible to
   other agents.

Put together: a myth written in round *t* is read by exactly one other agent,
the writer's round-*t* game partner, in that partner's round *t+1* myth call.
The two partners of a round swap myths with a one-round lag. There is no
broadcast and no board in this design (a shared board exists only as the
separate `frontier_myth_board_20261002` ablation).

Consequence for spread: each new myth has two inputs (own previous myth,
shown myth), so the number of founding myths in a myth's ancestry can at most
double per round. In the five Sonnet 4.5 myth-then-game reps, every agent's
myth has all eight round-1 myths in its ancestry by round 4, 5, 5, 5 and 6
respectively. In rep 00, Agent 1's founding myth is in the lineage of 1, 2, 4,
6, 8 agents in rounds 1 to 5 (Fig 2).

## Provenance

- Schedule and ancestry in Fig 2 and Fig 3: rep 00 of
  `data/json/noise_experiments/negative_only_crossmodel_reasoning_rerun_20260909/negative_only_reasoning_rerun_population_myth_game_claude_n5/claude-sonnet-4.5/myth_game/noisy8_crossmodel_negative_twotask_r3/`.
  The five reps' schedules are saved in
  `schedules_sonnet_myth_game_rep00-04.json`.
- Rule check: in that run and in rep 01 of
  `sonnet45_8agent_myth_directive_history3_anon_memprimary_r10_n5`, every
  later-round myth prompt (72 of 72 per run) quoted the previous-round
  partner's myth verbatim; none quoted the agent's own or a third agent's.
- Prompts quoted in Fig 1 are from the same rep 00 run (Agent 1, round 4) and
  from `config/experiments_noisy.yaml` / `config/experiments.yaml`.

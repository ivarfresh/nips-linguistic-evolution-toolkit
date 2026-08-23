# Frozen screen: Gemini 3.7 Flash task ordering

Frozen on 2026-08-23 before any Gemini 3.7 Flash experimental response was
requested.

## Question

Does Google's newest capable Flash model have enough behavioral headroom in
the corrected eight-agent trust game to provide an informative task-order
comparison, and does a small screen reproduce the Myth→Game ordering observed
in Sonnet 4.5 and GPT-5 Nano?

## Design

Run three paired replicates (IDs 90–92) in each condition:

- Game only;
- Game→Myth; and
- Myth→Game.

All nine populations use `google/gemini-3.7-flash` through the direct Google
API, eight anonymous agents, ten rounds, balanced rotating dyads, the corrected
memory-primary prompts, a three-game private-memory horizon, and informed
signed `U(-1,+1)` communication noise applied after true send and return
decisions. Pairing and noise seeds are identical across the three task orders
within each replicate. Myth conditions retain the shared decision instruction.

Gemini 3.7 Flash is run at its documented default `medium` thinking level.
The model no longer accepts legacy sampling parameters, so the configured
temperature is recorded but deliberately not sent to the provider. These API
differences are part of the model treatment and are recorded explicitly.

## Acceptance gate

Before inspecting behavior, require:

- nine populations, 90 population-rounds, and 360 dyads;
- 1,200 accepted Gemini calls: 240 Game-only and 480 in each myth condition;
- 720 exact signed-noise checks applied after true decisions;
- correct task boundaries, prompt additions, memory horizon, model/provider,
  medium thinking level, and absence of a transmitted temperature;
- identical realized schedules and exogenous seeds across task orders per
  replicate;
- clean embedded code/config provenance; and
- no unrecovered provider, response-boundary, accounting, identity, or hidden-
  state error. Any recovered retry must remain explicit in the saved record.

## Frozen screen outcomes and decision rule

The analysis unit is the independent population. Report condition means and
paired differences for final balance, proportion sent, receiver return ratio,
and dollars returned per dollar sent. Also report round trajectories, exact
maximum-send frequency, response-boundary retries, token use, and estimated
list-price cost. With only three pairs, confidence intervals and p-values are
descriptive and are not confirmatory evidence.

First evaluate headroom. If every sender choice is the full `$5`, declare the
standard task ceiling-limited and do not add baseline replicates. Move instead
to a defector stress test.

If headroom exists, expand the task-order comparison only if all three paired
Myth→Game minus Game-only final-balance differences are positive and their mean
is at least `$1.50` per agent. Otherwise stop the baseline cell and use the
controlled punishment calibration plus a small defector stress test to learn
where this model differs. Game→Myth is reported but does not gate escalation.

No punishment condition enters this screen.

## Outcome-blind technical amendment, 2026-08-23

The first attempted Game-only replicate reached round ten but failed when one
medium-thinking API response exceeded the runner's historical 120-second HTTP
timeout on both the provider-level and simulation-level retries. It produced no
accepted final population and is excluded. Before analyzing any behavioral
outcome, the timeout was made configurable and set to 300 seconds, with its
value and source embedded in run metadata. The same replicate ID is rerun from
the beginning. Model, thinking level, prompts, seeds, sample size, acceptance
gate, outcomes, and escalation thresholds are unchanged.

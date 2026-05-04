# Monday 2026-05-04 team meeting packet

Meeting: Monday 11:00 CEST.

## Main Agenda

Use the meeting to lock the story first, then decide whether any more experiments are worth the deadline cost.

1. Agree on the abstract spine: mechanism-led, not "myths generally increase cooperation".
2. Sanity-check the proposed mechanism language: "channel-opening / coherence anchor" versus a plainer phrase.
3. Decide whether today's remaining experiment time goes to prompt variants, Claude, GPT-5.5 completion, or writing.
4. Assign review roles and deadlines for the abstract draft.
5. Freeze what can go into the main paper by Monday 17:00 CEST.

## Proposed Spine

Inter-agent language is not a uniform cooperation switch. Most myth-writing effects are null or small. The sharp result is conditional: under GPT-5-Nano bootstrap noise, implicit myth-writing destabilises cooperation, while explicitly placing coherent text under the partner-story header restores the game-only trajectory. Because length-matched encyclopedic filler works about as well as the real partner myth, the mechanism is not partner-specific narrative content; it is closer to channel-opening / coherence anchoring.

## Figures To Show

Use the Sunday packet:

- `analysis/team_update_2026-05-03/figures/fig2_bootstrap_mechanism_game_myth.png`
- `analysis/team_update_2026-05-03/figures/fig3_boundary_conditions_game_myth.png`
- `analysis/team_update_2026-05-03/figures/fig4_neutral_framing_trajectories.png`

## Key Empirical Points

- Bootstrap: myth-writing alone drops GPT-5-Nano below game-only; partner-story-labelled text brings the trajectory back toward game-only.
- Filler: length-matched encyclopedic prose under the partner-story header works about as well as the partner's actual myth.
- Boundaries: no-noise is null; positive_5 is saturated; negative_5 is not restored by partner-story text.
- Neutral framing: ROLE A/B lowers GPT-5-Nano and opens headroom; myth-present conditions recover some of that loss. Claude moves less and remains mostly myth-null.

## Overnight Run Status

Status as of 10:06 CEST:

- Gemini-3.1-pro-preview completed all missing-condition sets: positive, negative, and bootstrap are each 90/90.
- Gemini bootstrap does not reproduce the GPT-5-Nano destabilisation. It trends upward with myth exposure: game 62.6, game-then-myth 67.6, myth-then-game 71.8 cumulative reward.
- GPT-5.5 hit OpenAI `insufficient_quota`. Completed usable files: positive 77/90, negative 1/90, bootstrap 1/90.
- The GPT-5.5 positive partial is saturated at ceiling, so it is useful only as boundary evidence. It should not change the abstract.
- The missing GPT-5.5 negative/bootstrap cells are not available for the 11:00 discussion unless OpenAI quota is topped up and the team decides they are worth chasing.
- No Claude was launched overnight.

Interpretation for the meeting: Gemini is ready as robustness/boundary material; GPT-5.5 is incomplete and should not become part of the main argument today.

## Decisions Needed

### 1. Story

Recommendation: mechanism-led.

Decision to lock: the abstract should lead with the conditional channel-opening mechanism, not with heterogeneous model/noise effects.

Question for team: does "channel-opening / coherence anchor" describe the result accurately enough, or should we use a plainer phrase such as "adding a coherent partner-story field stabilises play"?

### 2. Claude

Recommendation: optional, post-meeting only.

Decision to lock: do we need a minimal Claude bootstrap follow-up before submission?

If yes:

- Run Claude Sonnet 4.5 x bootstrap baseline first.
- Add partner-story / filler controls only if Claude shows the same bootstrap myth destabilisation.
- Stop if Claude is null or saturated.

### 3. Prompt Variants

Recommendation: run only if Ivar thinks this is submission-critical.

Decision to lock: should Monday afternoon be spent on less-constrained / no-game-knowledge prompt variants?

If yes:

- GPT-5-Nano x bootstrap only.
- Freeze results by Monday 17:00.
- Include only if clean; otherwise appendix or omit.

### 4. Review Roles

Proposed asks:

- Mario: title/abstract framing by Monday 17:00.
- Edward: mechanism wording sanity check by Monday 17:00.
- Ivar: prompt-variant and experiment-prioritisation advice in the meeting.
- Alexandra / Arabella / Jane: clarity/accuracy pass after Monday evening reframe.

## Monday 17:00 Freeze Rule

By 17:00 CEST, main-text claims are frozen. Any incomplete or ambiguous experiment becomes appendix/post-submission material. The abstract cannot depend on unfinished runs.

## Recommended Meeting Outcome

Leave the meeting with these decisions written down:

- Abstract story: conditional mechanism in GPT-5-Nano bootstrap, with model/noise heterogeneity as boundary conditions.
- Experiment priority: prompt variants only if Ivar judges the current prompt path to be a serious reviewer risk; Claude only if the team sees single-model mechanism evidence as a serious reviewer risk.
- GPT-5.5 quota: do not spend deadline time on it before abstract submission unless someone explicitly argues that missing GPT-5.5 bootstrap robustness is decisive.
- Writing: circulate abstract by Monday 17:00, then update the paper wording Monday evening.

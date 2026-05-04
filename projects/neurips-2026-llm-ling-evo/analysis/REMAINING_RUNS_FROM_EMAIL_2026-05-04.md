# Remaining Experiment Runs From Email Threads

Checked Monday 2026-05-04 against the relevant Gmail threads with Ivar Frisch and Edward Hughes plus the local experiment configs/results.

## Still Not Done Or Incomplete

### 1. GPT-5.5 full missing-condition sweep

Source: Ivar's Apr 24/25 priority list asked for updated models / newer GPT versions; the Apr 30 Ivar thread still had Gemini and prompt variants pending.

Status: incomplete because OpenAI quota stopped the overnight run.

- `noise_positive_mem3_gpt5_5`: 77/90 final runs.
- `noise_negative_mem3_gpt5_5`: 1/90 final runs.
- `noise_bootstrap_mem3_gpt5_5`: 1/90 final runs.

Recommendation: do not chase before the abstract unless quota is fixed and the team explicitly wants GPT-5.5 as robustness evidence. The available positive-noise GPT-5.5 cells are ceiling-saturated and do not affect the story.

### 2. Opus / newer Claude model sweep

Source: Ivar's Apr 24/25 priority list explicitly mentioned "Opus instead of older Claude".

Status: not done. No Opus runs were launched, by design, because Anthropic credit was uncertain and the user requested no Claude runs for now.

Possible run if approved: Opus 4.6/4.7 across positive, negative, and bootstrap missing-condition cells, 270 runs total. This is expensive and not necessary for the abstract spine unless the team wants stronger model-robustness evidence.

### 3. Claude bootstrap follow-up branch

Source: discussed in the Sunday-night plan and team update as a possible post-meeting branch, not an original hard requirement from Ivar.

Status: not done.

Minimal branch if approved:

- Claude Sonnet 4.5 x bootstrap baseline: 90 runs.
- Only if Claude shows GPT-5-Nano-like bootstrap myth destabilisation, add partner-story and filler controls: up to 120 more runs.

Recommendation: run only if the team thinks single-model mechanism evidence is a serious reviewer risk.

### 4. Less-constrained myth prompts

Source: Ivar's Apr 24/25 priority list: "Less constrained myth prompts (not cooperation-focused)"; Apr 30 thread said prompt variations had not yet been run.

Status: not done. Current myth-writing prompts are fairly minimal, but the surrounding task/system path still places myth-writing in the trust-game experiment context. We have not run a dedicated less-constrained prompt variant.

Feasible submission battery if approved:

- GPT-5-Nano x bootstrap only.
- Unconstrained myth prompt while preserving parseability: 60 runs.

Recommendation: only run if Ivar thinks the current prompt path is a serious reviewer risk.

### 5. Myths written before game knowledge

Source: Ivar's Apr 24/25 priority list: "Myths written without game knowledge initially".

Status: not done. This requires a new system-prompt path and a flow change: first write the myth under a myth-only/non-game system prompt, then switch to the trust-game system prompt.

Feasible submission battery if approved:

- Myth-first blind: 30 runs.
- Myth-first blind plus unconstrained prompt: 30 runs.

Recommendation: methodologically important, but higher implementation risk than normal config-only sweeps. Include only if it runs cleanly by Monday evening.

### 6. Minor Claude/Anthropic backfills

Source: current local run status, not a major email ask.

Status: not done because Claude/Anthropic runs were deliberately skipped.

- `neutral_framing_v4_mem3`: 89/90 final runs; one Claude neutral-framing run failed due Anthropic credit.

Recommendation: low priority. Do not spend abstract time on this unless Anthropic credit is already fixed and Claude work is approved.

## Discussed But Already Covered

### Gemini 3.1 Pro

Source: Ivar Apr 25 said to use `gemini-3.1-pro`, not deprecated `gemini-3-pro`; Apr 30 thread said Gemini was still pending.

Status: done overnight for the v4 missing-condition matrix.

- Positive: 90/90.
- Negative: 90/90.
- Bootstrap: 90/90.

### Inserted vs evolved myths

Source: Ivar's Apr 24/25 priority list.

Status: done for GPT-5-Nano mechanism controls.

- Real partner myth injection: 118/120 final runs.
- Cross-dyad shuffled myth control: 60/60.
- Encyclopedic filler control: 60/60.
- Own-myth control: 60/60.
- Boundary controls in positive, negative, and no-noise cells are also done.

The small real-partner-myth shortfall is not scientifically blocking; the control result is already clear.

### Negative noise myth-game / game-myth

Source: Edward's May 1 note: he liked the negative-noise shape and wanted to see myth-game and game-myth in that setting.

Status: done for GPT-5-Nano, Claude Sonnet 4.5, and Gemini 3.1 Pro in the current v4/direct-provider results. GPT-5.5 negative remains incomplete only as part of the newer-model sweep.

### Noise symmetry and invalid-return fix

Source: Ivar Apr 22 and Edward Apr 27 discussion.

Status: code-side issue addressed in the current run path: noise is applied to both sent and returned values where configured, and values are clamped so returns cannot violate the valid range.

## Discussed But Probably Out Of Scope Before Submission

- Deterministic-noise reruns: discussed and partly tried earlier; not a current priority because deterministic noise produced degenerate trajectories and some older runs had the pre-clamp caveat.
- Extra two-player games, donor game, public-goods game, or more than two agents: discussed as robustness/future work, not needed for this NeurIPS sprint.
- Hyperparameter sweeps such as multiplier and number of generations: discussed as possible robustness, not run and not advisable before deadline.
- Semantic-similarity, word-usage, linguistic-structure, LLM-as-judge ratings, and myth uptake in game reasoning: analysis tasks rather than simulation runs. Some scripts/data exist, but they should be treated as optional analysis polish, not new run commitments.

## Short Priority Order

1. Finish writing/abstract consistency first.
2. If experiments are approved today, run GPT-5-Nano prompt variants before any broad model sweep.
3. If model robustness is the bigger concern, run the minimal Claude bootstrap branch.
4. Resume GPT-5.5 only after OpenAI quota is fixed and only as robustness/boundary evidence.
5. Skip Opus unless the team explicitly wants to spend Anthropic credit and time.

---
title: Design constraints — what breaks ablations in this framework
status: current
updated: 2026-07-02
owner: ivar
---

# Design constraints

Hard-won rules that determine whether an ablation is a real condition or a
$60 tautology. Check new designs against all three.

## 1. Statelessness collapses task orders (Option A)

LLM calls are stateless: each API call sees only the messages passed to it.
Under Option A (`remember=False` on discarded calls), task orders that differ
only in "what unrelated API call fired earlier" present **identical**
`messages_sent` to the decision call — they are the same condition sampled
thrice, not three conditions. Empirically confirmed: baseline / s_start / s_end+
all within sd across `["game"]`, `["myth","game"]`, `["game","myth"]`
(~$60 spent confirming). **To make task orders matter, the decision-time call's
input must differ (Option B).** _(from researchlog 2026-06-30)_

## 2. Refusals are a knife-edge conjunction — probe with runtime messages

Sonnet 4.5 can deterministically refuse (`stop_reason="refusal"`, 0 content
blocks) when an unusual-register seed sits in the **assistant slot**:

- Jabberwocky (>~200 words): refused in Phase 5; length-correlated. Same text in
  system/user slot is fine. _(from researchlog 2026-06-30)_
- Gowith seed 4: **0/8 refusal with the game-built prompt wording, 16/16 pass
  with config-template wording differing by a few words**; the original English
  passes everywhere; 4 of 5 gowith siblings pass everywhere. The trigger is a
  conjunction of content × style × exact context. _(from researchlog 2026-07-02)_

Consequences: (a) probe candidate seeds with the **exact runtime-built messages**
(monkeypatch dump, see `data/phase7/debug_failing_call.json`), never template
reconstructions; (b) retries rescue only *stochastic* refusals —
`src/utils.py::_should_retry_anthropic` retries empty responses since 2026-07-02;
deterministic ones censor the rep (report as refusal-censored, don't force);
(c) the refusal classifier is itself a surface-form-sensitive "reader" —
potentially reportable.

## 3. Ceiling saturation hides effect sizes

S-end+ saturates $600 in every rep, so differences *above* its effect are
invisible under the current game. A harder game (or lower multiplier) is required
before claiming any manipulation "matches" S-end+. _(from researchlog 2026-06-23)_

## Cheap-screen-first economics

The round-1 behavioral probe (~$1.30/pool) reproduces full-cell orderings and
predicted the gowith cell. Default workflow: probe → fund only interesting cells
($11.50 each). _(from researchlog 2026-07-02)_

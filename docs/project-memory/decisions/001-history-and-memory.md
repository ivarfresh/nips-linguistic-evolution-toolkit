# D001 — Retain own history and partner history without duplicate reminders

- Recorded / last verified: 2026-09-15 / 2026-11-27
- Decision status: July memory-primary choice explicitly recorded; August24 history retention conditionally endorsed.
- Scope: ordinary rotating-population game/myth runs. Separate history-visibility controls and injected-myth-only ablations are not superseded.
- Decision authority: Ivar's July decision recorded in researchlog; Ed's August24 remarks attributed by meeting transcript.
- Implementation status: implemented; three bounded June–September final samples inspected, not a census.

## Decision and rationale

Keep game and myth exchanges in private chat memory; remove repeated own-history and own-myth prompt recaps. Keep the current co-player's three-game record in the sampled population condition. **Explicit rationale:** the July log says dropping game responses from memory would remove strategic response context from subsequent myth writing. This preserves the intended cross-task pathway; it does not prove a causal mechanism or access to all private model reasoning.

Ed's June5 request sought more gameplay history, especially the partner's. His August24 conditional position was to retain own-three/partner-three if already implemented. **Unknown:** why three was initially preferred over another length.

## Evidence

- [June 5 source, slide 650](../../experiment_design_reference.md#source-slide-650), Edward Hughes, 2026-06-05T11:30:54Z, thread `AAAB84WHhCk`: proposal, not a numerical selection.
- [June19 source slide660](../../experiment_design_reference.md#source-slide-660), thread `AAAB_5opJy8`: Ed flags two confusing memory mechanisms.
- [July17 researchlog](../../../researchlog.md), “memory-channel pilot,” and [commit9c392fcf](https://github.com/ivarfresh/nips-linguistic-evolution-toolkit/commit/9c392fcf): revised decision/implementation.
- [August24 transcript](https://docs.google.com/document/d/1K_7K443wQhyf86iMJEXySJgHTD_HeYEvTdvgJ1O1y3k/edit), 17:33–18:56: conditional retention; attribution has transcript limitations.
- [Final sample manifest](../../design_evidence/2026-09-15/sampled_runs.json): June self3/co-player3/capacity3; July and September self0/co-player3/capacity6, memory-primary. Full-state paths/hashes are in the manifest.
- [Agent truncation](../../../src/agents.py), [history builder](../../../games/trust_game_noisy.py), [myth template](../../../config/experiments.yaml), [task retention](../../../src/simulation.py).

## Chronology and supersession

June8 expanded history precedes the July17 duplication repair. July changes the channel carrying own history; self-window0 does not mean no own history. August12 no-injected-history isolation cells are separate experiments; the [August21 visibility protocol](../../history_visibility_confirmatory_protocol_2026-08-21.md) explicitly compares them with partner-history cells. June22–23 injected-myth-only agreement applies to a separate ablation.

## Unresolved / next evidence

Capacity counts interaction pairs: six pairs approximate three raw rounds with two tasks. Model-written summaries can preserve older events; see [effective-memory note](../../effective_memory_horizon_note.md), whose numerical findings were not reanalysed here. Reconcile exact prompts/task orders for any future matrix, including [D005's held proposal](005-hold-180-runs.md).

## 2026-11-27 — Saved-run sliding-window verification

**Explicit request:** Ivar requested independent verification using existing runs
only, with no new simulations/API calls or implementation/config/data edits.
**Completed evidence, narrower scope than the population decision above:** the
ordinary September Sonnet myth→game dyad, replicate 0, has ten complete rounds
and forty recorded logical calls. Its final SHA matches earlier research manifests;
all seventeen implementation hashes and the configuration hash match the recorded
clean commit. See the [audit report](../../research/memory_window_audit_20261127/REPORT.txt),
[source manifest](../../research/memory_window_audit_20261127/source_manifest.json)
(primary sources: completed final, historical code/config and research manifests),
and [offline extractor](../../research/memory_window_audit_20261127/audit.py).

- All forty saved pre-request snapshots retain the system prompt and exactly the
  latest up-to-six complete exchanges, unchanged and chronological. Each full-window
  transition evicts the oldest whole pair. Observed task counts are one myth and one
  game per round per agent. At myth calls this is three previous rounds; at game
  calls the six pairs span portions of four round labels.
- **Qualification:** final stored buffers contain seven complete exchanges;
  six bounds previous exchanges at request time, not all saved buffer states.
  Steady myth-writing inputs expose three own myths but **four partner myths**,
  because historical prompts contain partner texts authored one round earlier.
- Every later myth prompt equals the saved scaffold plus only the previous-round
  partner myth. Own texts are supplied by retained assistant content; no own or
  older partner full myth is reinserted. A name from own myth 2 persists through
  myth 10 after its original message leaves; this is textual continuity, not proof
  of causal ancestry or internal recall.
- The snapshots are contemporaneous **pre-adapter** inputs. Hash-verified Anthropic
  code separates system and removes stored usage/reasoning fields. Raw HTTP bodies
  and hidden retry counts are unavailable. The earlier effective-memory note's
  reasoning-recopy interpretation must not be transferred to this provider/run
  without separate evidence. No census or causal memory-horizon result is claimed.

**2026-11-27 task-order follow-up:** forty saved calls in the corresponding ordinary
game→myth dyad (replicate 0) also pass the complete-pair suffix checks. At steady
game calls in round R, own myths are R−3..R−1 and partner myths R−4..R−2; the
subsequent myth prompt adds partner R−1. Both orders therefore expose three
partner myths at game time, but game→myth exposes myths one source-round older
than myth→game. This is a measured input difference, not an estimated behavioural
effect. Primary final path/hash, full metadata and all forty exposure rows are in
[the follow-up evidence](../../research/memory_window_audit_20261127/game_myth_followup.json),
with an [offline extractor](../../research/memory_window_audit_20261127/game_myth_followup.py).

**2026-11-27 team meeting (Gemini-written notes, not quotation authority;
[notes](https://docs.google.com/document/d/17hQG5LqDBCbwG1QmMimmTdB53ieSpe2MLPH8hNk4W2o/edit)):** Ivar reported the four-partner/three-own myth asymmetry. The
notes attribute to Edward the request to run a few dyad replicates with equal
myth-history lengths and check that cooperation is unaffected, in place of
rerunning all experiments. Status: agreed, **not run**; pairings, replicate count
and which side changes (three partner myths or four own) are unspecified.

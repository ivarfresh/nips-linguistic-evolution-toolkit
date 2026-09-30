# Lineage-network view of moral spread (designer: network lens)

Files: `moral_lineage_network.png` (300 dpi, 7.0 × 5.3 in, double column), `moral_lineage_network.pdf`,
script `moral_lineage_network.py` (run with the moral-split worktree as cwd; command in its docstring).

**(a) Design idea.** Each 8-agent myth→game run is drawn as a time-expanded network: agents are rows, rounds are columns, each node is one myth filled by its moral label, and each edge links the myth an agent was shown to the myth it wrote next, coloured when the label was kept. Under each run the observed number of kept labels is set against the number expected from an unseen same-family myth, and panel c gives the test over all 45 runs on a 0–100% axis, so the large base rate stays visible.

**(b) Takeaway.** Morals spread only a little, and only within a model family: a shown myth's label is kept about 5 points more often than an unseen same-family myth's label, on top of a roughly 60% base rate. Across families the gain is about zero.

**(c) Run selection.** The rule was fixed before rendering and not changed afterwards. I picked the compositions for balance and label variety: 8 Sonnet is 48% fair and 52% generous, and 4 Gemini + 4 GPT pairs two families with distinct styles. Within each composition I took the replicate whose per-run label excess (shown minus unseen) is the median of its 5 replicates. For the homogeneous run that is computed over all edges; for the mixed run it is computed over same-family edges only, because that is where the claimed effect lives. Result: 8 Sonnet rep 3 (+5.1 pts; 42 of 72 labels kept vs 38.3 expected) and 4 Gemini + 4 GPT rep 4 (same family 22/32 vs 22.0 expected; across families 22/40 vs 21.3).

**Verification.** The script recomputes the per-child match and unseen values with the repo's own `null_candidates` and `load`. It asserts that the result reproduces all 7,373 rows of `moral_uptake_children.csv`, and that panel c's mean, sd and n match `moral_uptake_by_task_order.csv`: +4.8 (±4.5), p = .005, 13/15 runs; +5.9 (±7.8), p = .001, 21/30; +0.7 (±9.2), p = .37, 17/30. It also asserts that every exposure in these runs comes from round r−1 and that each run has 72 edges. κ = 0.54 comes from README §4. Nothing was written to the repo.

**(d) Weaknesses.**
- A single run cannot show the effect. The excess is about 3–4 extra kept labels out of 72, and the median mixed run shows none on same-family edges. The coloured edges are mostly base rate, which is why the observed-vs-expected line and panel c are essential and must not be cropped.
- Colour blocks mislead. GPT writes "be fair" 84% of the time in this composition, so GPT→GPT edges match almost automatically. The whole-column switches in panel a (round 5 all fair, round 1 almost all generous) are round-level shifts shared by everyone, most likely the state of play, not transmission; the same-round null absorbs them. The caption should say so.
- 72 curved edges per panel is dense at print size. I faded the cross-family edges and drew them dashed, but they are still busy in panel b.
- "Be cautious" appears in neither run, so it is dropped from the legend and named in the footnote (crimson).
- Labels come from one LLM judge (GLM-5.2, κ = 0.54 vs DeepSeek, lowest on Gemini at 0.30) and have no human validation yet. The figure also shows nothing about behaviour, and morals do not carry into play; the figure should not be placed next to a claim that they do.

**Colours (team scheme, lead decision 2026-09-30).** generous #D9A400 (gold), fair #0b5394, cautious #B2182B (crimson). The proposed fair blue #2a78d6 failed against Sonnet purple #7570b3 (colour-blind ΔE 5.6, normal-vision ΔE 9.1, floor 15), so I darkened it to #0b5394. All six colours (three labels plus the family colours #7570b3, #d95f02, #1b9e77) then pass the all-pairs check: worst colour-blind ΔE 11.6 (a GPT–Gemini family pair), worst normal-vision ΔE 15.6 (fair vs Sonnet, just above the floor). Gold has only 2.2:1 contrast on white, which the legend's text labels cover.

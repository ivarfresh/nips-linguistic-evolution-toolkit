# Myth analyses on the frontier runs, September beside frontier (2026-09-30)

**Answer.** The frontier models repeat the September picture, more sharply where they can be
tested at all. An agent's own first myth is its plan: before any play, frontier senders send
exactly the amount their myth names 50 of 52 times ($0.85 per $1 stated; September 44 of 62,
$0.67). Words and a myth's stated amount carry into the reader's next myth, and within a
family the moral does too. Nothing read from another agent has a detectable effect on play,
but on frontier that test mostly cannot run: Gemini 3.1 Pro always sends $5, Opus 5 nearly
always, and later-round sends sit at $5 for every family.

**Figure:** `scorecard/scorecard.png` (and `.pdf`). Rows: every simulation × population,
September above frontier. Columns: eleven findings. Each cell carries its status (colour +
symbol) and headline number; `scorecard/scorecard_cells.csv` traces every cell to its lens
row, and `scorecard.py` asserts each printed effect, CI and run count against it.

## Data

Main frontier set only: Claude Opus 5, Gemini 3.1 Pro Preview, GPT-5.6 Sol (effort high).
106 myth runs, 4,520 myths: 2-agent one model 30, 2-agent mixed 36, 8-agent one model 30,
8-agent mixed (2 GeminiPro + 3 Opus + 3 Sol) 10. Excluded: the update set (Opus 5.5, GPT-6
Sol), smoke and quarantined runs, game-only runs. Built by `analyses/linguistic_corpus.py
--dataset frontier` into `data/analysis/linguistic_frontier_20260930/`; the same code rebuilds
the September tables byte-identically. Judged with September's exact requests (GLM-5.2 labels,
summaries, rules, amount check, 0–10 giving score; DeepSeek V4 Flash as robustness, served
with hidden reasoning on frontier, so not strictly comparable). Spend $10.17 (summed billed cost
columns), plus a few cents of retries and in-flight calls from one restarted pass.

Round-1 identification holds on frontier: within each composition cell the send-call
messages are byte-identical apart from the sender's own myth, and no thinking content is
passed back (`analyses/frontier_round1_identification.py`). Senders only.

## Findings (frontier; September in brackets)

- **Own first myth = plan.** Sol 31/31 exact ($1.00 per $), Opus 19/21 ($0.61, suggestive:
  only 5 Opus senders sent under $5), GeminiPro 37/37 at $5 (ceiling). Stated amount and
  send rule add +0.61 held-out R² for round-1 sends [+0.25]. Fades after round 1 in both.
- **Read myth → next send:** not estimable (ceiling) [none: −$0.00, −0.10 to 0.10].
- **Read myth → next myth amount:** +$0.13 per $ (0.06–0.21), Holm-significant, unseen
  placebo null [+$0.07].
- **Words copied from the read myth:** stronger everywhere; across families +6.5 pts in the
  mixed population [+1.5].
- **Moral spreads within family** (8-agent one model, myth→game): +6.8 pts, future-myth
  placebo +0.6 (passes); rests on Opus and GeminiPro; Sol labels 96% "be fair" [+4.8, but
  its placebo is +3.9, p .05 — see corrections].
- **Moral spreads across families:** no (mixed population +13.1 with placebo +13.4) [+0.7].
- **Matching norms → cooperation:** no detectable effect wherever testable; most send cells
  ceiling or too few runs [no evidence].
- **Consistency drift:** the September words barely appear (Opus ~1%, Sol 8%); the judges'
  flag rises in every family but the judges agree less (κ 0.37). No detectable effect on
  cooperation.
- **Stories add beyond past moves:** sends +0.000 [+0.000], but underpowered for Sol. One
  suggestive frontier hint: the return rule Opus states tracks its returns (between agents).
- **Morals follow play:** yes, steeper (+0.53 per unit send [+0.16]); Sol's next myth names
  +$0.94 per $1 sent [Sonnet +$0.66].
- **GeminiPro's morals drift generous → fair while it keeps sending $5** (8 GeminiPro 78% →
  30%; 5 runs), without a stingy partner. This favours "the judge reads ceiling reciprocity as
  fair" over "Gemini stays generous only while reciprocated".

## Corrections to the September write-ups surfaced here

1. Moral uptake +4.8 (8-agent one model, myth→game) has a future-myth placebo of +3.9 (p .05)
   that was never reported; treat as suggestive (affects `docs/figures/linguistic_analysis_20260923/README.md`
   and the lineage-network figure).
2. The shown-amount → send "+0.04 (author ≠ current partner)" in the September synthesis came
   from the future-myth model; the headline spec gives −$0.00 (−0.10 to 0.10).
3. Consistency: the judge-free embedding rise is distinctive only as the consistency-minus-
   generosity difference; the embedding self-copying test is generic. The keyword drift and
   the 72%/34% keyword ratchet stand.
4. Consistency Holm: counted lens-wide over its R1–R4 rows, two Sonnet round-1 results survive
   (keyword consistency → later cooperation, +0.095 all settings, +0.110 8-agent mixed); the
   synthesis's "none over ~540 tests" used a different family.
5. September 8-agent "matching send scores → less sending" sits mostly in 2 GPT + 6 Sonnet
   (−0.039 per SD, Holm 0.050): opposite in sign to H1.

## Files

One folder per lens (`spread`, `plan`, `alignment`, `consistency`, `search`) with scripts,
CSVs, PNGs and `scorecard_rows.csv` (September and frontier rows). Each lens first reproduced
its committed September headline through the ported code. Files over 2 MB (decision and
feature tables, per-child uptake tables) are in the gitignored
`data/analysis/frontier_myth_predictors_20260930/` with the same layout. `BRIEF.md` is the
shared brief. The lens reports went to the lead by message; this README is their synthesis.

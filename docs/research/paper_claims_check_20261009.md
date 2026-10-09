# Paper claims check, 2026-10-09

Checked against the Overleaf checkout at `~/Desktop/Research/AI_projects/LLM_evolution/overleaf-paper`
(commit 8a2e4ca, fetched 2026-10-09 10:40) and the result files in this repo. Read-only; no edits
to the paper. Three parallel checkers covered Results 4.1–4.2, 4.3–4.4 and 4.5; the figure reading
and the claim mapping are mine.

Main-text figures: Fig 1 design, Fig 2 dyad boxplots, Fig 3 nine-panel population ladder,
Fig 4 frontier mixed dyads (boxplots + send per round), Table 1 myth excerpts, Fig 5 word
adoption + myth map, Fig 6 send/return by moral of the myth read.

## 1. Main claims and where they are supported

| Claim (abstract / intro bullets / section headings) | Main-text figure | Numbers in text | Verdict |
|---|---|---|---|
| Myths raise cooperation below the ceiling; largest when agents write first | Figs 2, 3 | 4.1 | Supported. Means check out (myth→game > game→myth in all 13 non-Gemini compositions, +4.7 to +11.6 in populations). Note: the boxplots show medians, and the Sonnet dyad medians go the other way (56.1 game→myth vs 54.5 myth→game), so a reader may see a contradiction with "all 13". |
| GPT-5 Nano sends almost nothing in game-only play; myths change even its round-1 send ($0 → $3.30) | Figs 2, 3 (game-only at 25) | 4.1 | Supported for single-model GPT groups, which is what the numbers are. Round-1 sends themselves are only in supplementary Fig s-mixed-8-send. In mixed groups GPT's game-only round-1 send is not 0 (1.00 in Sonnet+GPT dyads where GPT sends first; 0.66 in 2 Gemini + 6 GPT). |
| A withholding model drags mixed groups down without myths | Figs 2 (bottom), 3 | 4.2 | Supported and visible. |
| With myths, 17 of 18 mixed cells end above the average of their parts, 7 Holm-significant | Eyeballable from Figs 2–3 | 4.2 + supp Table mixed-vs-average | Supported by `docs/figures/mixed_vs_average_n10_20261001/mixed_vs_average.csv`. |
| Myths widen the margin over the parts in 17 of 18 cells, 4 Holm-significant | None | 4.2 + supp Table s-interaction | Point estimates reproduce from the CSV. The Welch/Holm p-values had no source in the repo until 2026-10-09; now `analyses/mixed_vs_average_interaction.py` → `docs/figures/mixed_vs_average_interaction_20261009/` (17/18 positive, 4/18 Holm-significant reproduce; a few last-digit entries differ from Overleaf, listed in that README). |
| Frontier models cooperate more; myths still help below the ceiling; myth→game lifts every mixed frontier pairing to near the ceiling | Fig 4 (mixed dyads only) | 4.3 | "Myths help below ceiling" and "every pairing near ceiling" are visible in Fig 4. "Frontier models cooperate more" (Opus 5 67.8 vs Sonnet 52.9) is **supplement only** (Fig s-frontier-single); no Sonnet appears in Fig 4. Opus n=5 vs Sonnet n=10. |
| Without myths, later frontier transfers track the opening (16/18 within $1; 56.2 vs 74.4 by first sender) | Fig 4b only loosely | 4.3 | Numbers match the frontier_main_mixed README. Neither number is readable from Fig 4b (mean ± sd band pools runs); the split is in supp Fig s-frontier-first-sender, as the text says. |
| Frontier defectors: myths raise non-defectors in Opus 5 populations | None | 4.3, 5.1 | Supplement only (Table s-frontier-defectors). n=5. Matches `frontier_defector_populations_20261002/myth_effects.csv`. |
| A myth behaves like a plan for its author: 71% send the named amount, $0.67 per $1 | Table 1 (6 hand-picked excerpts) | 4.4 | **Numbers match but there is no main-text plot**; the replay figure was moved to the supplement on 8 Oct. The regression is the September n=5 corpus (62 senders, 37 runs), stated as "first five replicates". Table 1 rows 1–4 are from top-scoring runs, as the caption says. |
| Opening carries much of the writing-first advantage in single-model groups ($0.42 → $0.17 after adjusting), not in mixed ($0.49 → $0.66) | None | 4.4 | Matches `myth_replay_probe_20261001/README.md`. n=5 era; the README itself calls the test imperfect. Hedged correctly in the text. |
| Injection case study: own-myth injection 45 → 70, filler 45.7 | None | 4.4 | Supplement only (Fig s-ablation-2). Matches `slide678_dyad_rerun_20260917/summary.json`. Old apparatus, n=5. |
| Agents reuse words from myths they read (1.4–2.4× unseen), more within a family (6.2 vs 1.6 pp) | Fig 5a | 4.5 | Supported and visible. Ratios 1.37–2.41 (text says 1.4), runs positive ≥96.7%. |
| The shown myth beats the author's-next-myth placebo (1.2–1.3×, 72–99% of runs) | **None** (Fig 5a has shown vs unshown only) | 4.5 | Was scratch-only until 2026-10-09; now `analyses/word_adoption_placebo.py` → `docs/figures/word_adoption_placebo_20261008/` (reproduces 1.16–1.34×, 44/60 to 119/120 runs, Holm p ≤ 2.2e-5). This is the control that lets the paper say "reading causes uptake". |
| Myths stay distinguishable by model (98.8–99.1%); families drift apart in single-model, not mixed | Fig 5b | 4.5 | Fig 5b and distance numbers match the n=10 myth map. **98.8–99.1% is the old n=5 map; the n=10 data behind Fig 5b give 99.5%.** |
| Amounts in a read myth show no association with the reader's send (−$0.00, CI −0.10 to 0.10) | None | 4.5 | Matches `frontier_myth_predictors_20260930/plan/h2_send.csv`. 8-agent myth→game populations only, 45 runs, n=5 era; the text does not say myth→game only. |
| Moral matching: 2.7 pp over unseen, but placebo explains it (+0.8, p=0.54) | None | 4.5 | Was scratch-only until 2026-10-09; now `analyses/moral_placebo.py` → `docs/figures/moral_placebo_20261008/` (reproduces 2.7 pp p=0.015, 1.6 pp p=0.13, +0.8 [−1.7, 3.3] p=0.54, −1.1 p=0.67). It is 60 mixed 8-agent myth→game runs, raw p; the committed n=10 moral table pools task orders (3.5 pp / 0.8 pp), which is why those differ. |
| Same player does not send more after a generous myth (generous vs fair −0.004) | Fig 6 | 4.5 | Supported and visible (bottom panels overlap). Regression includes Gemini (300 runs) while the figure drops Gemini (260 runs); the README states this. |
| Later myths restate own last send 71%; read amount adds $0.07 to the amount written | None | 4.5 | Matches `myth_predictors_20260930/README.md`. n=5 era; $0.07 not Holm-significant. "71%" is a different quantity from the 71% in 4.4. |
| Gemini's generous share falls 0.59 → 0.38 while it sends $5 every round | None (supp Fig morals) | 4.5 | Pooled over task orders; per order 0.53 → 0.20 and 0.66 → 0.56. Team rule is never to pool task orders. |
| Judge agreement 74.6%, κ 0.54 | — | 3.5, 5.2 | n=5 corpus only; no DeepSeek labels exist for the n=10 corpus. |

## 2. Statements that are wrong or overstated as written

1. **4.2** "a mixed dyad with GPT-5 Nano ends near the more cooperative of its two single-model dyads instead of near GPT-5 Nano's" is wrong for Sonnet+GPT: with myths GPT+GPT dyads (62.9 myth→game) are themselves above Sonnet+Sonnet (57.8), and Sonnet+GPT (63.2) sits next to GPT+GPT. Holds only for Gemini+GPT.
2. **4.2** GPT-5 Nano "returns almost nothing without myths (0.00–0.05)" holds for dyads only; populations return 0.00–0.17 without myths.
3. **4.2** "Sonnet sends it $0.9 per decision, against $2.8 to another Sonnet" is game-only; with myths Sonnet sends GPT slightly more than another Sonnet (3.25 vs 3.13; 3.70 vs 3.28). Add "without myths".
4. **4.3** "GPT-6 Sol ends at the ceiling in every condition": the update README says five of six cells; dyad game→myth is 70.7 (±8.8). Same overstatement in the supp caption fig:s-frontier-update.
5. **4.4** "Every Sonnet myth that told its reader to send 'three of five' was followed by a $3 transfer (12 of 12)": the source is the author's own founding myth predicting its own round-1 send, not a myth telling a reader. Write "every Sonnet founding myth that named 'three of five' was followed by a $3 transfer from its author".
6. **4.5** "GPT-5 Nano starts using a Sonnet signature word 3.9% of the time" sits inside the "In mixed populations" sentence but is the dyad row; the population row is 3.3% vs 0.6%.
7. **4.5** "98.8–99.1% of unseen runs" is the n=5 map; the n=10 map behind Fig 5b gives 99.5%.
8. **Discussion opening paragraph** says the results are "consistent with cultural transmission in humans" and that in mixed groups "less cooperative agents can be positively influenced by other, more cooperative agent behavior". Sections 4.4–4.5 say the opposite about mechanism: readers do not act on the amounts or morals they read, the myth works as its author's own plan, and only words pass between agents. The paragraph contradicts the paper's own story and is not supported by any figure.

## 3. Housekeeping before submission

- Visible markers in the compiled main text: red `\TODO{Ed: cite his paper on initial conditions here}` in 4.4 and orange `\CHECK{Ivar and Aron: confirm tools and versions}` in 3.5.
- All 23 `\ref`s in the main text resolve to labels in the supplement.
- Methods says ten runs per mid-tier cell; 4.4 and most of 4.5's regression numbers are from the September n=5 corpus. Two places say "first five replicates"; the $0.07, 71% restatement, κ and 98.8–99.1% do not.
- Figure provenance: the two mid-tier PNGs and the frontier send-per-round PNG match renders on branches `analysis/paper-figure-fixes-20261007` / `analysis/supp-figure-todos-20261008`, not origin/main. The frontier resources boxplot PNG (and `frontier_resources_boxplots_final.png`) match no committed blob at all (a later uncommitted re-render, same size, ~10% pixels differ). `docs/figures/myth_examples_table_20261008/` and `docs/research/mechanism_checks_20261007/`, both cited in the tex comments, do not exist anywhere; Table 1 rows were verified directly against run JSONs and all six match.
- ~~The two placebo analyses (word and moral) exist only in `/private/tmp/claude-502/`~~ Done 2026-10-09: scripts in `analyses/`, outputs and READMEs under `docs/figures/word_adoption_placebo_20261008/`, `docs/figures/moral_placebo_20261008/` and `docs/figures/mixed_vs_average_interaction_20261009/`; the gitignored n=10 inputs are mirrored on the shared HF dataset under `ivarfresh/analysis/linguistic_n10_20261001/`.

## 4. Second pass, Overleaf commit f0d9475 (2026-10-09, after 8a2e4ca)

Changed since the first pass: Fig 2 caption (box/whisker definition), Fig 4 panel labels (a)/(b),
Fig 5 caption (error bars, stars, distance numbers rescoped to populations), Fig 6 caption ("share
of the amount received"), a re-exported Fig 5a PNG (legend now "unseen myth"), two red Ivar notes
removed from supplement captions, and the AI-use section filled in. No body sentence changed.

New caption claims, verified:
- Fig 5a "mean over runs with 95% CI": `analyses/linguistic_uptake.py:173` uses t(0.975, n−1) × sd/√n. Correct.
- Fig 5a "Holm-corrected paired Wilcoxon, *** is p<0.001, number of runs in which the shown myth scores higher": `plot_word_adoption_panel.py:42` annotates `adopt_excess_p_holm` through `stars()` (`linguistic_uptake.py:194`, *** below 0.001) and `adopt_excess_runs_positive`/`n_runs`. Correct.
- Fig 5b "0.05–0.20 in single-model populations, 0.01–0.02 in mixed": 8-agent rows of `myth_convergence_map_n10_20261006/significance.csv`: single-model +0.054 to +0.198, mixed +0.006 to +0.022. Correct now that the sentence says populations (the old −0.08 included dyads). Note three of the four mixed-population intervals include 0; "by only" is fair.

Still open from sections 2 and 3: all eight wording/claim issues, the TODO in 4.4 (line 94), the CHECK in 3.5 (line 71), the uncommitted frontier boxplot PNG (the placebo analyses and the interaction table are committed as of 2026-10-09).

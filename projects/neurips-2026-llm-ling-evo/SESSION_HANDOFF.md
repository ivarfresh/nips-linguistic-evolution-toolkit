# Session handoff — LLM Linguistic Evolution NeurIPS 2026 submission

**Superseded for deadline planning:** use `analysis/SUBMISSION_RUNBOOK_2026-05-03_TO_05-07.md` for the current plan. The official NeurIPS deadlines are May 4 / May 6 Anywhere on Earth, which should be treated locally as Tuesday 2026-05-05 13:59 CEST for the abstract and Thursday 2026-05-07 13:59 CEST for the full paper.

*Saved 2026-05-02 09:49. For a fresh Claude Code session: **read this first, then `analysis/MORNING_BRIEFING.md` for the latest results.** Then ask the user what they want next.*

## What this project is (in 60 seconds)

NeurIPS 2026 paper on whether linguistic exchange between LLM agents is *load-bearing* on cooperation. Two-agent dyads, repeated trust game ± iterated myth-writing, GPT-5-Nano + Claude Sonnet 4.5, action-channel noise to manufacture behavioural headroom.

The paper has now landed on a clean falsifiable mechanism claim: **inter-agent language is load-bearing on cooperation, but through *visibility* (common-knowledge anchor à la Chwe 2001), not through content semantics.** This was established by a 2×3 factorial that ran overnight 2026-05-01 → 2026-05-02. Earlier, the paper was a heterogeneous-effects-on-cooperation paper; now it's a clean mechanism paper.

## Deadlines

- **Abstract: Monday 2026-05-04 AoE** (NeurIPS 2026 abstract deadline, T-2 days from now)
- **Full paper: Wednesday 2026-05-06 AoE** (T-4 days)

## What's done

### Manuscript state (all c_final.md files updated)
- `_0-introduction.c_final.md` — locked
- `_1-background.c_final.md` — locked
- `_2-method.c_final.md` — updated 2026-05-02 with §3.4 description of A1 (partner-myth injection) + A3 (forced reasoning) variants
- `_3-results.c_final.md` — updated 2026-05-02 via ms-writer; **includes new §4.6 "Tightening the cross-task channel"** with the 2×3 mechanism factorial. 1,792 prose words.
- `_4-discussion.c_final.md` — updated 2026-05-02 via ms-writer; **§5.2 rewritten** to lead with the empirical visibility-mechanism claim instead of speculation. 1,269 prose words.
- `_5-appendix.c_final.md` — locked
- `index.qmd` abstract — updated 2026-05-02 with the mechanism finding

### Overleaf bundle ready
- `manuscript/overleaf-upload.zip` (2.2 MB, dated 2026-05-02 09:49)
- `manuscript/overleaf-upload/main.tex` (44 KB, natbib citations, 32 \citep{} groups)
- 19 figures including 3 new ones (fig_bootstrap_rescue, fig_a1_deltas, fig_reason_coding_updated)
- Regenerate any time with: `python3 projects/neurips-2026-llm-ling-evo/analysis/rebuild_overleaf.py`

### Experiments completed (1,269 runs total)

| Group | Runs | Date | Outcome |
|---|---|---|---|
| v4_direct_provider (main) | ~480 | this past week | The original heterogeneous-effects data |
| v4_direct_provider_baseline | ~90 | 2026-05-01 | Filled the no-noise baseline gaps |
| v4_direct_provider_neutral | ~90 | 2026-05-01 | Neutral framing pilot (ROLE A/B) |
| v4_direct_provider_targeted_* | ~150 | 2026-05-01 | Targeted-myth (k1, k2 noise) |
| v4_direct_provider_A1_partner_myth | 118 | 2026-05-01 → 02 | **Bootstrap rescue Δmean −7.7 → +1.5** |
| v4_direct_provider_A3_forced_reasoning | 111 | 2026-05-02 | **Closes §4.5 cross-model blind spot** |
| v4_direct_provider_A1A3_combined | 74 | 2026-05-02 | **Negative interaction — flagged for §3.5 caveat** |
| v4_direct_provider_targeted_bootstrap | 90 | 2026-05-02 | Cooperative content alone does NOT rescue |
| v4_direct_provider_A1_targeted_bootstrap | 59 | 2026-05-02 | Targeted+inj amplifies A1 by ~+2 |
| v4_direct_provider_A1_no_noise | 30 | 2026-05-02 | A1 produces NO effect at no-noise (rules out alternative) |
| v4_direct_provider_A1_adversarial_bootstrap | 60 | 2026-05-02 | Adversarial+inj STILL rescues — content direction doesn't matter |

### Analysis pipeline (extended to all 13 dirs)
- `INCLUDE_VERSIONS` updated in all five `analysis/build_*.py` scripts
- `cell_summaries/*.csv` regenerated (103 cells, 53 deltas, 942 myth-bearing runs scored for lag-1)
- Headline tables and 5 publication-ready figures regenerated

### Key infrastructure changes (be aware before changing things)
- `games/trust_game_noisy.py` PATCHED — added `_get_other_agent_last_myth()` helper and threaded `other_agent_last_myth` / `other_agent_last_myth_block` through all four `.format()` calls
- `config/experiments_noisy.yaml` EXTENDED — 5 new prompt templates (with reasoning / with partner-myth injection) + 8 new experiment_sets
- `references/bib/bakhtin2022.json` FIXED — removed empty author entry that was producing `[ et al., 2022]` in renders

## What's NOT done (next-session candidates)

In rough priority order:

1. **You read the §3 / §4 c_final files** and decide what to revise. Both are slightly over word target (3,061 prose words combined vs ~2,000 target). Compression options listed at the bottom of `analysis/MANUSCRIPT_DIFF.md`.

2. **Mario's abstract review.** The new abstract in `index.qmd` reflects the mechanism finding; circulate to the team.

3. **Embedding analysis on extended corpus** (was deferred — takes ~25 min). Run with: `python3 projects/neurips-2026-llm-ling-evo/analysis/build_embedding_convergence.py`. Adds nothing critical to §3 but completes the analysis suite.

4. **Optional: A1 × Claude × bootstrap.** The 2×3 mechanism factorial is GPT-5-Nano-only. A small Claude follow-up (~60 runs, ~25 min) would let §5.3 carry a cross-model claim instead of a single-model claim. The patch already supports any model; just need a new experiment_set in `config/experiments_noisy.yaml`. **Not necessary for the headline finding — the mechanism story is solid as-is — but tightens external validity.**

5. **Update TEAM_SUMMARY.md and MANUSCRIPT_DIFF.md** to reflect 2026-05-02 work (currently they describe 2026-04-30 state).

6. **Render verification.** Open the rebuilt PDF (`manuscript/_manuscript/index.pdf` or compile main.tex on Overleaf) and check page count, figure placement, citation rendering.

7. **Submission mechanics.** Verify NeurIPS template compliance, page limit, anonymisation, supplementary material structure.

## Key files to read in order

For the fresh session to get oriented:

1. **THIS FILE** — overall state
2. `projects/neurips-2026-llm-ling-evo/analysis/MORNING_BRIEFING.md` — full empirical findings + numbers
3. `projects/neurips-2026-llm-ling-evo/manuscript/_3-results.c_final.md` (especially §4.6) — the new mechanism subsection
4. `projects/neurips-2026-llm-ling-evo/manuscript/_4-discussion.c_final.md` (especially §5.2) — the new mechanism prose
5. `projects/neurips-2026-llm-ling-evo/manuscript/index.qmd` (line 12) — updated abstract
6. `projects/neurips-2026-llm-ling-evo/analysis/MANUSCRIPT_DIFF.md` — earlier section-by-section diff (some items now done, some still open)

## Key decisions that have been made

- **GPT-5-Nano-only scope** for the new mechanism work (Aron's call). Cross-model generalisation deferred to EMNLP follow-up.
- **§5.2 rewritten** to lead with the empirical visibility-mechanism finding rather than speculation. The previous "consolidation vs destabilisation as two faces of one mechanism" framing is gone.
- **A1+A3 combined depresses cooperation** — this is reportable as a §5.2 control and a §3.5 caveat (Claude reasoning prose carryover may be partly correlational), but **don't recommend the combination in §5.6 future-work**.
- **Adversarial-myth content + injection still rescues** — content direction is a small modulator, not a switch. Don't make a "cooperative content matters" claim.
- **Manuscript pipeline rule:** main agent doesn't write to `*.c_final.md` directly — invoke ms-writer subagent. Main agent CAN write to `*.b_draft.md` (humans-only nominally, but Aron has been explicitly asking for this). Main agent CAN edit `*.qmd`, configs, and analysis files freely.

## Key open questions worth pushing on

- Is §3 + §4 too long for NeurIPS body? 3,061 prose words combined; target ~2,000. Aron may want a tighter pass via ms-writer with stricter targets.
- Should §5.6 future-work bullet on partner-myth injection be removed (since it's now the §4.6 finding) and replaced with the next-step (A2 self-myth recall, or A1 × Claude)?
- Does the abstract emphasise mechanism enough? Currently both heterogeneous-effects framing AND mechanism finding are in there; might be cleaner to lead with mechanism only.

## How to resume work

```bash
cd /Users/aron/nips-linguistic-evolution-toolkit

# See current state of everything
cat projects/neurips-2026-llm-ling-evo/SESSION_HANDOFF.md
cat projects/neurips-2026-llm-ling-evo/analysis/MORNING_BRIEFING.md

# Re-render after any manuscript change
python3 projects/neurips-2026-llm-ling-evo/analysis/rebuild_overleaf.py

# Re-run analyses if new data lands
python3 projects/neurips-2026-llm-ling-evo/analysis/build_cell_summary.py
python3 projects/neurips-2026-llm-ling-evo/analysis/build_lag_and_lexicon.py
python3 projects/neurips-2026-llm-ling-evo/analysis/build_reason_coding.py
python3 projects/neurips-2026-llm-ling-evo/analysis/build_neologism_analysis.py
python3 projects/neurips-2026-llm-ling-evo/analysis/build_embedding_convergence.py  # ~25 min
python3 projects/neurips-2026-llm-ling-evo/analysis/build_headline_tables.py

# Run a new experiment set (template — change exp name + output subdir)
env PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct \
  /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py \
  EXPERIMENT_SET_NAME --workers 2 --output-subdir SUBDIR_NAME

# Invoke ms-writer (from main agent) on a populated b_draft
# Use Agent tool with subagent_type=ms-writer; see prior session for prompt examples
```

## Collaborator context (from earlier sessions)

- **Mario Giulianelli** — leading abstract framing
- **Edward Hughes** — collaborator, feedback on framing
- **Alexandra Pafford, Arabella Sinclair, Jane Sinclair** — wider review
- **Ivar Frisch** — primary experimenter for the original framework (the v4 corpus and noise pipeline are his); back from holiday Mon 2026-05-04
- Workflow: collaborators leave inline `/comment[…]` annotations directly in markdown (Overleaf migration deferred post-submission)

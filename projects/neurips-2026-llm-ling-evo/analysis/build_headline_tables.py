#!/usr/bin/env python3
"""Pull a focused subset of cell_summary.csv + deltas.csv into a single
markdown summary keyed by (model x noise_label) for the headline cells.

Produces:
  - HEADLINE_TABLES.md  — what Aron pastes into / cites in the §3 draft.

Headline cells = baseline (no_noise) + v4_direct_provider runs.
v4 is the cleanest, most recent data (this week), and includes positive
noise across all three models.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).parent
SUMMARIES = ANALYSIS_DIR / "cell_summaries"
OUT = ANALYSIS_DIR / "HEADLINE_TABLES.md"

HEADLINE_VERSIONS = {"v4_direct_provider"}
MODEL_ORDER = ["claude-sonnet-4.5", "gpt-5-nano"]
TASK_ORDER = ["game", "game_myth", "myth_game"]
NOISE_ORDER = ["no_noise", "negative_5", "positive", "bootstrap", "deterministic_max"]


def fmt(v, prec=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{prec}f}"


def cell_table(df: pd.DataFrame, model: str) -> str:
    sub = df[(df["model"] == model) & (df["version"].isin(HEADLINE_VERSIONS))].copy()
    if sub.empty:
        return f"_No data for {model}._\n"
    # Combine noise_label + informed for compactness
    sub["regime"] = sub.apply(
        lambda r: f"{r['noise_label']}{' (inf)' if r['informed'] else ''}",
        axis=1,
    )
    rows = []
    rows.append("| Regime | Task order | n | Mean | Median | Std |")
    rows.append("|---|---|---:|---:|---:|---:|")
    sub["regime_rank"] = sub["noise_label"].map(
        {n: i for i, n in enumerate(NOISE_ORDER)}
    ).fillna(99)
    sub["task_rank"] = sub["task_order"].map(
        {t: i for i, t in enumerate(TASK_ORDER)}
    ).fillna(99)
    sub = sub.sort_values(["regime_rank", "informed", "task_rank"])
    for _, r in sub.iterrows():
        rows.append(
            f"| {r['regime']} | {r['task_order']} | {int(r['n'])} | "
            f"{fmt(r['mean'])} | {fmt(r['median'])} | {fmt(r['std'])} |"
        )
    return "\n".join(rows) + "\n"


def deltas_table(df: pd.DataFrame, model: str) -> str:
    sub = df[(df["model"] == model) & (df["version"].isin(HEADLINE_VERSIONS))].copy()
    if sub.empty:
        return f"_No deltas for {model}._\n"
    sub["regime"] = sub.apply(
        lambda r: f"{r['noise_label']}{' (inf)' if r['informed'] else ''}",
        axis=1,
    )
    sub["regime_rank"] = sub["noise_label"].map(
        {n: i for i, n in enumerate(NOISE_ORDER)}
    ).fillna(99)
    sub["task_rank"] = sub["myth_task_order"].map(
        {t: i for i, t in enumerate(TASK_ORDER)}
    ).fillna(99)
    sub = sub.sort_values(["regime_rank", "informed", "task_rank"])
    rows = []
    rows.append(
        "| Regime | Myth order | n_g | n_m | Δmean (95% CI) | "
        "Var ratio myth/game (95% CI) | Class |"
    )
    rows.append("|---|---|---:|---:|---|---|---|")
    for _, r in sub.iterrows():
        dm = f"{fmt(r['delta_mean'])} [{fmt(r['delta_mean_ci_lo'])}, {fmt(r['delta_mean_ci_hi'])}]"
        vr = f"{fmt(r['var_ratio_myth_over_game'])} [{fmt(r['var_ratio_ci_lo'])}, {fmt(r['var_ratio_ci_hi'])}]"
        rows.append(
            f"| {r['regime']} | {r['myth_task_order']} | "
            f"{int(r['n_game'])} | {int(r['n_myth'])} | {dm} | {vr} | "
            f"**{r['classification']}** |"
        )
    return "\n".join(rows) + "\n"


def classification_breakdown(df: pd.DataFrame) -> str:
    """High-level: how many cells fall into each 3x3 classification."""
    counts = df["classification"].value_counts().to_dict()
    order = ["lift+consolidation", "consolidation", "lift", "null",
             "lift+destabilizing", "pure_noise", "harmful",
             "harmful-lock-in", "destabilizing", "missing"]
    rows = ["**3×3 classification breakdown across all "
            f"{len(df)} (model × noise × myth-order) cells:**\n"]
    rows.append("| Classification | n cells | Interpretation |")
    rows.append("|---|---:|---|")
    interp = {
        "lift+consolidation": "Mean ↑, variance ↓ — strongest myth signal",
        "consolidation": "Mean flat, variance ↓ — pure consolidation",
        "lift": "Mean ↑, variance flat — myth helps cooperation",
        "null": "Mean flat, variance flat — no detectable myth effect",
        "lift+destabilizing": "Mean ↑ but variance ↑ — gambles paid off on average",
        "pure_noise": "Mean flat, variance ↑",
        "harmful": "Mean ↓, variance flat — myth depresses cooperation",
        "harmful-lock-in": "Mean ↓, variance ↓ — myth locks in defection",
        "destabilizing": "Mean ↓, variance ↑ — myth makes it worse and noisier",
        "missing": "insufficient data to classify",
    }
    for k in order:
        if k in counts:
            rows.append(f"| **{k}** | {counts[k]} | {interp[k]} |")
    return "\n".join(rows) + "\n"


def consolidation_highlights(df: pd.DataFrame) -> str:
    """Pull out the cells where the consolidation/lift+consolidation pattern
    is statistically supported (variance CI excludes 1 below)."""
    df = df[df["version"].isin(HEADLINE_VERSIONS)].copy()
    pure_consolidation = df[df["classification"] == "consolidation"]
    lift_plus = df[df["classification"] == "lift+consolidation"]
    harmful = df[df["classification"].isin(
        ["harmful", "harmful-lock-in", "destabilizing", "lift+destabilizing"]
    )]

    rows = ["### Cells with statistically supported variance reduction (CI < 1)\n"]
    rows.append("**Pure consolidation** (mean flat, variance reduced):\n")
    if pure_consolidation.empty:
        rows.append("- _none in headline cells_\n")
    else:
        for _, r in pure_consolidation.iterrows():
            rows.append(
                f"- `{r['model']}` × `{r['noise_label']}`"
                f"{' (informed)' if r['informed'] else ''} × `{r['myth_task_order']}` — "
                f"Δmean {fmt(r['delta_mean'])} "
                f"[{fmt(r['delta_mean_ci_lo'])}, {fmt(r['delta_mean_ci_hi'])}], "
                f"var ratio {fmt(r['var_ratio_myth_over_game'])} "
                f"[{fmt(r['var_ratio_ci_lo'])}, {fmt(r['var_ratio_ci_hi'])}]"
            )

    rows.append("\n**Lift + consolidation** (mean up, variance reduced — the strongest signal):\n")
    if lift_plus.empty:
        rows.append("- _none in headline cells_\n")
    else:
        for _, r in lift_plus.iterrows():
            rows.append(
                f"- `{r['model']}` × `{r['noise_label']}`"
                f"{' (informed)' if r['informed'] else ''} × `{r['myth_task_order']}` — "
                f"Δmean {fmt(r['delta_mean'])} "
                f"[{fmt(r['delta_mean_ci_lo'])}, {fmt(r['delta_mean_ci_hi'])}], "
                f"var ratio {fmt(r['var_ratio_myth_over_game'])} "
                f"[{fmt(r['var_ratio_ci_lo'])}, {fmt(r['var_ratio_ci_hi'])}]"
            )

    rows.append("\n**Harmful / destabilizing cells** (worth honest reporting):\n")
    if harmful.empty:
        rows.append("- _none in headline cells_\n")
    else:
        for _, r in harmful.iterrows():
            rows.append(
                f"- `{r['model']}` × `{r['noise_label']}`"
                f"{' (informed)' if r['informed'] else ''} × `{r['myth_task_order']}` — "
                f"class **{r['classification']}**, "
                f"Δmean {fmt(r['delta_mean'])} "
                f"[{fmt(r['delta_mean_ci_lo'])}, {fmt(r['delta_mean_ci_hi'])}], "
                f"var ratio {fmt(r['var_ratio_myth_over_game'])} "
                f"[{fmt(r['var_ratio_ci_lo'])}, {fmt(r['var_ratio_ci_hi'])}]"
            )
    return "\n".join(rows) + "\n"


def lag_summary_section(lag: pd.DataFrame) -> str:
    """Render lag-1 cross-agent correlation findings."""
    rows = [
        "Cooperativity-lexicon score per myth round, scored on a small "
        "lexicon (collective + connected + giving) minus an uncooperative "
        "lexicon. Pearson r between Agent_A's score at round t and "
        "Agent_B's score at round t+1, computed per dyad and aggregated.\n",
        "`lag1_max` = the larger of the two directions per dyad — this is "
        "the direction the project's pilot Claude r=0.72 finding pointed at.\n",
    ]
    rows.append(
        "| Model | Noise | Inf | Task order | n | lag1_AB mean (CI) | "
        "lag1_BA mean (CI) | same-round mean (CI) | lag1_max mean (CI) | "
        "share \\|lag\\|>0.5 |"
    )
    rows.append(
        "|---|---|---|---|---:|---|---|---|---|---:|"
    )
    lag = lag.sort_values(["model", "noise_label", "informed", "task_order"])
    for _, r in lag.iterrows():
        def ci(col):
            return (
                f"{fmt(r[col + '_mean'], 2)} "
                f"[{fmt(r[col + '_ci_lo'], 2)}, {fmt(r[col + '_ci_hi'], 2)}]"
            )
        rows.append(
            f"| {r['model']} | {r['noise_label']} | "
            f"{'Y' if r['informed'] else 'N'} | {r['task_order']} | "
            f"{int(r['n_runs'])} | {ci('lag1_AB')} | {ci('lag1_BA')} | "
            f"{ci('same_round_AB')} | {ci('lag1_max')} | "
            f"{fmt(r['share_lag_gt_0_5'], 2)} |"
        )
    rows.append("")
    rows.append(
        "**Headline cells with the strongest lag-1 signal** "
        "(`lag1_max mean > 0.45` AND `share |lag| > 0.5` ≥ 0.5):\n"
    )
    strong = lag[
        (lag["lag1_max_mean"] > 0.45)
        & (lag["share_lag_gt_0_5"] >= 0.50)
    ]
    if strong.empty:
        rows.append("- _none_\n")
    else:
        for _, r in strong.iterrows():
            rows.append(
                f"- `{r['model']}` × `{r['noise_label']}`"
                f"{' (informed)' if r['informed'] else ''} × `{r['task_order']}` — "
                f"lag1_max mean **{fmt(r['lag1_max_mean'], 2)}** "
                f"[{fmt(r['lag1_max_ci_lo'], 2)}, {fmt(r['lag1_max_ci_hi'], 2)}], "
                f"{int(round(100 * r['share_lag_gt_0_5']))}% of dyads have |r|>0.5"
            )
    return "\n".join(rows) + "\n"


def neologism_section(summary: pd.DataFrame, examples: pd.DataFrame) -> str:
    rows = [
        "Detection: alphabetic tokens of length ≥6, not in "
        "`/usr/share/dict/words` after stripping common suffixes "
        "(s/es/ed/d/ing/er/est/ly/ies/ied/ness/ment/ful/less/ous/ish/able/ible). "
        "Counts per (run × agent × myth chain).\n",
    ]
    rows.append(
        "| Model | Noise | Inf | Task | n_chains | "
        "Coinages mean (med) | Max persistence rounds (med) | "
        "Rare share % (med) | Share of runs with coinage in `reason` text |"
    )
    rows.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    summary = summary.sort_values(
        ["model", "noise_label", "informed", "task_order"]
    )
    for _, r in summary.iterrows():
        rows.append(
            f"| {r['model']} | {r['noise_label']} | "
            f"{'Y' if r['informed'] else 'N'} | {r['task_order']} | "
            f"{int(r['n_chains'])} | "
            f"{fmt(r['coinages_mean'], 2)} ({fmt(r['coinages_median'], 1)}) | "
            f"{fmt(r['max_persistence_mean'], 2)} ({fmt(r['max_persistence_median'], 1)}) | "
            f"{fmt(r['rare_share_pct_mean'], 2)} ({fmt(r['rare_share_pct_median'], 2)}) | "
            f"{fmt(r['share_runs_coinage_in_reasons'], 2)} |"
        )
    rows.append("")
    rows.append(
        "**Headline finding:** `share_runs_coinage_in_reasons` is **0.00 across "
        "every cell** — no run has any myth-coinage appearing in any agent's "
        "game-reasoning text. This *does not replicate* the pilot observation "
        "that linguistic content from myths leaks into game reasoning.\n"
    )
    rows.append(
        "Coinages do persist *within* myth chains: Claude reaches max "
        "persistence of ~2.6 rounds (median 1–2) in negative-noise myth_game "
        "(informed) cells, with individual coinages like `lumina`, `aelara`, "
        "`arachnis`, `celestia`, `luminara` appearing in **all 10 rounds** of "
        "specific chains. GPT-5-Nano produces more coinages per chain "
        "(~5–6) but they persist for fewer rounds (~1.5–2).\n"
    )
    if not examples.empty:
        top = (
            examples.sort_values("rounds_persisted", ascending=False)
            .head(15)
            .copy()
        )
        rows.append("**Top 15 longest-persisting coinages:**\n")
        rows.append(
            "| Model | Noise | Inf | Task order | Agent | Coinage | Rounds | In `reason`? |"
        )
        rows.append("|---|---|---|---|---|---|---:|---:|")
        for _, r in top.iterrows():
            rows.append(
                f"| {r['model']} | {r['noise_label']} | "
                f"{'Y' if r['informed'] else 'N'} | {r['task_order']} | "
                f"{r['agent']} | `{r['coinage']}` | "
                f"{int(r['rounds_persisted'])} | "
                f"{'Y' if r['in_reasons'] else 'N'} |"
            )
    return "\n".join(rows) + "\n"


def reason_coding_section(rc: pd.DataFrame) -> str:
    rows = [
        "Lexical proxy for whether myth content enters game reasoning. "
        "For each round, the agent's `game_responses[agent].content` "
        "(plus `reasoning` if present) is tokenised and checked for two "
        "things:\n",
        "  1. **Theme hit** — contains a token from a small fixed myth-language "
        "lexicon (myth/story/spirit/elder/sacred/...).\n",
        "  2. **Own-myth hit** — contains a content word (length ≥5, non-stop) "
        "that appears in the agent's own myth chain *up to the previous round* "
        "(i.e. round-1 reasons can't credit round-1 myth content).\n",
    ]
    rows.append(
        "| Model | Noise | Inf | Task | n_reasons | Share theme hit | "
        "Mean theme hits | Share own-myth hit | Mean own-myth hits | "
        "Mean unique overlap | Runs w/ ≥1 own-myth hit |"
    )
    rows.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    rc = rc.sort_values(["model", "noise_label", "informed", "task_order"])
    for _, r in rc.iterrows():
        rows.append(
            f"| {r['model']} | {r['noise_label']} | "
            f"{'Y' if r['informed'] else 'N'} | {r['task_order']} | "
            f"{int(r['n_reasons'])} | {fmt(r['share_theme_hit'], 2)} | "
            f"{fmt(r['mean_theme_hits'], 2)} | "
            f"{fmt(r['share_own_myth_hit'], 2)} | "
            f"{fmt(r['mean_own_myth_hits'], 2)} | "
            f"{fmt(r['mean_own_myth_unique_overlap'], 2)} | "
            f"{fmt(r['share_runs_any_own_myth_hit'], 2)} |"
        )
    rows.append("")
    rows.append(
        "**Two findings, dramatically asymmetric across models:**\n"
        "1. **Claude:** ~78–82% of game-response prose references the agent's "
        "own myth vocabulary, with mean ~5–7 unique vocabulary overlaps per "
        "reason. **100% of runs** have at least one such reference. Theme-"
        "lexicon hits (story/spirit/elder/etc.) reach ~10–33% of reasons. "
        "Linguistic content from the myth channel demonstrably enters Claude's "
        "game reasoning.\n"
        "2. **GPT-5-Nano:** **0% across every cell** — but only because "
        "GPT-5-Nano produces *no visible reasoning prose at all* in this "
        "configuration. Its `game_responses[ag].content` is the bare JSON "
        "action (e.g. `{\"send\": 3}`) with empty `reasoning`. The question is "
        "**undecidable** for this model from the visible output. This is a "
        "limitation to flag in §4.5: the cross-task channel exists for models "
        "that emit reasoning prose; we cannot probe it where reasoning is "
        "either silent or hidden.\n"
    )
    return "\n".join(rows) + "\n"


def embedding_section(emb: pd.DataFrame) -> str:
    rows = [
        "Pairwise cosine between `Agent_1` and `Agent_2` myths at the same "
        "round (between-agent convergence). Embeddings via "
        "`sentence-transformers/all-mpnet-base-v2` (normalized).\n",
        "Slope = OLS slope of cos(Agent_1, Agent_2) on round number, with "
        "bootstrap 95% CI. Positive slope = convergence over rounds.\n",
    ]
    rows.append(
        "| Model | Noise | Inf | Task | n_runs | "
        "cos round 1 | cos round N | Slope per round (95% CI) |"
    )
    rows.append("|---|---|---|---|---:|---:|---:|---|")
    emb = emb.sort_values(["model", "noise_label", "informed", "task_order"])
    for _, r in emb.iterrows():
        slope = (
            f"{fmt(r['cos_between_slope_per_round'], 4)} "
            f"[{fmt(r['cos_between_slope_ci_lo'], 4)}, "
            f"{fmt(r['cos_between_slope_ci_hi'], 4)}]"
        )
        rows.append(
            f"| {r['model']} | {r['noise_label']} | "
            f"{'Y' if r['informed'] else 'N'} | {r['task_order']} | "
            f"{int(r['n_runs'])} | {fmt(r['cos_between_round1'], 3)} | "
            f"{fmt(r['cos_between_roundN'], 3)} | {slope} |"
        )
    rows.append("")
    rows.append(
        "**Strongest convergence cells** (slope CI excludes 0 above):\n"
    )
    strong = emb[
        (emb["cos_between_slope_ci_lo"] > 0)
    ].sort_values("cos_between_slope_per_round", ascending=False)
    if strong.empty:
        rows.append("- _none — convergence is not robustly positive in any cell at this scope_\n")
    else:
        for _, r in strong.iterrows():
            rows.append(
                f"- `{r['model']}` × `{r['noise_label']}`"
                f"{' (informed)' if r['informed'] else ''} × `{r['task_order']}` — "
                f"slope **{fmt(r['cos_between_slope_per_round'], 4)}/round** "
                f"[{fmt(r['cos_between_slope_ci_lo'], 4)}, "
                f"{fmt(r['cos_between_slope_ci_hi'], 4)}]"
            )
    return "\n".join(rows) + "\n"


def main():
    cell_summary = pd.read_csv(SUMMARIES / "cell_summary.csv")
    deltas = pd.read_csv(SUMMARIES / "deltas.csv")
    # "null" is a valid classification label; overwrite with a string-typed reload
    # so pandas doesn't turn it into NaN.
    deltas["classification"] = pd.read_csv(
        SUMMARIES / "deltas.csv", dtype={"classification": str}, keep_default_na=False
    )["classification"].values

    parts = []
    parts.append("# Headline cell summary — NeurIPS 2026 submission\n")
    parts.append(
        "Generated by "
        "`projects/neurips-2026-llm-ling-evo/analysis/build_headline_tables.py`\n"
        "from `cell_summaries/cell_summary.csv` + `cell_summaries/deltas.csv`.\n\n"
        "Scope: **this week's runs only** "
        "(`noise_experiments/v4_direct_provider/`, Claude + ChatGPT/gpt-5-nano).\n"
        "Gemini and earlier (v1/v2/v3) data are excluded from this analysis.\n\n"
        "Balance metric = mean of Agent_1 and Agent_2 cumulative balances at round 10.\n"
        "Bootstrap CIs use 2,000 resamples (RNG seed 42).\n"
    )
    parts.append("\n---\n")
    parts.append("## Top-line findings\n")
    parts.append(classification_breakdown(deltas))
    parts.append("\n")
    parts.append(consolidation_highlights(deltas))

    # Lag-1 cross-agent correlations (A3).
    lag_path = SUMMARIES / "lag_summary.csv"
    if lag_path.exists():
        parts.append("\n---\n")
        parts.append("## Lag-1 cross-agent cooperativity correlations (A3)\n")
        parts.append(lag_summary_section(pd.read_csv(lag_path)))

    # Neologism / coinage findings (A9).
    neo_path = SUMMARIES / "neologism_summary.csv"
    if neo_path.exists():
        parts.append("\n---\n")
        parts.append("## Neologism / coinage findings (A9)\n")
        parts.append(neologism_section(
            pd.read_csv(neo_path),
            pd.read_csv(SUMMARIES / "neologisms_examples.csv")
            if (SUMMARIES / "neologisms_examples.csv").exists()
            else pd.DataFrame(),
        ))

    # Reason-field coding (A6).
    rc_path = SUMMARIES / "reason_coding_summary.csv"
    if rc_path.exists():
        parts.append("\n---\n")
        parts.append("## Reason-field coding — does myth content enter game reasoning? (A6)\n")
        parts.append(reason_coding_section(pd.read_csv(rc_path)))

    # Embedding convergence (A4).
    emb_path = SUMMARIES / "embedding_summary.csv"
    if emb_path.exists():
        parts.append("\n---\n")
        parts.append("## Embedding-based myth convergence (A4)\n")
        parts.append(embedding_section(pd.read_csv(emb_path)))

    parts.append("\n---\n")
    parts.append("## Per-model cell summaries\n")
    for model in MODEL_ORDER:
        parts.append(f"\n### {model}\n")
        parts.append("**Cumulative balance at r10 by cell:**\n")
        parts.append(cell_table(cell_summary, model))
        parts.append("\n**Myth effect (myth task order vs game-only) — bootstrap 95% CIs:**\n")
        parts.append(deltas_table(deltas, model))

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

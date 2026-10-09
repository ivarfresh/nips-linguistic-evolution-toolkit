"""Do myths change a mixed group's margin over its parts? (paper Table s-interaction)

For each mixed group and myth task order: (mixed - composition-weighted parts) under
the myth task order minus the same difference in game-only play. The contrast runs
over six groups (the mixed group and its two single-model parts, under both task
orders) with the weights of analyses/mixed_vs_average.py:

    +1 mix_myth  - w_a part_a_myth - w_b part_b_myth  - 1 mix_game + w_a part_a_game + w_b part_b_game

Reports the Welch t-test on the contrast (Welch-Satterthwaite df), its 95%
t-interval (the interval printed in the paper's table), the Holm-adjusted p
across the 18 cells, and, as extra columns, a 95% percentile bootstrap interval
(runs resampled within each of the six groups). Positive: myths raise the mix
relative to its parts.

Inputs are the per-run tables committed with Table 1 at n = 10, so no raw JSON
is read:
  docs/figures/mixed_vs_average_n10_20261001/dyad_decisions.csv          (last round, total / 2)
  docs/figures/mixed_vs_average_n10_20261001/population_agent_finals.csv (mean over 8 agents)
Outputs: docs/figures/mixed_vs_average_interaction_20261009/{interaction.csv, table_rows.tex}.

Run from the repo root: python analyses/mixed_vs_average_interaction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses.mixed_vs_average import SPECS, holm, welch_contrast  # noqa: E402

N10 = ROOT / "docs/figures/mixed_vs_average_n10_20261001"
OUT = ROOT / "docs/figures/mixed_vs_average_interaction_20261009"
MYTH_ORDERS = ["game_myth", "myth_game"]
B = 20000
SEED = 20261009


def per_run_values() -> pd.DataFrame:
    d = pd.read_csv(N10 / "dyad_decisions.csv")
    last = d.sort_values("round").groupby("path").tail(1)
    dyads = last.assign(v=last["total_balance"] / 2)[["path", "composition", "task_order", "v"]]
    a = pd.read_csv(N10 / "population_agent_finals.csv")
    pops = (a.groupby(["path", "composition", "task_order"])["final_balance"].mean()
            .reset_index().rename(columns={"final_balance": "v"}))
    return pd.concat([dyads, pops], ignore_index=True)


def fmt_p(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def write_table_rows(table: pd.DataFrame) -> None:
    """LaTeX rows for tab:s-interaction; bold = Holm p < 0.05 (the caption's rule)."""
    cell = {(r.group, r.task_order): r for r in table.itertuples()}
    lines = []
    for i, (label, _, _) in enumerate(SPECS):
        parts = []
        for to in MYTH_ORDERS:
            r = cell[(label, to)]
            diff = f"{r.diff:.1f}".replace("-", "$-$")
            value = rf"\textbf{{{diff}}}" if r.welch_p_holm < 0.05 else diff
            lo, hi = (f"{x:.1f}".replace("-", "$-$") for x in (r.ci_lo, r.ci_hi))
            holm_p = "1" if r.welch_p_holm >= 1 else fmt_p(r.welch_p_holm)
            parts.append(f"{value} [{lo}, {hi}] & {fmt_p(r.welch_p)} / {holm_p}")
        lines.append(f"{label.replace(' (dyad)', ' dyad')} & " + " & ".join(parts) + r" \\")
        if i == 2:
            lines.append(r"\midrule")
    (OUT / "table_rows.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    rng = np.random.default_rng(SEED)
    runs = per_run_values()
    get = lambda comp, to: runs[(runs.composition == comp) & (runs.task_order == to)]["v"].to_numpy()
    rows = []
    for label, comp, parts in SPECS:
        g_mix = get(comp, "game")
        g_parts = [(get(c, "game"), w) for c, w in parts]
        for to in MYTH_ORDERS:
            m_mix = get(comp, to)
            m_parts = [(get(c, to), w) for c, w in parts]
            groups = ([(m_mix, 1.0)] + [(h, -w) for h, w in m_parts]
                      + [(g_mix, -1.0)] + [(h, w) for h, w in g_parts])
            assert all(len(g) == 10 for g, _ in groups), (comp, to, [len(g) for g, _ in groups])
            est = sum(c * g.mean() for g, c in groups)
            boot = sum(c * rng.choice(g, (B, len(g))).mean(1) for g, c in groups)
            t, df, p = welch_contrast(groups)
            half = stats.t.ppf(0.975, df) * est / t
            diff_myth = m_mix.mean() - sum(w * h.mean() for h, w in m_parts)
            diff_game = g_mix.mean() - sum(w * h.mean() for h, w in g_parts)
            rows.append({
                "group": label, "task_order": to,
                "parts": " / ".join(f"{c} (w={w:g})" for c, w in parts),
                "diff_myth_order": diff_myth, "diff_game_only": diff_game, "diff": est,
                "ci_lo": est - half, "ci_hi": est + half,
                "boot_ci_lo": np.percentile(boot, 2.5), "boot_ci_hi": np.percentile(boot, 97.5),
                "welch_t": t, "welch_df": df, "welch_p": p,
            })
    out = pd.DataFrame(rows)
    out["welch_p_holm"] = holm(out["welch_p"].to_numpy())
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "interaction.csv", index=False)
    write_table_rows(out)
    wide = out.assign(cell=out.apply(
        lambda r: f"{r['diff']:+.1f} [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}] p={r.welch_p:.3f}/{r.welch_p_holm:.3f}", axis=1))
    print(wide.pivot(index="group", columns="task_order", values="cell")
          .reindex([s[0] for s in SPECS]).to_string())
    print(f"\n{int((out.welch_p_holm < 0.05).sum())} of {len(out)} cells Holm-significant; "
          f"{int((out['diff'] > 0).sum())} positive")


if __name__ == "__main__":
    main()

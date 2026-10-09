"""Mixed groups compared with the composition-weighted mean of their single-model parts.

For each mixed group and task order: final resources per agent in the mixed
runs minus sum_i w_i * R_i over the matching single-model groups (k/8 and
(8-k)/8 for a population with k agents of one model; 1/2 each for a dyad).
Reports a 95% percentile bootstrap interval (runs resampled within each group)
and a Welch t-test on the same linear contrast (Welch-Satterthwaite df),
Holm-corrected across all cells of the table.

Inputs are the per-run tables already in the repo, so no raw JSON is read:
  docs/figures/mixed_model_dyads_20260917/decisions.csv   (last round, total / 2)
  docs/figures/mixed_model_populations_20260918/agent_finals.csv (mean over 8 agents)

Run from the repo root: python analyses/mixed_vs_average.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "docs/figures"
OUT = FIGS / "mixed_vs_average_20260930"
TASK_ORDERS = ["game", "game_myth", "myth_game"]
B = 20000
SEED = 20260930


def per_run_values(dyads_csv: Path = FIGS / "mixed_model_dyads_20260917/decisions.csv",
                   populations_csv: Path = FIGS / "mixed_model_populations_20260918/agent_finals.csv") -> pd.DataFrame:
    """Final resources per agent per run: dyads from the last round (total / 2), populations as the mean over 8 agents."""
    d = pd.read_csv(dyads_csv)
    last = d.sort_values("round").groupby("path").tail(1)
    dyads = last.assign(v=last["total_balance"] / 2)[["path", "composition", "task_order", "v"]]
    a = pd.read_csv(populations_csv)
    pops = (a.groupby(["path", "composition", "task_order"])["final_balance"].mean()
            .reset_index().rename(columns={"final_balance": "v"}))
    return pd.concat([dyads, pops], ignore_index=True)


SPECS = (
    [("Sonnet + GPT (dyad)", "Sonnet+GPT", [("Sonnet+Sonnet", .5), ("GPT+GPT", .5)]),
     ("Sonnet + Gemini (dyad)", "Sonnet+Gemini", [("Sonnet+Sonnet", .5), ("Gemini+Gemini", .5)]),
     ("Gemini + GPT (dyad)", "Gemini+GPT", [("Gemini+Gemini", .5), ("GPT+GPT", .5)])]
    + [(f"{k} Gemini + {8 - k} GPT", f"{k} Gemini + {8 - k} GPT", [("8 Gemini", k / 8), ("8 GPT", (8 - k) / 8)])
       for k in (1, 2, 4)]
    + [(f"{k} GPT + {8 - k} Sonnet", f"{k} GPT + {8 - k} Sonnet", [("8 GPT", k / 8), ("8 Sonnet", (8 - k) / 8)])
       for k in (1, 2, 4)]
)


def welch_contrast(groups: list[tuple[np.ndarray, float]]) -> tuple[float, float, float]:
    """Welch t-test of sum_j c_j * mean_j = 0. Zero-variance groups add no variance or df."""
    est = sum(c * g.mean() for g, c in groups)
    terms = [c ** 2 * g.var(ddof=1) / len(g) for g, c in groups]
    se2 = sum(terms)
    if se2 == 0:
        return np.nan, np.nan, np.nan
    df = se2 ** 2 / sum(t ** 2 / (len(g) - 1) for t, (g, _) in zip(terms, groups) if t > 0)
    t = est / np.sqrt(se2)
    return t, df, 2 * stats.t.sf(abs(t), df)


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values; NaNs pass through."""
    adj = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    idx = np.flatnonzero(ok)[np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (len(idx) - rank) * p[i]))
        adj[i] = running
    return adj


def main() -> None:
    rng = np.random.default_rng(SEED)
    runs = per_run_values()
    get = lambda comp, to: runs[(runs.composition == comp) & (runs.task_order == to)]["v"].to_numpy()
    rows = []
    for label, comp, parts in SPECS:
        for to in TASK_ORDERS:
            m = get(comp, to)
            hs = [(get(c, to), w) for c, w in parts]
            assert len(m) and all(len(h) for h, _ in hs), (comp, to)
            expected = sum(w * h.mean() for h, w in hs)
            boot = rng.choice(m, (B, len(m))).mean(1) - sum(
                w * rng.choice(h, (B, len(h))).mean(1) for h, w in hs)
            t, df, p = welch_contrast([(m, 1.0)] + [(h, -w) for h, w in hs])
            rows.append({
                "group": label, "task_order": to, "n_mixed": len(m),
                "parts": " / ".join(f"{c} (w={w:g}, n={len(get(c, to))})" for c, w in parts),
                "mixed_mean": m.mean(), "mixed_sd": m.std(ddof=1), "expected": expected,
                "diff": m.mean() - expected,
                "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
                "welch_t": t, "welch_df": df, "welch_p": p,
            })
    out = pd.DataFrame(rows)
    out["welch_p_holm"] = holm(out["welch_p"].to_numpy())
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "mixed_vs_average.csv", index=False)
    wide = out.assign(cell=out.apply(lambda r: f"{r['diff']:+.1f} [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]", axis=1))
    print(wide.pivot(index="group", columns="task_order", values="cell")
          .reindex([s[0] for s in SPECS]).to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build scorecard_rows.csv for the spread lens from spread.py's per-dataset tables.

Status rules (in order):
  - design / ceiling flags set per finding (label almost all one value, coop without spread,
    moral label that never changes within an agent, no unseen same-family myth);
  - holm_p < 0.05 -> yes, except regressions with < 10 run clusters or within-agent carryover
    resting on < 10 agents whose label changes, capped at suggestive;
  - raw p < 0.05 -> suggestive;
  - Wilcoxon over <= 5 runs (floor p = 0.0625): suggestive if every run has the same sign,
    else underpowered;
  - otherwise no detectable effect.
Holm family: this lens's primary tests within dataset x setting x population x task order
(placebo rows are checks and excluded from Holm). Task order is written into `finding`.
Effects in percentage points for uptake/adoption; coefficients for regressions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ORDER = {"myth_game": "myth→game", "game_myth": "game→myth", "both": "both orders"}
COLS = ["dataset", "setting", "population", "family", "finding", "effect", "ci_low", "ci_high", "p", "holm_p",
        "n_runs", "n_obs", "status", "note"]


def boot_ci(children: pd.DataFrame, metric: str, n=5000):
    vals = children.groupby("run_id")[metric].mean().dropna().to_numpy()
    if len(vals) < 2:
        return np.nan, np.nan
    d = np.random.default_rng(20260930).choice(vals, size=(n, len(vals))).mean(axis=1)
    return tuple(np.percentile(d, [2.5, 97.5]))


def fam_label(fam, pfam, exposure):
    if fam == "all":
        return "all", f"{exposure}"
    return fam, f"{fam} shown {pfam}"


def rows_for(ds: str) -> list[dict]:
    out = []
    D = HERE / ds
    # ---- word adoption
    ch = pd.read_csv(D / "word_uptake_children.csv")
    ch["setting"] = [f"{s}-agent {'mixed' if m else 'homogeneous'}" for s, m in zip(ch["size"], ch["mixed"])]
    ch["exposure"] = np.where(ch["family"] == ch["parent_family"], "same family", "other family")
    w = pd.read_csv(D / "word_adoption_by_stratum.csv")
    for r in w.itertuples(index=False):
        if r.population.startswith("all ") and r.setting == "8-agent mixed" and ds == "frontier":
            continue  # identical to the single mixed population
        sub = ch[(ch["setting"] == r.setting) & (ch["exposure"] == r.exposure)]
        if not r.population.startswith("all "):
            sub = sub[(sub["population"] == r.population) & (sub["family"] == r.family) & (sub["parent_family"] == r.parent_family)]
        if r.task_order != "both":
            sub = sub[sub["task_order"] == r.task_order]
        lo, hi = boot_ci(sub, "adopt_excess")
        fam, what = fam_label(r.family, r.parent_family, r.exposure)
        out.append(dict(dataset=ds, setting=r.setting, population=r.population, family=fam,
                        finding=f"word adoption shown − unseen, {what} [{ORDER[r.task_order]}]",
                        effect=100 * r.adopt_excess_mean, ci_low=100 * lo, ci_high=100 * hi, p=r.adopt_excess_p,
                        n_runs=r.n_runs, n_obs=r.n_children, _order=r.task_order, _kind="wilcoxon",
                        _pos=r.adopt_excess_runs_positive,
                        note=f"shown {100 * r.adopt_parent_mean:.1f}% vs unseen {100 * r.adopt_null_mean:.1f}% (±{100 * r.adopt_excess_sd:.1f} sd over runs); "
                             f"{r.adopt_excess_runs_positive}/{r.n_runs} runs positive"))
    # ---- moral label uptake, placebo, R3
    m = pd.read_csv(D / "moral_uptake_by_stratum.csv")
    for r in m.itertuples(index=False):
        if r.population.startswith("all ") and r.setting == "8-agent mixed" and ds == "frontier":
            continue
        if r.family == "all" and not r.population.startswith("all ") and r.setting == "2-agent mixed":
            continue
        fam, what = fam_label(r.family, r.parent_family, r.exposure)
        o = ORDER[r.task_order]
        special = "not estimable: ceiling" if r.label_ceiling else None
        base = dict(dataset=ds, setting=r.setting, population=r.population, family=fam, _order=r.task_order)
        out.append(dict(base, finding=f"moral label uptake shown − unseen, {what} [{o}]",
                        effect=100 * r.uptake_effect, ci_low=100 * r.uptake_ci_low, ci_high=100 * r.uptake_ci_high,
                        p=r.uptake_p, n_runs=r.uptake_n_runs, n_obs=r.uptake_n_obs, _kind="wilcoxon",
                        _pos=r.uptake_runs_positive, _special=special,
                        note=f"±{100 * r.uptake_sd:.1f} sd over runs; {r.uptake_runs_positive}/{r.uptake_n_runs} runs positive; "
                             f"child-level permutation p {r.uptake_perm_p:.3g} (anti-conservative, not used for status)"
                             + ("; labels ≥95% one value" if r.label_ceiling else "")))
        if r.setting.startswith("8-agent") and pd.notna(r.future_effect):
            note = (f"author's next myth (never shown) vs unseen; {r.future_runs_positive}/{r.future_n_runs} runs positive; "
                    f"author had read the reader's previous myth in {100 * r.future_author_read_child_share:.0f}% of cases; "
                    "placebo as large as the shown effect = match not specific to the shown text")
            out.append(dict(base, finding=f"PLACEBO future myth − unseen, {what} [{o}]", effect=100 * r.future_effect,
                            ci_low=np.nan, ci_high=np.nan, p=r.future_p, n_runs=r.future_n_runs, n_obs=r.future_n_obs,
                            _kind="placebo", _pos=r.future_runs_positive, _special=special, note=note))
            if r.setting == "8-agent mixed" and pd.notna(getattr(r, "matched_excess_mean", np.nan)):
                out.append(dict(base, finding=f"PLACEBO-CHECK uptake vs partner-matched unseen, {what} [{o}]",
                                effect=100 * r.matched_excess_mean, ci_low=np.nan, ci_high=np.nan, p=r.matched_excess_p,
                                n_runs=r.matched_excess_n_runs, n_obs=r.matched_excess_n_obs, _kind="placebo",
                                _special=special,
                                note=f"unseen authors who also just played the reader's family; future-myth version "
                                     f"{100 * r.future_matched_excess_mean:+.1f} pts"))
        if pd.notna(getattr(r, "r3_effect", np.nan)):
            out.append(dict(base, finding=f"moral label uptake, R3-adjusted (author ≠ current partner, t−1 pair game controlled), {what} [{o}]",
                            effect=100 * r.r3_effect, ci_low=100 * r.r3_ci_low, ci_high=100 * r.r3_ci_high, p=r.r3_p,
                            n_runs=r.r3_n_runs, n_obs=r.r3_n_obs, _kind="regression", _special=special,
                            note="child-level OLS intercept, centred covariates, SE clustered by run; headline without the future myth"))
    if ds == "frontier":  # the pooled 8-agent homogeneous row without Sol, whose labels are 96% 'be fair'
        from scipy import stats
        u = pd.read_csv(D / "moral_uptake_children.csv")
        u = u[(u["setting"] == "8-agent homogeneous") & u["family"].isin(["Opus", "GeminiPro"])]
        for o, g in u.groupby("task_order"):
            per = g.groupby("run_id")["same_label_excess"].mean().round(10)
            lo, hi = boot_ci(g, "same_label_excess")
            out.append(dict(dataset=ds, setting="8-agent homogeneous", population="8 Opus + 8 GeminiPro runs (Sol excluded)",
                            family="all", _order=o, _kind="wilcoxon", _pos=int((per > 0).sum()),
                            finding=f"moral label uptake shown − unseen, same family [{ORDER[o]}]",
                            effect=100 * per.mean(), ci_low=100 * lo, ci_high=100 * hi, p=stats.wilcoxon(per).pvalue,
                            n_runs=len(per), n_obs=len(g),
                            note=f"±{100 * per.std(ddof=1):.1f} sd over runs; {int((per > 0).sum())}/{len(per)} runs positive"))
        for o in ("myth_game", "game_myth"):
            out.append(dict(dataset=ds, setting="8-agent mixed", population="2 GeminiPro + 3 Opus + 3 Sol", family="GeminiPro",
                            finding=f"moral label uptake shown − unseen, GeminiPro shown GeminiPro [{ORDER[o]}]",
                            _order=o, _kind="design", _special="not testable by design",
                            note="2 GeminiPro per run: no unseen same-family myth"))
    # ---- carryover (within agent)
    c = pd.read_csv(D / "carryover_by_family.csv")
    for r in c.itertuples(index=False):
        what = "send/5" if r.role == "investor" else "return proportion"
        for t, name in (("own_gen", "own"), ("shown_gen", "shown")):
            coef = getattr(r, f"{t}_coef", np.nan)
            varies = r.n_agents_own_varies if t == "own_gen" else r.n_agents_shown_varies
            special = None
            if (r.coop_sd < 0.02) or pd.isna(coef) and r.coop_sd < 0.05:
                special = "not estimable: ceiling"
            elif varies < 3:
                special = "not estimable: ceiling"
            elif pd.isna(coef):
                special = "underpowered"
            out.append(dict(dataset=ds, setting=r.setting, population=r.population, family=r.family, _order=r.task_order,
                            finding=f"carryover: {name} myth 'be generous' (vs fair) → next {what}, within agent [{ORDER[r.task_order]}]",
                            effect=coef if special is None else np.nan,
                            ci_low=getattr(r, f"{t}_ci_low", np.nan) if special is None else np.nan,
                            ci_high=getattr(r, f"{t}_ci_high", np.nan) if special is None else np.nan,
                            p=getattr(r, f"{t}_p", np.nan) if special is None else np.nan,
                            n_runs=r.n_runs, n_obs=r.n_obs, _kind="regression", _special=special, _varies=varies, _perfam=True,
                            note=f"agent-within-run + round FE, own last move; {varies} agents whose label changes; "
                                 f"coop sd {r.coop_sd:.3f}"))
    # ---- reverse
    v = pd.read_csv(D / "reverse_by_family.csv")
    for r in v.itertuples(index=False):
        what = "send/5" if r.role == "investor" else "return proportion"
        special = None
        if r.coop_sd < 0.02 or pd.isna(r.coef) and r.coop_sd < 0.05:
            special = "not estimable: ceiling"
        elif r.gen_after_share <= 0.05 or r.gen_after_share >= 0.95:
            special = "not estimable: ceiling"
        elif pd.isna(r.coef):
            special = "underpowered"
        out.append(dict(dataset=ds, setting=r.setting, population=r.population, family=r.family, _order=r.task_order,
                        finding=f"reverse: +0.1 {what} in a game → P(next myth 'be generous') [{ORDER[r.task_order]}]",
                        effect=r.coef / 10 if special is None else np.nan, ci_low=r.ci_low / 10 if special is None else np.nan,
                        ci_high=r.ci_high / 10 if special is None else np.nan, p=r.p if special is None else np.nan,
                        n_runs=r.n_runs, n_obs=r.n_obs, _kind="regression", _special=special,
                        _perfam=True,
                        note=f"LPM, effect per +0.1 of {what} (coop sd in cell {r.coop_sd:.3f}), own previous label, agent-within-run + round FE; "
                             f"run-FE (September spec) {r.coef_runfe / 10:+.3f} per 0.1 (p {r.p_runfe:.3g}); generous-after share {r.gen_after_share:.2f}"))
    # ---- label drift
    g = pd.read_csv(D / "moral_generous_change_r1_r10.csv")
    for r in g.itertuples(index=False):
        diff_pos = np.nan
        out.append(dict(dataset=ds, setting=r.setting, population=r.population, family=r.family, _order=r.task_order,
                        finding=f"'be generous' share, round 10 − round 1 [{ORDER[r.task_order]}]",
                        effect=100 * r.change_mean, ci_low=np.nan, ci_high=np.nan, p=r.p, n_runs=r.n_runs, n_obs=r.n_myths,
                        _kind="wilcoxon_nosign", _special="not estimable: ceiling" if (r.gen_share_all <= 0.03 or r.gen_share_all >= 0.97) else None,
                        note=f"round 1 {100 * r.gen_r1:.0f}% → round 10 {100 * r.gen_r10:.0f}% (±{100 * r.change_sd:.0f} sd over runs); "
                             f"overall generous {100 * r.gen_share_all:.0f}% / fair {100 * r.fair_share_all:.0f}% / cautious {100 * r.cautious_share_all:.0f}%"))
    # ---- pooled carryover / reverse (September-comparable, all families)
    for fn, kind in (("carryover_models_pooled.csv", "carry"), ("reverse_models_pooled.csv", "rev")):
        if not (D / fn).exists():
            continue
        t = pd.read_csv(D / fn)
        if kind == "carry":
            t = t[(t["fe"] == "agent") & (t["model"] == "own + shown label, own lag") & (t["level"] == "be generous")]
            dsf = D / "carryover_models_pooled_deepseek__deepseek-v4-flash.csv"
            dsk = {}
            if dsf.exists():
                x = pd.read_csv(dsf)
                x = x[(x["fe"] == "agent") & (x["model"] == "own + shown label, own lag") & (x["level"] == "be generous")]
                dsk = {(a.setting, a.role, a.predictor): (a.coef, a.p) for a in x.itertuples(index=False)}
            for r in t.itertuples(index=False):
                dk = dsk.get((r.setting, r.role, r.predictor))
                what = "send/5" if r.role == "investor" else "return proportion"
                src = "own" if r.predictor == "own_label" else "shown"
                out.append(dict(dataset=ds, setting=r.setting, population=f"all {r.setting} runs" if r.setting != "all settings" else "all runs",
                                family="all", _order="both", _kind="regression",
                                finding=f"carryover (September pooled spec): {src} myth 'be generous' → next {what}, within agent [both orders]",
                                effect=r.coef, ci_low=r.ci_low, ci_high=r.ci_high, p=r.p, n_runs=r.n_runs, n_obs=r.n_decisions,
                                _ds_p=dk[1] if dk else np.nan,
                                note="mc.carryover_models, agent-within-run + round FE, all families pooled as committed in September"
                                     + (f"; DeepSeek labels {dk[0]:+.4f} (p {dk[1]:.3g})" if dk else "")))
        else:
            for r in t.itertuples(index=False):
                what = "send/5" if r.role == "investor" else "return proportion"
                out.append(dict(dataset=ds, setting=r.setting, population=f"all {r.setting} runs" if r.setting != "all settings" else "all runs",
                                family="all", _order="both", _kind="regression",
                                finding=f"reverse (September pooled spec): {what} → next myth 'be generous' [both orders]",
                                effect=r.coef_coop_on_generous_after, ci_low=r.ci_low, ci_high=r.ci_high, p=r.p,
                                n_runs=np.nan, n_obs=r.n, note="mc.reverse_models, run + round FE, all families pooled"))
    return out


def status(r) -> str:
    if isinstance(r.get("_special"), str):
        return r["_special"]
    p, h, n = r.get("p"), r.get("holm_p"), r.get("n_runs")
    if pd.isna(p):
        return "underpowered"
    if pd.notna(h) and h < 0.05:
        if pd.notna(r.get("_ds_p")) and r["_ds_p"] >= 0.05:
            return "suggestive"  # second judge does not agree (September's bar: both judges)
        if r["_kind"] == "regression" and pd.notna(n) and n < 10:
            return "suggestive"
        if r.get("_perfam") is True and pd.notna(n) and n < 15:
            return "suggestive"  # per-family regression on fewer than 15 run clusters
        if pd.notna(r.get("_varies")) and r["_varies"] < 10:
            return "suggestive"  # within-agent estimate resting on fewer than 10 agents whose label changes
        return "yes"
    if p < 0.05:
        return "suggestive"
    if r["_kind"].startswith("wilcoxon") or r["_kind"] == "placebo":
        if pd.notna(n) and n <= 5:
            pos = r.get("_pos")
            if pd.notna(pos) and r["_kind"] != "wilcoxon_nosign" and (pos == n or pos == 0) and n >= 4:
                return "suggestive"
            return "underpowered"
    return "no detectable effect"


def robustness_rows(ds: str, tag: str = "_deepseek__deepseek-v4-flash") -> list[dict]:
    """DeepSeek V4 Flash labels (robustness only; frontier DeepSeek was served with hidden reasoning):
    label uptake for pooled and per-family 8-agent strata, pooled carryover and reverse. Not in Holm."""
    D, out = HERE / ds, []
    if not (D / f"moral_uptake_by_stratum{tag}.csv").exists():
        return out
    m = pd.read_csv(D / f"moral_uptake_by_stratum{tag}.csv")
    for r in m.itertuples(index=False):
        if r.population.startswith("all ") and r.setting == "8-agent mixed" and ds == "frontier":
            continue
        if not (r.population.startswith("all ") or r.setting.startswith("8-agent")) or (r.family == "all" and not r.population.startswith("all ")):
            continue
        fam, what = fam_label(r.family, r.parent_family, r.exposure)
        out.append(dict(dataset=ds, setting=r.setting, population=r.population, family=fam, _order=r.task_order,
                        finding=f"ROBUSTNESS (DeepSeek labels) moral label uptake shown − unseen, {what} [{ORDER[r.task_order]}]",
                        effect=100 * r.uptake_effect, ci_low=100 * r.uptake_ci_low, ci_high=100 * r.uptake_ci_high,
                        p=r.uptake_p, n_runs=r.uptake_n_runs, n_obs=r.uptake_n_obs, _kind="placebo",
                        _pos=r.uptake_runs_positive, _special="not estimable: ceiling" if r.label_ceiling else None,
                        note=f"future-myth placebo {100 * r.future_effect:+.1f} pts; " + (f"R3-adjusted {100 * r.r3_effect:+.1f} (p {r.r3_p:.3g})" if pd.notna(getattr(r, 'r3_effect', np.nan)) else "")))
    t = pd.read_csv(D / f"carryover_models_pooled{tag}.csv")
    t = t[(t["fe"] == "agent") & (t["model"] == "own + shown label, own lag") & (t["level"] == "be generous") & (t["setting"] == "all settings")]
    for r in t.itertuples(index=False):
        what = "send/5" if r.role == "investor" else "return proportion"
        out.append(dict(dataset=ds, setting="all settings", population="all runs", family="all", _order="both", _kind="placebo",
                        finding=f"ROBUSTNESS (DeepSeek labels) carryover pooled: {'own' if r.predictor == 'own_label' else 'shown'} myth 'be generous' → next {what} [both orders]",
                        effect=r.coef, ci_low=r.ci_low, ci_high=r.ci_high, p=r.p, n_runs=r.n_runs, n_obs=r.n_decisions,
                        note="mc.carryover_models, agent-within-run + round FE"))
    t = pd.read_csv(D / f"reverse_models_pooled{tag}.csv")
    for r in t[t["setting"] == "all settings"].itertuples(index=False):
        what = "send/5" if r.role == "investor" else "return proportion"
        out.append(dict(dataset=ds, setting="all settings", population="all runs", family="all", _order="both", _kind="placebo",
                        finding=f"ROBUSTNESS (DeepSeek labels) reverse pooled: {what} → next myth 'be generous' [both orders]",
                        effect=r.coef_coop_on_generous_after, ci_low=r.ci_low, ci_high=r.ci_high, p=r.p, n_obs=r.n,
                        note="mc.reverse_models, run + round FE"))
    return out


def main():
    rows = []
    for ds in ("september", "frontier"):
        if (HERE / ds / "moral_uptake_by_stratum.csv").exists():
            rows += rows_for(ds)
    rows += robustness_rows("frontier")
    df = pd.DataFrame(rows)
    df["holm_p"] = np.nan
    prim = df["_kind"].isin(["wilcoxon", "regression", "wilcoxon_nosign"]) & df["p"].notna() & df["_special"].isna() \
        if "_special" in df else df["p"].notna()
    for _, idx in df[prim].groupby(["dataset", "setting", "population", "_order"]).groups.items():
        df.loc[idx, "holm_p"] = multipletests(df.loc[idx, "p"], method="holm")[1]
    df["status"] = [status(r) for r in df.to_dict("records")]
    # flag uptake rows whose future-myth placebo is at least 75% of the effect
    for i, r in df[df["finding"].str.contains("moral label uptake")].iterrows():
        key = r["finding"].split(", ", 1)[-1] if "R3-adjusted" in r["finding"] else r["finding"].replace("moral label uptake shown − unseen, ", "")
        what = key if "R3" not in r["finding"] else r["finding"].rsplit("), ", 1)[-1]
        plc = df[(df["dataset"] == r["dataset"]) & (df["population"] == r["population"]) & (df["family"] == r["family"])
                 & (df["finding"] == f"PLACEBO future myth − unseen, {what}")]
        if len(plc) and pd.notna(r["effect"]) and r["effect"] > 0 and plc["effect"].iloc[0] >= 0.75 * r["effect"]:
            df.at[i, "note"] = f"placebo ≈ effect (future myth {plc['effect'].iloc[0]:+.1f} pts): match comes from the shared game, not the text; " + r["note"]
    df.loc[df["_kind"] == "placebo", "note"] = df.loc[df["_kind"] == "placebo", "note"] + " (check or robustness, not in Holm)"
    df[COLS].to_csv(HERE / "scorecard_rows.csv", index=False)
    print(df.groupby(["dataset", "status"]).size())


if __name__ == "__main__":
    main()

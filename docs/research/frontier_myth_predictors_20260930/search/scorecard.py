#!/usr/bin/env python3
"""Scorecard rows for the search lens, September beside frontier.

September gain rows come from the committed September gain_table.csv / mde_calibration.csv;
September confirm and R4-amount rows come from the ported rerun in search/september/, which
reproduces the committed pooled numbers exactly and adds per-family strata and wild-cluster p's.
Frontier rows come from search/frontier/. Status rules:
  yes                    Holm (on CR1 p, within stratum) < 0.05 AND wild-cluster p < 0.05
                         (strata with < 10 runs: "yes" only if Holm on the wild p < 0.05)
  suggestive             CR1 p < 0.05 and wild p < 0.05, not Holm-significant
  p / holm_p columns     CR1 cluster-robust p and its Holm; the wild p is in the note
  underpowered           not significant and fewer than 8 runs (or flagged by the fit); for gain
                         rows, also when the planted-signal calibration cannot recover 0.05 R2
  no detectable effect   otherwise
  gain rows: yes if the bootstrap CI excludes 0 and the gain beats the same-width noise block.
Writes scorecard_rows.csv next to this script.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import HERE, SEPT_SEARCH

COLS = ["dataset", "setting", "population", "family", "finding", "effect", "ci_low", "ci_high", "p", "holm_p",
        "n_runs", "n_obs", "status", "note"]
POP = {  # (dataset, family, setting) -> population label
    ("frontier", "2-agent homogeneous"): "{f} dyads",
    ("frontier", "2-agent mixed"): {"Opus": "Opus+Sol and Opus+GeminiPro dyads", "Sol": "Opus+Sol and GeminiPro+Sol dyads",
                                    "GeminiPro": "Opus+GeminiPro and GeminiPro+Sol dyads"},
    ("frontier", "8-agent homogeneous"): "8 {f}",
    ("frontier", "8-agent mixed"): "2 GeminiPro + 3 Opus + 3 Sol",
}
OUTCOME = {"send": "round-r send", "return": "return share", "dsend": "change in send"}


def population(ds, fam, setting):
    if setting in ("all", None):
        return f"all {ds} runs" + (f" ({fam} agents)" if fam not in ("pooled", "all") else "")
    p = POP.get((ds, setting))
    if isinstance(p, dict):
        return p[fam]
    return p.format(f=fam) if p else setting


def status_of(p, holm, n_runs, flagged, p_wild=np.nan, holm_wild=np.nan):
    """yes needs both: Holm on the cluster-robust p (CR1) and a wild-cluster p below 0.05. CR1 alone
    is anti-conservative with 5-20 clusters; Holm on the wild p alone is too strict when only a few
    clusters carry the variation (September GPT R4: CR1 p 4e-11, wild p 0.003)."""
    if flagged and flagged != "fitted":
        return flagged
    wild_ok = np.isnan(p_wild) or p_wild < 0.05
    if n_runs < 10:  # 5-9 clusters: CR1 is unreliable, so only Holm on the wild p can give "yes"
        if pd.notna(holm_wild) and holm_wild < 0.05:
            return "yes"
    elif pd.notna(holm) and holm < 0.05 and wild_ok:
        return "yes"
    if pd.notna(p) and p < 0.05 and wild_ok:
        return "suggestive"
    if n_runs < 8:
        return "underpowered"
    return "no detectable effect"


def gain_rows(ds, g, mde):
    rows = []
    for (o, st), gg in g.groupby(["outcome", "stratum"]):
        fam = st.split(" ")[0]
        task = st.split(" ", 1)[1] if " " in st else None
        # gain strata split by family x task order, not by setting (2/8-agent, homogeneous/mixed pooled)
        setting = "all settings pooled, discovery split" + (f", {task}" if task else "")
        base = {"dataset": ds, "setting": setting, "population": population(ds, fam, None), "family": fam}
        if "status" in gg and (gg["status"] != "fitted").all():
            r = gg.iloc[0]
            rows.append(dict(base, finding=f"all myth features add held-out R2 for {OUTCOME[o]}", n_runs=r["n_runs"],
                             n_obs=r["n"], status=r["status"], note=f"outcome sd {r.get('y_sd', np.nan):.3f}"))
            continue
        for version, learner in (("raw", "ridge"), ("within-agent", "ridge"), ("raw", "hgb")):
            sel = gg[(gg["version"] == version) & (gg["learner"] == learner)]
            a = sel[sel["feature_set"] == "ALL myth features"]
            nz = sel[sel["feature_set"] == "NOISE placebo (same width)"]
            if a.empty:
                continue
            a, nzg = a.iloc[0], (nz["gain"].iloc[0] if len(nz) else np.nan)
            ok = a["gain_lo"] > 0 and (np.isnan(nzg) or a["gain"] > nzg)
            m = mde[(mde["stratum"] == fam) & (mde["outcome"] == o) & (mde["version"] == version)] if mde is not None else []
            mtxt, blind = "", a["n_runs"] < 8
            if learner == "ridge":
                if len(m):
                    det = m.groupby("target_gain")["detected"].mean()
                    hit = det[det >= 2 / 3]
                    mtxt = f"; planted-signal MDE {hit.index.min():g} R2" if len(hit) else \
                        f"; planted signal not recovered even at {det.index.max():g} R2"
                    blind = blind or not len(hit) or hit.index.min() > 0.05
                else:
                    mtxt = "; MDE not calibrated for this stratum"
            rows.append(dict(base, finding=f"all myth features add held-out R2 for {OUTCOME[o]} ({version}, {learner})",
                             effect=a["gain"], ci_low=a["gain_lo"], ci_high=a["gain_hi"], n_runs=a["n_runs"], n_obs=a["n"],
                             status="yes" if ok else ("underpowered" if blind else "no detectable effect"),
                             note=f"baseline R2 {a['r2_base']:.3f}; noise block gain {nzg:+.4f}{mtxt}"
                                  + ("; September feature bank includes the style group (headline also checked "
                                     "without it: identical gain)" if ds == "september" else "")))
    return rows


def confirm_rows(ds, c, exploratory=False):
    """Primary rows (R4, and R3 total effect in the never-played-before sample). Holm on the
    wild-cluster p within each stratum; the ported September run reproduces the committed pooled
    coefficients, CIs and CR1 p's exactly (the committed Holm used CR1 p's; same verdicts)."""
    rows = []
    prim = c[(c["rung"] == "R4") | ((c["sample"] == "never played before") & c["estimate"].str.startswith("total"))]
    for r in prim.itertuples():
        parts = r.stratum.split(" | ")
        fam, setting = parts[0], (parts[1] if len(parts) > 1 else "all")
        rung_set = "8-agent myth_game rounds>=2 (R3)" if r.rung == "R3" else "myth_game round 1 (R4)"
        if r.rung == "R4":
            finding = f"own round-1 myth {r.feature} (per SD) -> round-1 {'send' if r.outcome == 'send' else 'return'}"
        else:
            finding = f"shown myth {r.feature} (per SD, minus unseen placebo) -> {OUTCOME[r.outcome]}"
        if exploratory:
            finding = "[exploratory] " + finding
        if setting != "all":
            pop = population(ds, fam, setting)
        elif r.rung == "R4":
            pop = f"all {ds} myth_game runs" + ("" if fam == "pooled" else f" ({fam} agents)")
        else:
            pop = f"{ds} 8-agent myth_game runs" + ("" if fam == "pooled" else f" ({fam} agents)")
        note = (f"wild-cluster p {r.p_wild:.3f} (Holm on wild p {r.p_holm_wild:.3f})"
                if pd.notna(getattr(r, "p_wild", np.nan)) else "")
        if fam == "pooled" and isinstance(r.pooled_excludes, str) and r.pooled_excludes:
            note += f"; pooled excludes ceiling families {r.pooled_excludes}"
        if r.status == "fitted" and pd.notna(r.ci_low) and np.isclose(r.ci_low, r.ci_high):
            note += "; zero residual: every sender that states an amount sends exactly it (CI degenerate)"
        st = r.status if r.status != "fitted" else status_of(r.p, r.p_holm, r.n_runs, r.status, r.p_wild, r.p_holm_wild)
        rows.append({"dataset": ds, "setting": f"{setting}; {rung_set}", "population": pop, "family": fam,
                     "finding": finding, "effect": getattr(r, "coef", np.nan), "ci_low": getattr(r, "ci_low", np.nan),
                     "ci_high": getattr(r, "ci_high", np.nan), "p": getattr(r, "p", np.nan),
                     "holm_p": getattr(r, "p_holm", np.nan), "n_runs": r.n_runs, "n_obs": r.n, "status": st,
                     "note": (note + (f"; Holm family {int(r.n_holm_tests)} tests" if r.n_holm_tests else "")
                              + ("; exploratory candidate set (frontier screen), separate Holm" if exploratory else "")
                              ).strip("; ")})
    return rows


def amount_rows(ds, a):
    rows = []
    for r in a.itertuples():
        if pd.isna(getattr(r, "slope_per_dollar", np.nan)):
            rows.append({"dataset": ds, "setting": "all; myth_game round 1 (R4)", "population": f"all {ds} myth_game runs",
                         "family": r.family, "finding": "$ sent per $ stated in own round-1 myth",
                         "n_obs": r.n_investors, "status": "not estimable: ceiling" if r.family in ("Gemini", "GeminiPro")
                         else "no detectable effect", "note": f"exact match {r.exact_match_share:.2f} of {r.n_with_amount}"})
            continue
        pw = getattr(r, "slope_p_wild", np.nan)
        rows.append({"dataset": ds, "setting": "all; myth_game round 1 (R4)", "population": f"all {ds} myth_game runs",
                     "family": r.family, "finding": "$ sent per $ stated in own round-1 myth", "effect": r.slope_per_dollar,
                     "ci_low": r.slope_lo, "ci_high": r.slope_hi, "p": pw if pd.notna(pw) else r.slope_p,
                     "n_runs": getattr(r, "slope_n_runs", np.nan), "n_obs": r.n_with_amount,
                     "status": "yes" if (pw if pd.notna(pw) else r.slope_p) < 0.05 else "no detectable effect",
                     "note": f"exact match {r.exact_match_share:.2f}; stated amount {getattr(r, 'stated_mean', np.nan):.2f} "
                             f"(sd {getattr(r, 'stated_sd', np.nan):.2f}); {r.n_with_amount} of {r.n_investors} senders "
                             f"state an amount" + ("; p is wild-cluster" if pd.notna(pw) else "; p is CR1")})
    return rows


def post_checks(out):
    """Verdict adjustments from follow-up checks run after the first pass (documented in the note)."""
    fr, note = out["dataset"] == "frontier", out["note"].fillna("")
    # 1. Gain rows are discovery-split, uncorrected over ~40 strata x versions; the one positive
    #    family (Opus return) holds within each task order (opus_return_check.csv: own return rule
    #    +0.060 game_myth, +0.039 myth_game) but only between agents and not at R4.
    g = fr & out["finding"].str.startswith("all myth features") & out["finding"].str.contains("return share (raw", regex=False)
    pos = g & (out["status"] == "yes")
    out.loc[pos, "status"] = "suggestive"
    out.loc[pos, "note"] = note[pos].str.replace(r"; planted-signal MDE.*$", "", regex=True) + (
        "; discovery split, uncorrected over ~40 strata; carried by the own myth's stated return rule (Opus rule-only "
        "+0.068 R2, holds in game_myth +0.060 and myth_game +0.039 and with full cell dummies); within-agent +0.016 "
        "(CI -0.002 to 0.037); not confirmed at R4 (own round-1 return rule -> round-1 return +0.006, ns)")
    # 2. Multiplier words stand in for 'send all five' (Sol R4: -0.02, p 0.36 once stated amount and send rule
    #    are controlled).
    m = fr & out["finding"].str.contains("own round-1 myth lex_multiplier") & out["finding"].str.contains("send") \
        & out["status"].isin(["yes", "suggestive"])
    out.loc[m, "status"] = "suggestive"
    out.loc[m, "note"] = out.loc[m, "note"].fillna("") + ("; proxy for the stated amount: Sol coefficient -0.02 "
                                                         "(p 0.36) with stated amount + send rule controlled")
    # 3. Numerically zero coefficients = outcome constant within agent (dsend at the ceiling).
    z = out["effect"].abs() < 1e-10
    out.loc[z, "status"] = "not estimable: ceiling"
    out.loc[z, "note"] = out.loc[z, "note"].fillna("") + "; outcome constant within agent after fixed effects"
    # 4. Mixed 8-agent population: only 2 GeminiPro members, so a GeminiPro-authored shown myth read by the other
    #    GeminiPro has no same-family unseen placebo and drops out.
    mx = fr & out["population"].eq("2 GeminiPro + 3 Opus + 3 Sol") & out["finding"].str.startswith(("shown", "[exploratory] shown"))
    out.loc[mx, "note"] = out.loc[mx, "note"].fillna("") + ("; GeminiPro-authored shown myths read by the other "
                                                           "GeminiPro have no unseen placebo (not testable by design)")
    return out


def main():
    rows = []
    sg = pd.read_csv(SEPT_SEARCH / "gain_table.csv")
    smde = pd.read_csv(SEPT_SEARCH / "mde_calibration.csv").assign(stratum="pooled")
    rows += gain_rows("september", sg, smde)
    fg = pd.read_csv(HERE / "frontier/gain_table.csv")
    fm = HERE / "frontier/mde_calibration.csv"
    rows += gain_rows("frontier", fg, pd.read_csv(fm) if fm.exists() else None)
    rows += confirm_rows("september", pd.read_csv(HERE / "september/confirm_R3_R4.csv"))
    rows += confirm_rows("frontier", pd.read_csv(HERE / "frontier/confirm_R3_R4.csv"))
    rows += confirm_rows("frontier", pd.read_csv(HERE / "frontier/confirm_R3_R4_exploratory.csv"), exploratory=True)
    rows += amount_rows("september", pd.read_csv(HERE / "september/r4_amount_models.csv"))
    rows += amount_rows("frontier", pd.read_csv(HERE / "frontier/r4_amount_models.csv"))
    out = pd.DataFrame(rows).reindex(columns=COLS)
    out = post_checks(out)
    out.to_csv(HERE / "scorecard_rows.csv", index=False)
    print(out.groupby(["dataset", "status"]).size())


if __name__ == "__main__":
    main()

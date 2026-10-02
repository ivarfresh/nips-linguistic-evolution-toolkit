#!/usr/bin/env python3
"""Part 2: does consistency language predict cooperation LEVEL or STABILITY?

Decision pairing follows moral_carryover.decision_table: the agent's own myth written
before the decision (myth_game: same round; game_myth: previous round), the myth it was
shown before writing that myth, and the co-player's myth from the same round (placebo).

Outcomes per decision:
  coop       investor sent/5, trustee return proportion
  absdelta   |coop - own previous coop in the same role|
  zero       investor sent 0
  cut        investor sends less than last time, only after a letdown (last time as investor got
             back less than sent: return_proportion < 1/3, true amounts)
Rungs: R1 (cell FE + round FE), R2 (agent-within-run FE + round FE + own lagged coop),
R3 (8-agent myth_game, shown myth by an author who never played the reader; placebo = an
unseen same-family myth from the same round), R4 (myth_game round-1 myth -> round-1 move and
rounds 2-10 stability, per agent). Outputs: predict_*.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from fe import fe_ols, rows_from

OUT = Path(__file__).resolve().parent
DATA = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-spread-viz/data/analysis/linguistic_20260923")
FEATS = ["cons_judge", "cons_lex", "cons_emb"]
RNG = np.random.default_rng(0)


def load():
    m = pd.read_csv(OUT / "myth_features.csv")
    m.loc[~m["valid"], FEATS] = np.nan
    d = pd.read_csv(DATA / "decisions.csv").sort_values(["run_id", "agent", "round"]).reset_index(drop=True)
    d["setting"] = d["size"].astype(str) + "-agent " + np.where(d["mixed"], "mixed", "homogeneous")
    d["cell"] = d["composition"] + "|" + d["task_order"]
    d["run_agent"] = d["run_id"] + "|" + d["agent"]
    key = {(r, t, a): i for i, (r, t, a) in enumerate(zip(m.run_id, m["round"], m.agent))}
    fv = m[FEATS].to_numpy()
    fam = m["family"].to_numpy()
    d["r_before"] = np.where(d["task_order"] == "myth_game", d["round"], d["round"] - 1)
    own_i, shown_i, cop_i = [], [], []
    for g in d.itertuples(index=False):
        i = key.get((g.run_id, g.r_before, g.agent))
        own_i.append(i)
        if i is not None and isinstance(m.at[i, "exposed_author"], str):
            shown_i.append(key.get((g.run_id, int(m.at[i, "exposed_round"]), m.at[i, "exposed_author"])))
        else:
            shown_i.append(None)
        cop_i.append(key.get((g.run_id, g.r_before, g.partner)))
    for name, idx in [("own", own_i), ("shown", shown_i), ("cop", cop_i)]:
        arr = np.array([fv[i] if i is not None else [np.nan] * len(FEATS) for i in idx])
        for j, f in enumerate(FEATS):
            d[f"{name}_{f}"] = arr[:, j]
    d["shown_idx"] = shown_i
    d["shown_family"] = [fam[i] if i is not None else None for i in shown_i]
    d["shown_author"] = [m.at[i, "agent"] if i is not None else None for i in shown_i]
    # outcomes
    grp = d.groupby(["run_id", "agent", "role"])
    d["coop_lag"] = grp["coop"].shift(1)
    d["absdelta"] = (d["coop"] - d["coop_lag"]).abs()
    inv = d["role"] == "investor"
    d["zero"] = np.where(inv, (d["sent"] == 0).astype(float), np.nan)
    prev_rp = grp["return_proportion"].shift(1)
    prev_sent = grp["sent"].shift(1)
    letdown = inv & (prev_rp < 1 / 3) & (prev_sent > 0)
    d["cut_after_letdown"] = np.where(letdown, (d["sent"] < prev_sent).astype(float), np.nan)
    d["prev_zero"] = np.where(inv, (prev_sent == 0).astype(float), np.nan)
    # what last round's partner did to this agent (investor: share returned; trustee: share sent)
    ga = d.groupby(["run_id", "agent"])
    lr, lrp, ls = ga["role"].shift(1), ga["return_proportion"].shift(1), ga["sent"].shift(1)
    d["partner_prev_move"] = np.where(lr == "investor", lrp, np.where(lr == "trustee", ls / 5, np.nan))
    d.loc[~inv | prev_sent.isna(), "prev_zero"] = np.nan
    return m, d


def standardise(d):
    """cons_emb in sd units (pooled) so coefficients are comparable; binary features stay 0/1."""
    for p in ["own", "shown", "cop", "null"]:
        c = f"{p}_cons_emb"
        if c in d:
            d[c + "_z"] = d[c] / d["own_cons_emb"].std()
    return d


def r1_r2(d):
    rows = []
    outcomes = [("coop", "investor"), ("coop", "trustee"), ("absdelta", "investor"), ("absdelta", "trustee"),
                ("zero", "investor"), ("cut_after_letdown", "investor")]
    fams = ["Sonnet", "GPT", "Gemini", "all"]
    for fam in fams:
        g0 = d if fam == "all" else d[d["family"] == fam]
        for y, role in outcomes:
            g = g0[g0["role"] == role]
            if fam == "Gemini" and role == "investor":
                continue  # Gemini sends 5 in 99.7% of decisions: nothing to predict
            for f in ["cons_judge", "cons_lex", "cons_emb_z"]:
                x = f"own_{f}"
                # R1: within composition x task order
                try:
                    res = fe_ols(g, y, [x], absorb="cell", dummies=["round"])
                    rows += rows_from(res, [x], rung="R1", family=fam, role=role, outcome=y, feature=f)
                except Exception:
                    pass
                # R2: agent FE + round FE + own lagged move (not for 'cut', which conditions on the lag)
                xs = [x] + ([] if y == "cut_after_letdown" else ["coop_lag"])
                try:
                    res = fe_ols(g, y, xs, absorb="run_agent", dummies=["round"])
                    rows += rows_from(res, [x], rung="R2", family=fam, role=role, outcome=y, feature=f)
                except Exception:
                    pass
    return pd.DataFrame(rows)


def lockin(d):
    """'Keep doing the same' vs reliable reciprocity.
    (a) persistence: does consistency raise the weight on the own last move (coop ~ lag x cons)?
    (b) GPT zero-lock: after sending 0, does a consistency myth make a second 0 more likely?
    (c) does consistency lower |change| equally for low and high previous cooperators?"""
    rows = []
    d = d.copy()
    for f in ["cons_judge", "cons_lex", "cons_emb_z"]:
        x = f"own_{f}"
        d["lagXcons"] = d["coop_lag"] * d[x]
        for fam in ["Sonnet", "GPT", "all"]:
            for role in ["investor", "trustee"]:
                g = d[(d["role"] == role) & ((d["family"] == fam) | (fam == "all"))]
                res = fe_ols(g, "coop", ["coop_lag", x, "lagXcons"], absorb="cell", dummies=["round", "family"])
                rows += rows_from(res, ["lagXcons"], test="persistence (coop ~ lag*cons)", family=fam, role=role,
                                  feature=f, rung="R1")
                # (c) split by previous level
                med = g["coop_lag"].median()
                for lab, sub in [("prev below median", g[g["coop_lag"] < med]), ("prev at/above median", g[g["coop_lag"] >= med])]:
                    try:
                        res = fe_ols(sub, "absdelta", [x, "coop_lag"], absorb="run_agent", dummies=["round"])
                        rows += rows_from(res, [x], test=f"|change| ~ cons, {lab}", family=fam, role=role,
                                          feature=f, rung="R2")
                    except Exception:
                        pass
        g = d[(d["family"] == "GPT") & (d["prev_zero"] == 1)]
        res = fe_ols(g, "zero", [x], absorb="cell", dummies=["round"])
        rows += rows_from(res, [x], test="GPT: send 0 again after sending 0", family="GPT", role="investor",
                          feature=f, rung="R1")
    return pd.DataFrame(rows)


def r3(d, m):
    """8-agent myth_game: shown myth's consistency -> reader's decision in the same round.
    Keep only shown authors who never played the reader before or in this round. Placebo:
    a random unseen same-family myth from the same run and round (not the reader, author or partner)."""
    g = d[(d["size"] == 8) & (d["task_order"] == "myth_game") & d["shown_idx"].notna()].copy()
    # The shown myth is always by last round's partner (checked: 6,480/6,480 8-agent exposures), so
    # "never played" is impossible. Closest clean version: shown author is not this round's partner,
    # and we control what that author did to the reader last round (partner_prev_move).
    g = g[g["shown_author"] != g["partner"]]
    # placebo: unseen same-family myth, same run & round as the shown myth
    mm = m[m["valid"]]
    pools = {k: v.index.to_list() for k, v in mm.groupby(["run_id", "round", "family"])}
    fv = m[FEATS]
    nulls = []
    for x in g.itertuples(index=False):
        sr = int(m.at[int(x.shown_idx), "round"])
        c = [j for j in pools.get((x.run_id, sr, x.shown_family), [])
             if m.at[j, "agent"] not in (x.agent, x.shown_author, x.partner)]
        nulls.append(fv.loc[c].mean().to_numpy() if c else [np.nan] * len(FEATS))
    nulls = np.array(nulls)
    for j, f in enumerate(FEATS):
        g[f"null_{f}"] = nulls[:, j]
    g = standardise(g.assign(own_cons_emb=g["own_cons_emb"]))
    g["null_cons_emb_z"] = g["null_cons_emb"] / d["own_cons_emb"].std()
    g["shown_cons_emb_z"] = g["shown_cons_emb"] / d["own_cons_emb"].std()
    g["own_cons_emb_z"] = g["own_cons_emb"] / d["own_cons_emb"].std()
    rows = []
    for fam in ["Sonnet", "GPT", "Gemini", "all"]:
        gf = g if fam == "all" else g[g["family"] == fam]
        for y, role in [("coop", "investor"), ("coop", "trustee"), ("absdelta", "investor"), ("absdelta", "trustee"), ("zero", "investor")]:
            if fam == "Gemini" and role == "investor":
                continue
            s = gf[gf["role"] == role]
            for f in ["cons_judge", "cons_lex", "cons_emb_z"]:
                xs = [f"shown_{f}", f"null_{f}", f"own_{f}", "coop_lag", "partner_prev_move"]
                try:
                    res = fe_ols(s, y, xs, absorb="run_agent", dummies=["round"])
                except Exception:
                    continue
                rows += rows_from(res, [f"shown_{f}", f"null_{f}"], rung="R3", family=fam, role=role, outcome=y, feature=f)
    return pd.DataFrame(rows), len(g)


def r4(d):
    """myth_game round-1 myth (before any play) -> round-1 move and the agent's later stability."""
    one = d[(d["task_order"] == "myth_game") & (d["round"] == 1)][["run_id", "agent", "run_agent", "family", "cell",
                                                                    "role", "coop", "own_cons_judge", "own_cons_lex", "own_cons_emb"]]
    later = d[(d["task_order"] == "myth_game") & (d["round"] >= 2)]
    agg = later.groupby("run_agent").agg(mean_absdelta=("absdelta", "mean"), zero_share=("zero", "mean"))
    agg["sd_sent"] = later[later["role"] == "investor"].groupby("run_agent")["sent"].std()
    agg["sd_retprop"] = later[later["role"] == "trustee"].groupby("run_agent")["return_proportion"].std()
    agg["mean_coop"] = later.groupby("run_agent")["coop"].mean()
    a = one.merge(agg.reset_index(), on="run_agent")
    a["own_cons_emb_z"] = a["own_cons_emb"] / d["own_cons_emb"].std()
    rows = []
    for fam in ["Sonnet", "GPT", "Gemini", "all"]:
        s = a if fam == "all" else a[a["family"] == fam]
        for y in ["coop", "mean_absdelta", "sd_sent", "sd_retprop", "zero_share", "mean_coop"]:
            for f in ["cons_judge", "cons_lex", "cons_emb_z"]:
                x = f"own_{f}"
                if s[x].nunique() < 2:
                    continue
                dummies = ["role"] + (["family"] if fam == "all" else [])
                try:
                    res = fe_ols(s, y, [x], absorb="cell", dummies=dummies)
                except Exception:
                    continue
                rows += rows_from(res, [x], rung="R4", family=fam, outcome=y + (" (round 1)" if y == "coop" else " (rounds 2-10)"),
                                  feature=f, role="both")
    return pd.DataFrame(rows), a


def main():
    m, d = load()
    d = standardise(d)
    out = {}
    out["r12"] = r1_r2(d)
    out["lockin"] = lockin(d)
    out["r3"], n3 = r3(d, m)
    out["r4"], a4 = r4(d)
    for k, t in out.items():
        t = t.copy()
        t["p_holm_within_table"] = multipletests(t["p"].fillna(1), method="holm")[1]
        t.round(4).to_csv(OUT / f"predict_{k}.csv", index=False)
        out[k] = t
    a4.to_csv(OUT / "predict_r4_agents.csv", index=False)
    d.drop(columns=["path"]).to_csv(OUT / "decision_table.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    cols = ["rung", "family", "role", "outcome", "feature", "term", "coef", "ci_low", "ci_high", "p", "p_holm_within_table", "n", "n_runs"]
    print(out["r12"][cols].round(3).to_string())
    print(out["lockin"].round(3).to_string())
    print("R3 decisions:", n3)
    print(out["r3"][cols].round(3).to_string())
    print(out["r4"][cols].round(3).to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyse the edit-and-replay probe (analyses/myth_replay_probe.py --stage main).

Intent to treat: every replayed edit counts, whatever the judge read. Per arm x family x
edit mode, OLS of the replayed send on the edited amount ($1/$2/$3/$5) with a fixed
effect per context (so contexts are compared only with themselves), SE clustered by run,
t(G-1) inference. Primary test per arm: rule edits, Sonnet and GPT pooled (Gemini sends
$5 almost always and is reported separately); Holm over the three arms. Decision rules
were fixed before the run (docs/research/myth_opening_plan_20260930/README.md and the
2026-10-01 design):

  A, C: confirmed if slope >= $0.30 per $1 with the CI above 0; overturned if the CI includes 0.
  B:    "read myths don't move play" confirmed if the CI upper bound < $0.20; overturned if
        B's slope is about as large as A's (A's point estimate inside B's CI).

Secondary: Sonnet rule vs natural edits (amount x mode interaction on contexts with both),
judge-pass-only slopes, the unedited control against the logged original send.

  python3 analyses/myth_replay_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import configure_matplotlib  # noqa: E402

RAW = ROOT / "data/analysis/myth_replay_probe_20261001/main.jsonl"
OUT = ROOT / "docs/research/myth_replay_probe_20261001"
FAM_COLORS = {"Sonnet": "#7570b3", "GPT": "#d95f02", "Gemini": "#1b9e77"}
ARM_NAME = {"A": "A: own first myth (round 1)", "B": "B: partner's myth read (8-agent)",
            "C": "C: own later myth (game first)"}


def load() -> pd.DataFrame:
    rows = [json.loads(x) for x in RAW.read_text().splitlines() if x.strip()]
    d = pd.DataFrame(rows).drop_duplicates("key", keep="last")
    return d


def slope(sub: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf
    sub = sub.dropna(subset=["send", "edited_amount"])
    rec = {"n_obs": len(sub), "n_contexts": sub["cid"].nunique(), "n_runs": sub["run"].nunique()}
    if sub["cid"].nunique() < 5 or sub["send"].nunique() < 2:
        return {**rec, "status": "not estimable"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols("send ~ edited_amount + C(cid)", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(sub["run"])[0]}, use_t=True)
    lo, hi = fit.conf_int().loc["edited_amount"]
    return {**rec, "slope": fit.params["edited_amount"], "ci_low": lo, "ci_high": hi,
            "p": fit.pvalues["edited_amount"]}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    rep = d[(d["made"] == True) & d["error"].isna()]  # noqa: E712
    edited = rep[rep["mode"].isin(["rule", "natural"])]
    print(f"rows {len(d)} | replayed {len(rep)} | errors {int(d['error'].notna().sum())} | "
          f"parse failures {int((rep['send'].isna()).sum())} | refusals {int((rep['finish_reason'] == 'refusal').sum())}")

    # per arm x family x mode
    rows = []
    for (arm, fam, mode), g in edited.groupby(["arm", "family", "mode"]):
        rows.append({"arm": arm, "family": fam, "mode": mode, **slope(g)})
    for arm, g in edited[(edited["mode"] == "rule") & edited["family"].isin(["Sonnet", "GPT"])].groupby("arm"):
        rows.append({"arm": arm, "family": "Sonnet+GPT (primary)", "mode": "rule", **slope(g)})
    for (arm, fam), g in edited[(edited["mode"] == "rule") & (edited["check"] == "reads_intended")].groupby(["arm", "family"]):
        rows.append({"arm": arm, "family": fam, "mode": "rule, judge-pass only", **slope(g)})
    res = pd.DataFrame(rows)
    prim = res["family"] == "Sonnet+GPT (primary)"
    res.loc[prim, "p_holm"] = holm(res.loc[prim, "p"].tolist())
    res.to_csv(OUT / "slopes.csv", index=False)
    print("\nslope of send on edited amount ($ per $1), context FE, run-clustered t CI:")
    print(res.round(3).to_string(index=False))

    # decision rules on the primary rows
    p = res[prim].set_index("arm")
    verdict = {}
    for arm in "AC":
        r = p.loc[arm]
        verdict[arm] = ("confirmed" if r["slope"] >= 0.30 and r["ci_low"] > 0 else
                        "overturned" if r["ci_low"] <= 0 <= r["ci_high"] else "inconclusive")
    rb, ra = p.loc["B"], p.loc["A"]
    verdict["B"] = ("no effect confirmed" if rb["ci_high"] < 0.20 else
                    "overturned (B about as large as A)" if rb["ci_low"] <= ra["slope"] <= rb["ci_high"] else
                    "inconclusive")
    pd.Series(verdict, name="verdict").to_csv(OUT / "verdicts.csv")
    print("\nverdicts:", verdict)

    # rule vs natural (Sonnet, contexts with both)
    import statsmodels.formula.api as smf
    son = edited[edited["family"] == "Sonnet"]
    both = son.groupby("cid")["mode"].nunique()
    sb = son[son["cid"].isin(both[both == 2].index)].assign(natural=lambda x: (x["mode"] == "natural").astype(float))
    inter = []
    for arm, g in sb.groupby("arm"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols("send ~ edited_amount * natural + C(cid)", data=g.dropna(subset=["send"])).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(g.dropna(subset=["send"])["run"])[0]}, use_t=True)
        lo, hi = fit.conf_int().loc["edited_amount:natural"]
        inter.append({"arm": arm, "rule_slope": fit.params["edited_amount"],
                      "natural_minus_rule": fit.params["edited_amount:natural"], "ci_low": lo, "ci_high": hi,
                      "p": fit.pvalues["edited_amount:natural"], "n_contexts": g["cid"].nunique()})
    inter = pd.DataFrame(inter)
    inter.to_csv(OUT / "rule_vs_natural.csv", index=False)
    print("\nSonnet: natural-edit slope minus rule-edit slope:\n" + inter.round(3).to_string(index=False))

    # manipulation check, means, control fidelity
    mc = edited.assign(ok=edited["check"] == "reads_intended").groupby(["arm", "family", "mode"])["ok"].mean()
    mc.rename("judge_reads_intended").to_csv(OUT / "manipulation_check.csv")
    means = edited.groupby(["arm", "family", "mode", "edited_amount"])["send"].agg(["mean", "std", "count"]).reset_index()
    means.to_csv(OUT / "mean_send_by_amount.csv", index=False)
    ctrl = rep[rep["mode"] == "orig"].groupby(["arm", "family"]).agg(
        replay_mean=("send", "mean"), original_mean=("original_send", "mean"), n=("send", "size")).reset_index()
    ctrl.to_csv(OUT / "unedited_control.csv", index=False)
    print("\nunedited control vs logged original:\n" + ctrl.round(2).to_string(index=False))
    print("\nmanipulation check:\n" + mc.round(2).to_string())

    # figure
    import matplotlib.pyplot as plt
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, arm in zip(axes, "ABC"):
        for fam, col in FAM_COLORS.items():
            for mode, ls in (("rule", "-"), ("natural", "--")):
                m = means[(means["arm"] == arm) & (means["family"] == fam) & (means["mode"] == mode)]
                if m.empty:
                    continue
                ax.plot(m["edited_amount"], m["mean"], ls, marker="o", ms=4, color=col,
                        label=f"{fam} ({mode})" if arm == "A" else None)
            c = ctrl[(ctrl["arm"] == arm) & (ctrl["family"] == fam)]
            if len(c):
                ax.axhline(c["replay_mean"].iloc[0], color=col, lw=0.8, alpha=0.4)
        ax.plot([1, 5], [1, 5], color="#999999", lw=0.8, ls=":")
        ax.set_title(ARM_NAME[arm], fontsize=10)
        ax.set_xticks([1, 2, 3, 5])
        ax.set_xlabel("amount stated in the edited myth ($)")
        ax.set_ylim(0, 5.2)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean replayed send ($)")
    fig.legend(loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Does the amount a myth states cause the next send? Replayed September decisions "
                 "(faint lines: unedited control; dotted: send = stated amount)", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(OUT / "replay_send_by_amount.png", dpi=200)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

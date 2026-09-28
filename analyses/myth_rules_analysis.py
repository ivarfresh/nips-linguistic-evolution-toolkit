#!/usr/bin/env python3
"""Which play rules the myths prescribe, how they change, and whether agents follow them.

Reads the rule extraction from analyses/myth_rule_judge.py and the decision
tables behind Figures 7 and 8. Four parts:

1. Rules by family and round: shares of each send rule, return rule and
   "what to do when let down" rule, plus the prescribed send in dollars.
2. Prescribed vs actual sending per round, against the game-only runs.
3. Does an agent send what its own myth prescribes? Tested only where past
   play cannot explain the myth:
     T1  myth->game, round 1: the myth is written before any game.
     T2  game->myth, round 2: the myth follows one round, no partner myth seen
         yet; controls for what happened to the agent in round 1.
     T3  rounds 2-10, each agent compared with itself (agent + round fixed
         effects, own previous send as control). Myths mostly describe recent
         play, so a null is expected here; the reverse direction is reported too.
4. Transplant donors: the rule extracted from each injected donor text against
   what the hosts sent (the text was set by the experimenter, so this is the
   causal check).

Prescribed send in dollars = the named amount when the myth gives one, else the
midpoint of its send_rule band (all 5, most 4.25, moderate 2.75, little 1.25,
none 0); "unspecified" is missing.

  python3 analyses/myth_rules_analysis.py [--model z-ai/glm-5.2]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses._shared import configure_matplotlib  # noqa: E402

DATA = ROOT / "data/analysis/linguistic_20260923"
OUT = ROOT / "docs/figures/myth_rules_20260928"
FAMILY_COLORS = {"Sonnet": "#7570b3", "GPT": "#d95f02", "Gemini": "#1b9e77"}
FAMILIES = ["Sonnet", "GPT", "Gemini"]
MIDPOINT = {"all": 5.0, "most": 4.25, "moderate": 2.75, "little": 1.25, "none": 0.0}
ORDERS = {
    "send_rule": ["all", "most", "moderate", "little", "none", "unspecified"],
    "return_rule": ["more_than_half", "half", "match_partner", "at_least_sent", "little", "unspecified"],
    "after_letdown": ["keep_trusting", "reduce", "withdraw", "unspecified"],
}
COLORS = {  # generous -> stingy runs dark green -> red, unspecified grey
    "all": "#1b7837", "most": "#7fbf7b", "moderate": "#dfc27d", "little": "#d6604d", "none": "#b2182b",
    "more_than_half": "#1b7837", "half": "#7fbf7b", "match_partner": "#80cdc1", "at_least_sent": "#dfc27d",
    "keep_trusting": "#1b7837", "reduce": "#dfc27d", "withdraw": "#b2182b", "unspecified": "#d9d9d9",
}
TITLES = {"send_rule": "Send rule", "return_rule": "Return rule (share of the tripled amount)",
          "after_letdown": "After being let down"}


def setting(size, mixed) -> str:
    return f"{size}-agent {'mixed' if mixed else 'homogeneous'}"


def load_rules(model: str) -> pd.DataFrame:
    tag = model.replace("/", "__")
    r = pd.read_csv(DATA / f"myth_rules_september_{tag}.csv")
    r = r[r["status"] == "ok"].copy()
    r["prescribed"] = r["send_amount"].where(r["send_amount"].notna(), r["send_rule"].map(MIDPOINT))
    r["setting"] = [setting(s, m) for s, m in zip(r["size"], r["mixed"])]
    return r


def load_sends() -> pd.DataFrame:
    """Every send in every validated September run, game-only included."""
    d2 = pd.read_csv(ROOT / "docs/figures/mixed_model_dyads_20260917/decisions.csv")
    d8 = pd.read_csv(ROOT / "docs/figures/mixed_model_populations_20260918/games.csv")
    cols = ["path", "composition", "mixed", "task_order", "round", "sender_family", "sent"]
    return pd.concat([d2[cols].assign(size=2), d8[cols].assign(size=8)], ignore_index=True)


# ---------------------------------------------------------------- part 1
def rule_shares(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field, levels in ORDERS.items():
        g = r.groupby(["setting", "family", "task_order", "round"])[field].value_counts(normalize=True)
        g = g.rename("share").reset_index().rename(columns={field: "level"})
        rows.append(g.assign(field=field))
    out = pd.concat(rows)
    out["share"] = out["share"].round(4)
    return out


def plot_rule_shares(r: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    h = r[~r["mixed"]]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    for i, field in enumerate(ORDERS):
        for j, fam in enumerate(FAMILIES):
            ax = axes[i, j]
            share = pd.crosstab(h.loc[h.family == fam, "round"], h.loc[h.family == fam, field], normalize="index")
            bottom = np.zeros(len(share))
            for lev in ORDERS[field]:
                if lev in share:
                    ax.bar(share.index, share[lev], bottom=bottom, color=COLORS[lev], width=0.85,
                           edgecolor="white", linewidth=1)
                    bottom += share[lev].to_numpy()
            if i == 0:
                ax.set_title(fam)
            if j == 0:
                ax.set_ylabel(f"{TITLES[field]}\nshare of myths")
            if i == 2:
                ax.set_xlabel("round")
        axes[i, 2].legend(handles=[Patch(color=COLORS[l], label=l.replace("_", " ")) for l in ORDERS[field]],
                          loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
    axes[0, 0].set_xticks(range(1, 11))
    fig.suptitle("Rules the myths prescribe, homogeneous runs (2- and 8-agent, both myth orders)", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "rules_by_family_over_rounds.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- part 2
def prescribed_vs_actual(r: pd.DataFrame, sends: pd.DataFrame) -> pd.DataFrame:
    h = sends[~sends["mixed"]].copy()
    h["family"] = h["sender_family"]
    per_run = h.groupby(["size", "family", "task_order", "path", "round"])["sent"].mean().reset_index()
    actual = per_run.groupby(["size", "family", "task_order", "round"])["sent"].agg(["mean", "std"])
    actual.columns = ["sent_mean", "sent_sd"]
    rh = r[~r["mixed"]].copy()
    # line each send up with the myth written before it: in game->myth that is the previous round's myth
    rh["round"] = np.where(rh["task_order"] == "game_myth", rh["round"] + 1, rh["round"])
    rh = rh[rh["round"] <= 10]
    presc = rh.groupby(["size", "family", "task_order", "round"]).agg(
        prescribed_mean=("prescribed", "mean"), prescribed_specified=("prescribed", lambda s: s.notna().mean()))
    return actual.join(presc, how="left").reset_index().round(3)


def plot_prescribed_vs_actual(t: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    for i, size in enumerate((2, 8)):
        for j, fam in enumerate(FAMILIES):
            ax = axes[i, j]
            c = FAMILY_COLORS[fam]
            s = t[(t["size"] == size) & (t["family"] == fam)]
            g = s[s.task_order == "game"]
            ax.plot(g["round"], g["sent_mean"], color="#888888", ls=":", lw=2, label="game only: sent")
            for order, alpha in (("myth_game", 1.0), ("game_myth", 0.45)):
                o = s[s.task_order == order]
                name = order.replace("_", "→")
                ax.plot(o["round"], o["sent_mean"], color=c, alpha=alpha, lw=2, marker="o", ms=3,
                        label=f"{name}: sent")
                ax.plot(o["round"], o["prescribed_mean"], color=c, alpha=alpha, lw=2, ls="--",
                        label=f"{name}: myth before this game prescribes")
            ax.set_ylim(-0.2, 5.3)
            ax.set_title(f"{fam}, {size}-agent homogeneous")
            if j == 0:
                ax.set_ylabel("send ($ of 5)")
            if i == 1:
                ax.set_xlabel("round")
    axes[0, 0].set_xticks(range(1, 11))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("What the myths prescribe and what the agents send (mean over runs)", y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.savefig(OUT / "prescribed_vs_actual_send.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- part 3
def decisions(r: pd.DataFrame) -> pd.DataFrame:
    dec = pd.read_csv(DATA / "decisions.csv")
    key = r.set_index(["run_id", "round", "agent"])["prescribed"]
    d = dec.copy()
    # the myth an agent had written before this decision
    d["myth_round"] = np.where(d["task_order"] == "myth_game", d["round"], d["round"] - 1)
    d["prescribed_before"] = key.reindex(pd.MultiIndex.from_arrays([d.run_id, d.myth_round, d.agent])).to_numpy()
    # the myth written straight after this decision (reverse-direction check)
    after = np.where(d["task_order"] == "myth_game", d["round"] + 1, d["round"])
    d["prescribed_after"] = key.reindex(pd.MultiIndex.from_arrays([d.run_id, after, d.agent])).to_numpy()
    d["setting"] = [setting(s, m) for s, m in zip(d["size"], d["mixed"])]
    return d


def ols(formula: str, data: pd.DataFrame, term: str) -> dict:
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols(formula, data=data).fit(cov_type="cluster",
                                              cov_kwds={"groups": pd.factorize(data["run_id"])[0]})
    ci = fit.conf_int().loc[term]
    return {"coef": fit.params[term], "ci_low": ci[0], "ci_high": ci[1], "p": fit.pvalues[term],
            "n_decisions": int(fit.nobs), "n_runs": data["run_id"].nunique()}


def clean_tests(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, by_rule = [], []
    inv = d[d["role"] == "investor"]

    # T1: myth->game round 1
    t1 = inv[(inv.task_order == "myth_game") & (inv["round"] == 1)].dropna(subset=["prescribed_before"])
    for fam, g in [("all", t1), *t1.groupby("family")]:
        f = "sent ~ prescribed_before" + (" + C(family)" if fam == "all" else "") + " + C(size) + mixed"
        if g["prescribed_before"].nunique() > 1 and len(g) >= 10:
            rho, p_rho = stats.spearmanr(g["prescribed_before"], g["sent"])
            rows.append({"test": "T1 myth→game round 1", "family": fam, "spearman": rho, "spearman_p": p_rho,
                         **ols(f, g, "prescribed_before")})
    # T2: game->myth round 2, controlling for round-1 experience
    r1 = d[(d.task_order == "game_myth") & (d["round"] == 1)]
    got = r1[r1.role == "trustee"].set_index(["run_id", "agent"])["sent"].rename("sent_to_me_r1")
    gave = r1[r1.role == "investor"].set_index(["run_id", "agent"])["sent"].rename("my_send_r1")
    t2 = inv[(inv.task_order == "game_myth") & (inv["round"] == 2)].dropna(subset=["prescribed_before"])
    t2 = t2.join(got, on=["run_id", "agent"]).join(gave, on=["run_id", "agent"])
    t2["was_sender_r1"] = t2["my_send_r1"].notna().astype(int)
    t2[["sent_to_me_r1", "my_send_r1"]] = t2[["sent_to_me_r1", "my_send_r1"]].fillna(0)
    for fam, g in [("all", t2), *t2.groupby("family")]:
        f = ("sent ~ prescribed_before + sent_to_me_r1 + my_send_r1 + was_sender_r1"
             + (" + C(family)" if fam == "all" else "") + " + C(size) + mixed")
        if g["prescribed_before"].nunique() > 1 and len(g) >= 10:
            rho, p_rho = stats.spearmanr(g["prescribed_before"], g["sent"])
            rows.append({"test": "T2 game→myth round 2", "family": fam, "spearman": rho, "spearman_p": p_rho,
                         **ols(f, g, "prescribed_before")})
    # T3: rounds 2-10 within agent, forward and reverse
    w = inv[inv["round"] >= 2].copy()
    w["run_agent"] = w["run_id"] + "|" + w["agent"]
    w = w.sort_values(["run_agent", "round"])
    w["lag_send"] = w.groupby("run_agent")["sent"].shift(1)
    w["lag_prescribed"] = w.groupby("run_agent")["prescribed_before"].shift(1)
    fw = w.dropna(subset=["prescribed_before", "lag_send"])
    rows.append({"test": "T3 rounds 2-10, own myth → next send (agent FE)", "family": "all",
                 **ols("sent ~ prescribed_before + lag_send + C(run_agent) + C(round)", fw, "prescribed_before")})
    rv = w.dropna(subset=["prescribed_after", "prescribed_before"])
    rows.append({"test": "T3 reverse: send → next myth's prescription (agent FE)", "family": "all",
                 **ols("prescribed_after ~ sent + prescribed_before + C(run_agent) + C(round)", rv, "sent")})

    for name, t in (("T1 myth→game round 1", t1), ("T2 game→myth round 2", t2)):
        g = t.groupby(["family", "prescribed_before"])["sent"].agg(["mean", "std", "size"]).reset_index()
        by_rule.append(g.assign(test=name))
    return pd.DataFrame(rows).round(4), pd.concat(by_rule).round(3)


def letdown_response(d: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    """Next send after being sent $0, split by the after_letdown rule of the agent's latest myth.
    Descriptive: the myth was written after the agent's own history, so this is not causal."""
    lab = r.set_index(["run_id", "round", "agent"])["after_letdown"]
    rows = []
    for (run, agent), g in d.sort_values("round").groupby(["run_id", "agent"]):
        last_got = None
        for x in g.itertuples(index=False):
            if x.role == "trustee":
                last_got = x.sent
            elif last_got is not None and last_got == 0:
                rows.append({"setting": x.setting, "family": x.family, "run_id": run,
                             "rule": lab.get((run, x.myth_round, agent)), "sent": x.sent})
    t = pd.DataFrame(rows).dropna(subset=["rule"])
    out = t.groupby(["family", "rule"]).agg(n=("sent", "size"), next_send=("sent", "mean"),
                                            zero_again=("sent", lambda s: (s == 0).mean()),
                                            runs=("run_id", "nunique"))
    return out.reset_index().round(3)


# ---------------------------------------------------------------- part 4
def donors(model: str) -> pd.DataFrame:
    tag = model.replace("/", "__")
    dn = pd.read_csv(DATA / f"myth_rules_donors_{tag}.csv")
    dn["prescribed"] = dn["send_amount"].where(dn["send_amount"].notna(), dn["send_rule"].map(MIDPOINT))
    host_all, host_r1 = [], []
    for f in dn["final"]:
        run = json.loads((ROOT / f).read_text())
        s = pd.DataFrame([{"round": e["round"], "sent": float(x["sent"])}
                          for e in run["conversation_history"] for x in e.get("dyads") or []])
        host_all.append(s["sent"].mean())
        host_r1.append(s.loc[s["round"] == 1, "sent"].mean())
    dn["host_send_mean"], dn["host_send_r1"] = host_all, host_r1
    return dn


def plot_donors(dn: pd.DataFrame, baselines: dict) -> None:
    import matplotlib.pyplot as plt
    kinds = {"s_start": ("early Sonnet", "#7570b3"), "s_end_plus": ("late Sonnet, high coop", "#3f007d"),
             "s_end_minus": ("late, low coop", "#b2182b"), "s_end_plus_gemini": ("late Gemini", "#1b9e77"),
             "s_end_plus_gpt": ("late GPT", "#d95f02"), "s_filler": ("Wikipedia filler", "#999999")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, size in zip(axes, (8, 2)):
        s = dn[dn["size"] == size]
        for k, (lab, col) in kinds.items():
            x = s[s.seed_type == k]
            xs = x["prescribed"].fillna(-0.6) + np.random.default_rng(0).uniform(-0.12, 0.12, len(x))
            ax.scatter(xs, x["host_send_mean"], color=col, s=42, label=lab, edgecolor="white", linewidth=1)
        ax.axhline(baselines[size], color="#666666", ls=":", lw=1.5, label="no text (baseline)")
        ax.set_xticks([-0.6, 0, 1.25, 2.75, 4.25, 5])
        ax.set_xticklabels(["none\nstated", "0", "little", "moderate", "most", "all 5"], fontsize=8)
        ax.set_xlabel("send the donor myth prescribes")
        ax.set_title(f"{size}-agent transplant rerun (Sonnet hosts)")
    axes[0].set_ylabel("host mean send, rounds 1–10 ($)")
    axes[1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(OUT / "transplant_prescribed_vs_host_send.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="z-ai/glm-5.2")
    args = ap.parse_args()
    configure_matplotlib()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    r = load_rules(args.model)
    sends = load_sends()
    rule_shares(r).to_csv(OUT / "rule_shares_by_round.csv", index=False)
    plot_rule_shares(r)

    pva = prescribed_vs_actual(r, sends)
    pva.to_csv(OUT / "prescribed_vs_actual_send.csv", index=False)
    plot_prescribed_vs_actual(pva)

    d = decisions(r)
    tests, by_rule = clean_tests(d)
    tests.to_csv(OUT / "rule_following_tests.csv", index=False)
    by_rule.to_csv(OUT / "rule_following_by_prescription.csv", index=False)
    print(tests.to_string())
    ld = letdown_response(d, r)
    ld.to_csv(OUT / "letdown_rule_vs_next_send.csv", index=False)
    print(ld.to_string())

    dn = donors(args.model)
    base = {}
    for size, root in ((8, "slide678_rerun_20260916"), (2, "slide678_dyad_rerun_20260917")):
        runs = sorted((ROOT / "data/json/noise_experiments" / root / "baseline").glob("rep??.json"))
        base[size] = np.mean([np.mean([float(x["sent"]) for e in json.loads(p.read_text())["conversation_history"]
                                       for x in e.get("dyads") or []]) for p in runs])
    dn.drop(columns=["final"]).to_csv(OUT / "transplant_donor_rules.csv", index=False)
    plot_donors(dn, base)
    for size in (8, 2):
        s = dn[(dn["size"] == size) & dn["prescribed"].notna()]
        rho, p = stats.spearmanr(s["prescribed"], s["host_send_mean"])
        print(f"transplant {size}-agent: donors with a prescription {len(s)}/{(dn['size'] == size).sum()}, "
              f"Spearman(prescribed, host mean send) = {rho:.2f} (p={p:.3f}); baseline send {base[size]:.2f}")
    print(dn[["size", "seed_type", "rep", "send_rule", "send_amount", "return_rule", "after_letdown",
              "host_send_r1", "host_send_mean"]].round(2).to_string())


if __name__ == "__main__":
    main()

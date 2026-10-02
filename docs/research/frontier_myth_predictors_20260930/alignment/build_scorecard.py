#!/usr/bin/env python3
"""Scorecard rows + figure for the H1 norm-alignment lens, September beside frontier.

Reads <dataset>/{primary_models,secondary_models,founding_window_R4,reverse_models,
artefact_checks}.csv written by norm_alignment_fast.py and writes scorecard_rows.csv and
h1_alignment_primary_grid.png in this folder.

Status is mechanical (Holm within stratum < 0.05 -> yes; raw p < 0.05 -> suggestive;
fewer than 8 contributing runs -> underpowered) with one documented override: a DeepSeek-
label Holm hit whose GLM-label primary is null is downgraded to suggestive (judge-dependent).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COLS = ["dataset", "setting", "population", "family", "finding", "effect", "ci_low", "ci_high", "p", "holm_p",
        "n_runs", "n_obs", "status", "note"]
OUT_NAME = {"sent_frac": "send fraction", "return_proportion": "return proportion", "giving_gap": "giving gap"}


def setting_of(r) -> str:
    return r["stratum"].split(" | ")[0] if " | " in r["stratum"] else r["setting"] + " (all, pooled)"


def base_note(r) -> str:
    bits = [f"per SD of measure (raw sd {r['raw_sd']:.3f})" if pd.notna(r.get("raw_sd")) else "",
            f"contributing runs {int(r['g_eff'])}" if pd.notna(r.get("g_eff")) else ""]
    if isinstance(r.get("skip_reason"), str):
        bits.append(r["skip_reason"])
    return "; ".join(b for b in bits if b)


def rows_for(ds: str) -> list[dict]:
    d = HERE / ds
    prim = pd.read_csv(d / "primary_models.csv")
    art = pd.read_csv(d / "artefact_checks.csv")
    sec = pd.read_csv(d / "secondary_models.csv")
    r4 = pd.read_csv(d / "founding_window_R4.csv")
    rev = pd.read_csv(d / "reverse_models.csv")
    out = []

    def add(r, finding, status=None, note=""):
        out.append({"dataset": ds, "setting": setting_of(r), "population": r["population"], "family": r["family"],
                    "finding": finding, "effect": r.get("coef"), "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"),
                    "p": r.get("p"), "holm_p": r.get("holm_p"), "n_runs": r.get("n_runs"), "n_obs": r.get("n_games"),
                    "status": status or r["status"], "note": "; ".join(x for x in [note, base_note(r)] if x)})

    for r in prim.to_dict("records"):
        f = f"H1 primary: {r['measure']} alignment -> {OUT_NAME[r['outcome']]} (partners' latest myths before the game)"
        note, status = [], None
        if r["stratum"] in ("2-agent all", "8-agent all") and pd.notna(r.get("holm_p_sept30")):
            note.append(f"Holm over September's pooled 30-test family {r['holm_p_sept30']:.4f}")
        if r["status"] == "yes" and r["outcome"] == "giving_gap":
            a = art[(art["stratum"] == r["stratum"]) & (art["measure"] == r["measure"]) &
                    (art["outcome"] == r["outcome"]) & (art["spec"] == "gap model + own sent_frac")]
            if len(a):
                note.append(f"with own send added {a['coef'].iloc[0]:+.4f} (p {a['p'].iloc[0]:.3f})")
        if r["status"] == "yes" and r["measure"] == "give_align" and r["outcome"] == "sent_frac":
            a = art[(art["stratum"] == r["stratum"]) & (art["measure"] == "give_align") & (art["outcome"] == "sent_frac")]
            cat = a[a["spec"] == "own send scores as 0-10 categories"]
            if len(cat):
                note.append(f"negative = MISaligned pairs send more (opposite of H1); survives 0-10 categorical own "
                            f"scores ({cat['coef'].iloc[0]:+.4f}, p {cat['p'].iloc[0]:.4f})")
        if r["outcome"] == "giving_gap" and pd.notna(r.get("corr_gap_send")):
            note.append(f"corr(gap, send) {r['corr_gap_send']:.2f}, corr(gap, return) {r['corr_gap_return']:.2f}")
        add(r, f, status, "; ".join(note))

    for r in sec[sec["spec"].isin(["first meetings", "vs non-partner placebo", "DeepSeek labels",
                                   "agent FE (investor, trustee within run) + round"])].to_dict("records"):
        if "side term" in r["spec"]:
            continue
        f = f"H1 {r['spec']}: {r['measure']} alignment -> {OUT_NAME[r['outcome']]}"
        note = ""
        if r["spec"] == "vs non-partner placebo":
            pl = sec[(sec["stratum"] == r["stratum"]) & (sec["measure"] == f"np_{r['measure']}") &
                     (sec["outcome"] == r["outcome"])]
            if len(pl):
                note = f"partner alignment coef shown; non-partner placebo coef {pl['coef'].iloc[0]:+.4f} (p {pl['p'].iloc[0]:.3f})"
        if r["spec"] == "DeepSeek labels":
            gl = prim[(prim["stratum"] == r["stratum"]) & (prim["measure"] == "same_label") & (prim["outcome"] == r["outcome"])]
            note = "GLM level controls" + (f"; GLM-label primary {gl['coef'].iloc[0]:+.4f} (p {gl['p'].iloc[0]:.3f})" if len(gl) else "")
            if ds == "frontier":
                note = "DeepSeek frontier labels served with hidden reasoning, robustness only; " + note
        if "setting" not in r or pd.isna(r.get("setting")):
            continue
        status = None
        if r["spec"] == "DeepSeek labels" and r["status"] == "yes" and len(gl) and gl["p"].iloc[0] >= 0.05:
            status = "suggestive"
            note += "; judge-dependent: the GLM-label primary is null"
        add(r, f, status, note=note)

    for r in r4.to_dict("records"):
        f = f"H1 R4 round 1 (myth_game, before any play), {r['spec']}: {r['measure']} alignment -> {OUT_NAME[r['outcome']]}"
        add(r, f, note="Holm within stratum over R4 tests")

    for r in rev.to_dict("records"):
        f = f"H1 reverse ({r['task_order']}): {r['predictor']} -> {r['measure']}"
        r = dict(r, raw_sd=r.get("sd_after"))
        add(r, f, note="per unit of predictor; raw sd shown is the after-measure's")
    return out


def grid(prim: dict[str, pd.DataFrame], path: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("div", ["#2b6cb0", "#e8e8e8", "#c05621"])
    meas = ["same_label", "moral_cos", "rule_index", "give_align", "myth_cos"]
    outs = ["sent_frac", "return_proportion", "giving_gap"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 15), gridspec_kw={"height_ratios": [17, 12], "hspace": 0.25})
    for ax, (ds, p) in zip(axes, prim.items()):
        art = pd.read_csv(HERE / ds / "artefact_checks.csv")
        art = art[art["spec"] == "gap model + own sent_frac"]
        strata = list(dict.fromkeys(p["stratum"]))
        M = np.full((len(strata), 15), np.nan)
        txt = [["" for _ in range(15)] for _ in strata]
        for i, st in enumerate(strata):
            for j, (o, m) in enumerate([(o, m) for o in outs for m in meas]):
                r = p[(p["stratum"] == st) & (p["outcome"] == o) & (p["measure"] == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                if str(r["status"]).startswith("not estimable"):
                    txt[i][j] = "ceil"
                    continue
                if r["status"] == "underpowered":
                    txt[i][j] = "few"
                    continue
                M[i, j] = r["coef"]
                txt[i][j] = "H" if r["status"] == "yes" else ("*" if r["p"] < 0.05 else "")
                a = art[(art["stratum"] == st) & (art["measure"] == m) & (art["outcome"] == o)]
                if txt[i][j] == "H" and o == "giving_gap" and len(a) and a["p"].iloc[0] >= 0.05:
                    txt[i][j] = "H†"  # kept for safety; the few-send-runs rule now pre-empts these
        ax.imshow(M, cmap=cmap, vmin=-0.06, vmax=0.06, aspect="auto")
        for i in range(len(strata)):
            for j in range(15):
                if txt[i][j]:
                    ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=7 if len(txt[i][j]) > 2 else 10,
                            color="#555555" if len(txt[i][j]) > 2 else "#111111")
        ax.set_xticks(range(15))
        ax.set_xticklabels([m.replace("_", " ") for _ in outs for m in meas], rotation=90, fontsize=8)
        for k, o in enumerate(outs):
            ax.text(k * 5 + 2, -1.0, OUT_NAME[o], ha="center", fontsize=10, fontweight="bold")
            if k:
                ax.axvline(k * 5 - 0.5, color="white", lw=3)
        ax.set_yticks(range(len(strata)))
        ax.set_yticklabels([s.replace("homogeneous", "hom.") for s in strata], fontsize=8)
        ax.set_title(f"{ds}", pad=24, fontsize=13, loc="left", fontweight="bold")
        for s in ax.spines.values():
            s.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.06, 0.06))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("effect of +1 SD alignment (outcome units, 0-1 scale)")
    fig.suptitle("H1: does norm alignment between partners predict their play? Primary model per stratum\n"
                 "H = Holm-significant within stratum; H† = Holm hit that vanishes once the pair's own send is controlled;\n"
                 "* = raw p<0.05; few = <8 contributing runs (or saturated FE); ceil = no variation",
                 fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = rows_for("september") + rows_for("frontier")
    sc = pd.DataFrame(rows)[COLS]
    sc.to_csv(HERE / "scorecard_rows.csv", index=False)
    grid({ds: pd.read_csv(HERE / ds / "primary_models.csv") for ds in ["september", "frontier"]},
         HERE / "h1_alignment_primary_grid.png")
    print(sc.groupby(["dataset", "status"]).size())


if __name__ == "__main__":
    main()

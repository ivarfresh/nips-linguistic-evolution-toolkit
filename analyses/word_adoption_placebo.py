#!/usr/bin/env python3
"""Future-myth placebo for word adoption (paper Section 4.5, "the shown myth beats the placebo").

Reproduces the shown-vs-unseen word-adoption numbers behind Figure 5a
(analyses/linguistic_uptake.py on the September n = 10 corpus), then adds a
placebo: the shown myth's author's NEXT myth.

Timing. The reader R writes its round-t myth (the outcome) after being shown
author A's round-(t-1) myth (the shown myth; always t-1, always R's partner at
t-1, asserted below). The placebo is A's round-t myth. It is written in the
same round as the outcome, so R cannot have seen it before writing; R would
only be shown it (dyads always, populations if A is R's partner in round t)
before writing its round-(t+1) myth, which is not the outcome here.

All adoption shares use the paper's definition: of the comparison myth's words
that are new to the reader (not in any of R's own myths before round t), the
share that R uses in its round-t myth.

Variants:
  placebo_raw     A's round-t myth, all its new-to-reader words (the placebo in the paper)
  shown_only      shown-myth words that are absent from A's round-t myth
  future_unseen   A's round-t words absent from every myth R was shown up to round t
                  (A's next-myth words the reader truly never saw anywhere)
  sym_shown_only  shown-myth words absent from A's round-t myth AND from every myth R
                  was shown earlier; the symmetric partner of future_unseen (which by
                  construction equals sym_future_only)

Inputs (gitignored, mirrored on the shared HF dataset under
ivarfresh/analysis/linguistic_n10_20261001/): data/analysis/linguistic_n10_20261001/
{myths.csv, uptake_children.csv} and the committed
docs/figures/linguistic_analysis_n10_20261001/reuse_summary.csv as the gate.
Outputs: docs/figures/word_adoption_placebo_20261008/. No API calls.

Run from the repo root: python analyses/word_adoption_placebo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LINGUISTIC_DATASET", "september_n10")
from analyses.linguistic_uptake import (DATA, FIGS, MIN_WORDS, child_table, content_words, holm,  # noqa: E402
                                        index_of, null_candidates, own_history, run_level_summary)

OUT = ROOT / "docs/figures/word_adoption_placebo_20261008"
ORDER = [(2, "homogeneous"), (2, "mixed, other family"), (8, "homogeneous"),
         (8, "mixed, same family"), (8, "mixed, other family")]
CONTRASTS = {"shown_vs_unseen": ("adopt_parent", "adopt_null"),
             "shown_vs_placebo": ("adopt_parent", "adopt_placebo"),
             "placebo_vs_unseen": ("adopt_placebo", "adopt_null"),
             "placebo_vs_unseen_same_round": ("adopt_placebo", "adopt_null_same_round"),
             "shownonly_vs_futureunseen": ("adopt_shown_only", "adopt_future_unseen"),
             "sym_shown_vs_future": ("adopt_sym_shown_only", "adopt_sym_future_only")}


def main() -> None:
    assert DATA.name == "linguistic_n10_20261001", DATA
    myths = pd.read_csv(DATA / "myths.csv")
    myths["valid"] = myths["n_words"] >= MIN_WORDS
    myths["text"] = myths["text"].fillna("")
    myths = myths.reset_index(drop=True)
    later = myths[myths["round"] > 1]
    assert (later.exposed_round == later["round"] - 1).all()

    words = [content_words(t) for t in myths["text"]]
    hist = own_history(myths, words)
    cands = null_candidates(myths)
    emb = np.zeros((len(myths), 1))

    # ---- 1. gate: reproduce the paper's child table and run-level summary exactly
    children = child_table(myths, words, hist, emb, cands)
    ref = pd.read_csv(DATA / "uptake_children.csv")
    key = ["run_id", "round", "agent"]
    chk = children.merge(ref[key + ["adopt_parent", "adopt_null"]], on=key, suffixes=("", "_ref"))
    assert len(chk) == len(children) == len(ref), (len(chk), len(children), len(ref))
    for c in ("adopt_parent", "adopt_null"):
        assert np.allclose(chk[c], chk[c + "_ref"], equal_nan=True), c
    summ = run_level_summary(children, ["size", "exposure"])
    paper = pd.read_csv(FIGS / "reuse_summary.csv")
    for c in ("adopt_parent_mean", "adopt_null_mean", "adopt_excess_runs_positive", "adopt_excess_p_holm"):
        assert np.allclose(summ[c], paper[c]), c
    print("reproduced uptake_children.csv and reuse_summary.csv exactly")

    # ---- 2. placebo: the shown author's next (round-t) myth
    idx = index_of(myths)
    # every myth word the reader was shown up to and including round t (memory window ignored: conservative)
    shown_words: dict[tuple, set] = {}
    earlier_words: dict[tuple, set] = {}  # words of myths shown BEFORE the current shown myth
    for (run, agent), g in myths.groupby(["run_id", "agent"]):
        acc: set[str] = set()
        for i in g.sort_values("round").index:
            p = idx.get((run, myths.at[i, "exposed_round"], myths.at[i, "exposed_author"])) \
                if pd.notna(myths.at[i, "exposed_author"]) else None
            if p is not None:
                acc |= words[p]
            t = int(myths.at[i, "round"])
            earlier_words[(run, agent, t)] = shown_words.get((run, agent, t - 1), set())
            shown_words[(run, agent, t)] = set(acc)

    def share(i, pool):
        new = words[i] - hist[i]
        return len(new & pool) / len(pool) if pool else np.nan

    # same-round unseen baseline: unseen same-family myths written in the reader's round t
    valid = myths[myths["valid"]]
    by_run_round = {k: g.index.to_list() for k, g in valid.groupby(["run_id", "round"])}
    by_cell_round = {k: g.index.to_list() for k, g in valid.groupby(["composition", "task_order", "round"])}

    def same_round_nulls(i, p):
        r = myths.loc[i]
        t, pfam = int(r["round"]), myths.at[p, "family"]
        if r["size"] == 8:
            return [j for j in by_run_round.get((r.run_id, t), [])
                    if myths.at[j, "agent"] not in (r.agent, r.exposed_author, r.partner_this_round)
                    and myths.at[j, "family"] == pfam]
        return [j for j in by_cell_round.get((r.composition, r.task_order, t), [])
                if myths.at[j, "run_id"] != r.run_id and myths.at[j, "family"] == pfam
                and (r.mixed or myths.at[j, "agent"] == r.exposed_author)]

    rows = []
    for i, js in cands.items():
        p = js[0]
        r = myths.loc[i]
        f = idx.get((r.run_id, int(r["round"]), r.exposed_author))  # author's myth in the reader's round
        if f is None or not myths.at[f, "valid"]:
            rows.append({"i": i, "placebo_ok": False})
            continue
        H = hist[i]
        pool_p = words[p] - H
        pool_f = words[f] - H
        seen = shown_words[(r.run_id, r.agent, int(r["round"]))]
        earlier = earlier_words[(r.run_id, r.agent, int(r["round"]))]
        rows.append({
            "i": i, "placebo_ok": True,
            "adopt_placebo": share(i, pool_f),
            "adopt_null_same_round": np.nanmean([share(i, words[j] - H) for j in same_round_nulls(i, p)] or [np.nan]),
            "adopt_shown_only": share(i, pool_p - words[f]),
            "adopt_future_unseen": share(i, pool_f - seen),
            "adopt_shared": share(i, pool_p & words[f]),
            # symmetric split: both pools drop the reader's own words and every earlier-shown myth's words
            "adopt_sym_shown_only": share(i, pool_p - words[f] - earlier),
            "adopt_sym_future_only": share(i, pool_f - words[p] - earlier),
            "n_pool_shown": len(pool_p), "n_pool_placebo": len(pool_f),
            "n_pool_shown_only": len(pool_p - words[f]), "n_pool_future_unseen": len(pool_f - seen),
            "overlap_placebo_with_shown": len(pool_f & pool_p) / len(pool_f) if pool_f else np.nan,
            "partner_now_is_author": r.partner_this_round == r.exposed_author,
        })
    pl = pd.DataFrame(rows).set_index("i")
    ch = children.copy()
    ch.index = list(cands.keys())
    ch = ch.join(pl)
    n_drop = int((~ch["placebo_ok"]).sum())
    print(f"{len(ch)} reader-myth pairs; {n_drop} dropped (placebo myth missing or under {MIN_WORDS} words)")
    ch = ch[ch["placebo_ok"]].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    ch.to_csv(OUT / "placebo_children.csv")

    # ---- 3. run-level summary: per-run means, paired Wilcoxon over runs, Holm across the five settings
    cols = ["adopt_parent", "adopt_null", "adopt_null_same_round", "adopt_placebo", "adopt_shown_only",
            "adopt_future_unseen", "adopt_shared", "adopt_sym_shown_only", "adopt_sym_future_only",
            "overlap_placebo_with_shown"]
    per_run = ch.groupby(["size", "exposure", "run_id"])[cols].mean().reset_index()
    per_run.to_csv(OUT / "placebo_per_run.csv", index=False)
    out = []
    for size, exp in ORDER:
        g = per_run[(per_run["size"] == size) & (per_run["exposure"] == exp)]
        rec = {"size": size, "exposure": exp, "n_runs": len(g),
               "n_pairs": int(((ch["size"] == size) & (ch["exposure"] == exp)).sum())}
        for c in cols:
            rec[c] = g[c].mean()
            rec[c + "_sd"] = g[c].std(ddof=1)
        for name, (a, b) in CONTRASTS.items():
            d = (g[a] - g[b]).dropna()
            rec[f"{name}_ratio"] = g[a].mean() / g[b].mean()
            rec[f"{name}_runs_pos"] = int((d > 0).sum())
            rec[f"{name}_n"] = len(d)
            rec[f"{name}_p"] = stats.wilcoxon(d).pvalue
        out.append(rec)
    out = pd.DataFrame(out)
    for name in CONTRASTS:
        out[f"{name}_p_holm"] = holm(out[f"{name}_p"].to_numpy())
    out.to_csv(OUT / "placebo_summary.csv", index=False)

    pd.set_option("display.width", 250)
    print(out[["size", "exposure", "n_runs", "n_pairs", "adopt_parent", "adopt_null", "adopt_placebo",
               "shown_vs_unseen_ratio", "shown_vs_unseen_runs_pos", "shown_vs_placebo_ratio",
               "shown_vs_placebo_runs_pos", "shown_vs_placebo_p_holm", "placebo_vs_unseen_ratio",
               "placebo_vs_unseen_runs_pos", "placebo_vs_unseen_p_holm"]].round(4).to_string(index=False))
    print(out[["size", "exposure", "adopt_shown_only", "adopt_future_unseen", "adopt_shared",
               "shownonly_vs_futureunseen_ratio", "shownonly_vs_futureunseen_runs_pos",
               "shownonly_vs_futureunseen_p_holm", "overlap_placebo_with_shown"]].round(4).to_string(index=False))
    print(out[["size", "exposure", "adopt_sym_shown_only", "adopt_sym_future_only", "sym_shown_vs_future_ratio",
               "sym_shown_vs_future_runs_pos", "sym_shown_vs_future_p_holm"]].round(4).to_string(index=False))
    print(out[["size", "exposure", "adopt_null", "adopt_null_same_round", "adopt_placebo",
               "placebo_vs_unseen_same_round_ratio", "placebo_vs_unseen_same_round_runs_pos",
               "placebo_vs_unseen_same_round_n", "placebo_vs_unseen_same_round_p_holm"]].to_string(index=False))
    na = ch[["adopt_null_same_round", "adopt_shown_only", "adopt_sym_shown_only", "adopt_sym_future_only"]].isna().mean()
    print("share of reader myths with an empty pool / no candidate:\n", na.round(4).to_string())

    # populations, split by whether the shown author is also the reader's partner this round
    for flag, g in ch[ch["size"] == 8].groupby("partner_now_is_author"):
        pr = g.groupby(["exposure", "run_id"])[["adopt_parent", "adopt_placebo", "adopt_null"]].mean()
        print("8-agent, author is current partner =", flag, len(g), "pairs")
        print(pr.groupby(level=0).mean().round(4).to_string())


if __name__ == "__main__":
    main()

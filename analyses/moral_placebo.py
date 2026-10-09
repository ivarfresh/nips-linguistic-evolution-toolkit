#!/usr/bin/env python3
"""Future-myth placebo for moral-label carry-over, September n = 10 corpus (paper Section 4.5).

Reproduces the paper's moral-matching numbers (8-agent mixed populations, myth->game,
same-label share for the shown myth minus an unseen same-family myth; per-run means, paired
Wilcoxon over runs, no Holm) from the cached GLM-5.2 labels, exactly as
analyses/moral_carryover.py:summary_measures + run_summary do, then adds:

  placebo      the shown author's NEXT myth (round exposed_round + 1, written in the same round as
               the reader's myth; the reader never saw it before writing)
  unseen_next  the next myths (round exposed_round + 1) of the same unseen same-family authors
               used for the unseen baseline (the placebo's own baseline; removes round drift)

Inputs (gitignored, mirrored on the shared HF dataset under
ivarfresh/analysis/linguistic_n10_20261001/): data/analysis/linguistic_n10_20261001/
{myths.csv, moral_labels_z-ai__glm-5.2.csv, embeddings_moral_summary_mpnet.npy,
moral_uptake_children.csv (gate)}. Outputs: docs/figures/moral_placebo_20261008/. No API calls.

Run from the repo root: python analyses/moral_placebo.py
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
from analyses import linguistic_uptake as lu  # noqa: E402

DATA = lu.DATA
OUT = ROOT / "docs/figures/moral_placebo_20261008"
KEY = ["run_id", "round", "agent"]
MAIN_SETTING, MAIN_ORDER = "8-agent mixed", "myth_game"
SUMMARY_COLS = ["n_runs", "shown", "unseen", "placebo", "unseen_next", "shown_minus_unseen", "shown_minus_unseen_p",
                "shown_minus_placebo", "shown_minus_placebo_ci_lo", "shown_minus_placebo_ci_hi",
                "shown_minus_placebo_p", "shown_minus_placebo_runs_pos", "shown_minus_placebo_runs_neg",
                "placebo_minus_unseen_next", "placebo_minus_unseen_next_p", "dd", "dd_p"]


def wil(v):
    v = pd.Series(v).dropna().round(10)
    if len(v) < 5 or not (v != 0).any():
        return np.nan, int((v > 0).sum()), int((v < 0).sum())
    return stats.wilcoxon(v).pvalue, int((v > 0).sum()), int((v < 0).sum())


def boot_ci(v, n=10000, seed=0):
    v = pd.Series(v).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    m = rng.choice(v, (n, len(v))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def summarise(df, by, metric="same_label"):
    sh, un, pl, un2 = (f"{metric}_shown", f"{metric}_unseen", f"{metric}_placebo", f"{metric}_unseen_next")
    per_run = df.groupby(by + ["run_id"])[[sh, un, pl, un2]].mean().reset_index()
    per_run["shown_minus_unseen"] = per_run[sh] - per_run[un]
    per_run["shown_minus_placebo"] = per_run[sh] - per_run[pl]
    per_run["placebo_minus_unseen_next"] = per_run[pl] - per_run[un2]
    per_run["dd"] = per_run["shown_minus_unseen"] - per_run["placebo_minus_unseen_next"]
    out = []
    for keys, g in per_run.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = {**dict(zip(by, keys)), "n_runs": len(g), "n_children": int(df.set_index(by).loc[keys].shape[0])}
        for c, name in ((sh, "shown"), (un, "unseen"), (pl, "placebo"), (un2, "unseen_next")):
            rec[name] = g[c].mean()
            rec[name + "_sd"] = g[c].std(ddof=1)
        for d in ["shown_minus_unseen", "shown_minus_placebo", "placebo_minus_unseen_next", "dd"]:
            p, pos, neg = wil(g[d])
            lo, hi = boot_ci(g[d])
            rec.update({d: g[d].mean(), d + "_sd": g[d].std(ddof=1), d + "_ci_lo": lo, d + "_ci_hi": hi,
                        d + "_p": p, d + "_runs_pos": pos, d + "_runs_neg": neg})
        out.append(rec)
    return pd.DataFrame(out)


def main() -> None:
    assert DATA.name == "linguistic_n10_20261001", DATA
    myths = lu.load_myths()
    lab = pd.read_csv(DATA / "moral_labels_z-ai__glm-5.2.csv")
    myths = myths.merge(lab[KEY + ["label", "summary"]], on=KEY, how="left")
    myths["setting"] = [f"{s}-agent {'mixed' if m else 'homogeneous'}" for s, m in zip(myths["size"], myths["mixed"])]
    emb = np.load(DATA / "embeddings_moral_summary_mpnet.npy")
    assert emb.shape[0] == len(myths), (emb.shape, len(myths))
    has = myths["summary"].notna().to_numpy()
    labs = myths["label"].to_numpy()
    idx = {k: i for i, k in enumerate(zip(myths["run_id"], myths["round"], myths["agent"]))}

    def islab(j):
        return isinstance(labs[j], str)

    cands = lu.null_candidates(myths)
    rows = []
    for i, js in cands.items():
        p, nulls = js[0], [j for j in js[1:] if has[j]]
        if not (has[i] and has[p]) or not nulls:
            continue  # identical filter to moral_carryover.summary_measures
        r = myths.loc[i]
        lab_i, lab_p = labs[i], labs[p]
        rec = {"run_id": r.run_id, "setting": r.setting, "family": r.family, "agent": r.agent,
               "parent_family": myths.at[p, "family"], "round": r["round"], "task_order": r.task_order,
               "author": myths.at[p, "agent"], "exposed_round": myths.at[p, "round"],
               "moral_cos_shown": float(emb[i] @ emb[p]), "moral_cos_unseen": float(np.mean(emb[nulls] @ emb[i])),
               "same_label_shown": float(lab_i == lab_p) if isinstance(lab_i, str) and isinstance(lab_p, str) else np.nan,
               "same_label_unseen": np.nanmean([float(lab_i == labs[j]) for j in nulls if islab(j)]) if isinstance(lab_i, str) else np.nan}
        # placebo: the shown author's next myth
        f = idx.get((myths.at[p, "run_id"], myths.at[p, "round"] + 1, myths.at[p, "agent"]))
        rec["placebo_exists"] = f is not None and has[f] and islab(f)
        rec["same_label_placebo"] = float(lab_i == labs[f]) if rec["placebo_exists"] and isinstance(lab_i, str) else np.nan
        rec["moral_cos_placebo"] = float(emb[i] @ emb[f]) if rec["placebo_exists"] else np.nan
        # reciprocity: did the author read the reader's myth when writing the placebo myth?
        rec["author_read_child"] = (myths.at[f, "exposed_author"] == r.agent) if f is not None else np.nan
        # unseen-next: the same unseen authors' next myths
        nxt = [idx.get((myths.at[j, "run_id"], myths.at[j, "round"] + 1, myths.at[j, "agent"])) for j in nulls]
        nxt = [k for k in nxt if k is not None and has[k] and islab(k)]
        rec["n_unseen_next"] = len(nxt)
        rec["same_label_unseen_next"] = np.mean([float(lab_i == labs[k]) for k in nxt]) if nxt and isinstance(lab_i, str) else np.nan
        rec["moral_cos_unseen_next"] = float(np.mean(emb[nxt] @ emb[i])) if nxt else np.nan
        rows.append(rec)

    U = pd.DataFrame(rows)
    U["exposure"] = np.where(U["family"] == U["parent_family"], "same family", "other family")
    U["same_label_excess"] = U["same_label_shown"] - U["same_label_unseen"]
    U["moral_cos_excess"] = U["moral_cos_shown"] - U["moral_cos_unseen"]
    OUT.mkdir(parents=True, exist_ok=True)
    U.to_csv(OUT / "moral_placebo_children.csv", index=False)

    # ---- gate: reproduce the stored child table exactly
    stored = pd.read_csv(DATA / "moral_uptake_children.csv")
    assert len(stored) == len(U), (len(stored), len(U))
    for c in ["same_label_shown", "same_label_unseen", "moral_cos_shown", "moral_cos_unseen"]:
        assert np.allclose(stored[c].to_numpy(), U[c].to_numpy(), equal_nan=True, atol=1e-6), c
    print(f"child table reproduces the stored moral_uptake_children.csv ({len(U)} rows)")

    # paper numbers on the full set (no placebo restriction): 2.7 pp p=0.015 within, 1.6 pp p=0.13 across
    paper = U[(U.setting == MAIN_SETTING) & (U.task_order == MAIN_ORDER)]
    paper_rows = []
    for exp, g in paper.groupby("exposure"):
        v = g.groupby("run_id")["same_label_excess"].mean()
        p, pos, neg = wil(v)
        paper_rows.append({"exposure": exp, "n_runs": len(v), "shown_minus_unseen": v.mean(), "p": p,
                           "runs_pos": pos, "runs_neg": neg})
        print(f"paper row, {exp}: excess {v.mean():.4f}  p={p:.4f}  runs={len(v)}  ({pos}+/{neg}-)")
    pd.DataFrame(paper_rows).to_csv(OUT / "paper_rows_8agent_mixed_myth_game.csv", index=False)

    print("author read the child's myth when writing the placebo myth (8-agent mixed, myth_game):",
          paper["author_read_child"].astype(float).mean().round(4))
    print("same, all 8-agent rows:", U[U.setting.str.startswith("8")]["author_read_child"].astype(float).mean().round(4))

    # restricted set: every quantity present for the same children
    R = U.dropna(subset=["same_label_shown", "same_label_unseen", "same_label_placebo", "same_label_unseen_next"])
    print(f"restricted children: {len(R)} of {len(U)}; paper subset "
          f"{len(R[(R.setting == MAIN_SETTING) & (R.task_order == MAIN_ORDER)])} of {len(paper)}")
    main_tab = summarise(R[(R.setting == MAIN_SETTING) & (R.task_order == MAIN_ORDER)], ["exposure"])
    main_tab.to_csv(OUT / "placebo_main_8agent_mixed_myth_game.csv", index=False)
    allt = summarise(R, ["setting", "exposure", "task_order"])
    allt.to_csv(OUT / "placebo_by_setting_task_order.csv", index=False)
    Rc = U.dropna(subset=["moral_cos_shown", "moral_cos_unseen", "moral_cos_placebo", "moral_cos_unseen_next"])
    cos = summarise(Rc[(Rc.setting == MAIN_SETTING) & (Rc.task_order == MAIN_ORDER)], ["exposure"], "moral_cos")
    cos.to_csv(OUT / "placebo_main_moral_cos.csv", index=False)

    pd.set_option("display.width", 250)
    print("\nMAIN (8-agent mixed, myth->game, same label):")
    print(main_tab[["exposure"] + SUMMARY_COLS].round(4).T.to_string())
    print("\nMAIN, moral-summary cosine:")
    print(cos[["exposure"] + SUMMARY_COLS].round(4).T.to_string())
    print("\nALL settings:")
    print(allt[["setting", "exposure", "task_order"] + SUMMARY_COLS].round(4).to_string())


if __name__ == "__main__":
    main()

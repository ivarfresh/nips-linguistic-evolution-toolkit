#!/usr/bin/env python3
"""Where can the frontier myth runs show a myth effect at all? Spread of sends and returns per cell.

A cell whose send fraction (sent / 5) or return proportion barely varies cannot show
that a myth moves play, whatever the myth says. For every composition x size x task
order x family (x first sender in mixed dyads) x round, this reports the spread of both
outcomes and flags the cells that are locked.

Reads data/analysis/linguistic_frontier_20260930/decisions.csv (analyses/linguistic_corpus.py
--dataset frontier) and writes, to the same folder:
  testability_by_round.csv   one row per cell x round
  testability_round1.csv     round 1 only (the founding window in myth->game)
  testability_cells.csv      pooled over rounds, with the share of rounds that are locked

  python3 analyses/frontier_testability.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses import linguistic_datasets  # noqa: E402

DATA = linguistic_datasets.get("frontier").data
LOCKED_SD = 0.02    # below this, treat the outcome as constant (send: $0.10 of $5)
CEILING_SHARE = 0.95


def first_sender(dec: pd.DataFrame) -> pd.Series:
    """Family of the round-1 investor in each dyad run (Agent_1 in mixed dyads)."""
    r1 = dec[(dec["round"] == 1) & (dec["role"] == "investor") & (dec["size"] == 2)]
    return r1.drop_duplicates("run_id").set_index("run_id")["family"]


def describe(g: pd.DataFrame) -> pd.Series:
    send = g.loc[g.role == "investor", "sent"] / 5
    ret = g.loc[g.role == "trustee", "return_proportion"].dropna()
    out = {"n_runs": g.run_id.nunique(), "n_send": len(send), "n_return": len(ret),
           "send_mean": send.mean(), "send_sd": send.std(ddof=0), "send_share_full": (send == 1).mean(),
           "send_share_modal": send.value_counts(normalize=True).max() if len(send) else np.nan,
           "return_mean": ret.mean(), "return_sd": ret.std(ddof=0),
           "return_share_modal": ret.round(3).value_counts(normalize=True).max() if len(ret) else np.nan}
    out["send_locked"] = bool(len(send) and (out["send_sd"] < LOCKED_SD or out["send_share_modal"] >= CEILING_SHARE))
    out["send_at_ceiling"] = bool(len(send) and out["send_share_full"] >= CEILING_SHARE)
    out["return_locked"] = bool(len(ret) and (out["return_sd"] < LOCKED_SD or out["return_share_modal"] >= CEILING_SHARE))
    return pd.Series(out)


def main() -> None:
    dec = pd.read_csv(DATA / "decisions.csv")
    fs = first_sender(dec)
    dec["first_sender"] = np.where(dec["mixed"] & (dec["size"] == 2), dec["run_id"].map(fs), "")
    keys = ["size", "mixed", "composition", "task_order", "family", "first_sender"]
    by_round = dec.groupby(keys + ["round"]).apply(describe, include_groups=False).reset_index()
    by_round.to_csv(DATA / "testability_by_round.csv", index=False)
    by_round[by_round["round"] == 1].to_csv(DATA / "testability_round1.csv", index=False)
    pooled = dec.groupby(keys).apply(describe, include_groups=False).reset_index()
    # share of the rounds in which the family actually made that decision (a mixed-dyad family
    # sends only every other round)
    share = pd.DataFrame({
        f"rounds_{kind}_locked_share": by_round[by_round[f"n_{kind}"] > 0].groupby(keys)[f"{kind}_locked"].mean()
        for kind in ("send", "return")}).reset_index()
    pooled = pooled.merge(share, on=keys)
    pooled.to_csv(DATA / "testability_cells.csv", index=False)
    pd.set_option("display.width", 250)
    cols = keys + ["n_runs", "n_send", "send_mean", "send_sd", "send_share_full", "rounds_send_locked_share",
                   "n_return", "return_mean", "return_sd", "rounds_return_locked_share"]
    print(pooled[cols].round(3).to_string(index=False))
    r1 = by_round[by_round["round"] == 1]
    print("\nround 1:")
    print(r1[keys + ["n_send", "send_mean", "send_sd", "send_locked", "n_return", "return_mean", "return_sd",
                     "return_locked"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

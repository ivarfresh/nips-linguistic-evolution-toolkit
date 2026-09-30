#!/usr/bin/env python3
"""Decision-level table: each decision with the baseline (everything the agent sees in the
decision prompt, plus its own lagged moves) and the features of its own latest myth and of the
latest myth it was shown. Timing follows analyses/moral_carryover.py: the myth before a
round-t decision is round t (myth_game) or round t-1 (game_myth); the shown myth is the one the
agent read just before writing that own myth. Both sit in the decision context.

Also tags each decision with its split:
  discovery   game_myth (both sizes, rounds >= 3) + 2-agent myth_game rounds >= 2
  R3          8-agent myth_game rounds >= 2 (held out from discovery)
  R4          myth_game round 1 (both sizes; held out)
Writes decision_table.csv and myth_index.npy-free columns own_i / shown_i (row in myths.csv).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/linguistic-mixed/data/analysis/linguistic_20260923")
OUT = Path(__file__).resolve().parent


def main() -> None:
    myths = pd.read_csv(D / "myths.csv")
    feat = pd.read_csv(OUT / "myth_features.csv")
    assert len(myths) == len(feat) == 8520
    idx = {(r, t, a): i for i, (r, t, a) in enumerate(zip(myths["run_id"], myths["round"], myths["agent"]))}

    dec = pd.read_csv(D / "decisions.csv").sort_values(["run_id", "agent", "round"]).reset_index(drop=True)
    dec["coop"] = np.where(dec["role"] == "investor", dec["sent"] / 5, dec["return_proportion"])
    dec["setting"] = dec["size"].astype(str) + "-agent " + np.where(dec["mixed"], "mixed", "homogeneous")
    dec["run_agent"] = dec["run_id"] + "|" + dec["agent"]

    # Own history (strictly before this round).
    rows = []
    by_agent = {k: g for k, g in dec.groupby(["run_id", "agent"])}
    for (run, ag), g in by_agent.items():
        last = {"send": np.nan, "ret": np.nan, "got_ret": np.nan, "got_send": np.nan}
        sends, rets = [], []
        for r in g.itertuples():
            rows.append({"_row": r.Index, "lag_send": last["send"], "lag_return": last["ret"],
                         "lag_got_return": last["got_ret"], "lag_got_send": last["got_send"],
                         "mean_send_sofar": np.mean(sends) if sends else np.nan,
                         "mean_return_sofar": np.mean(rets) if rets else np.nan})
            if r.role == "investor":
                last["send"], last["got_ret"] = r.sent / 5, r.return_proportion
                sends.append(r.sent / 5)
            else:
                last["ret"], last["got_send"] = r.return_proportion, r.sent / 5
                rets.append(r.return_proportion)
    h = pd.DataFrame(rows).set_index("_row").sort_index()
    dec = dec.join(h)

    # Co-player's last 3 games (shown in the 8-agent prompt; in dyads it mirrors own history).
    cop = {}
    for (run, ag), g in by_agent.items():
        g = g.sort_values("round")
        for r in g.itertuples():
            prev = g[(g["round"] < r.round) & (g["round"] >= r.round - 3)]
            s = prev.loc[prev["role"] == "investor", "coop"]
            t = prev.loc[prev["role"] == "trustee", "coop"]
            cop[(run, ag, r.round)] = (s.mean() if len(s) else np.nan, t.mean() if len(t) else np.nan)
    dec["cop_last3_send"] = [cop.get((a, b, c), (np.nan, np.nan))[0] for a, b, c in zip(dec.run_id, dec.partner, dec["round"])]
    dec["cop_last3_return"] = [cop.get((a, b, c), (np.nan, np.nan))[1] for a, b, c in zip(dec.run_id, dec.partner, dec["round"])]
    dec["received_now"] = np.where(dec["role"] == "trustee", dec["sent"] / 5, np.nan)
    dec["dsend"] = np.where(dec["role"] == "investor", dec["coop"] - dec["lag_send"], np.nan)

    own_i, shown_i, shown_author, shown_round = [], [], [], []
    for g in dec.itertuples():
        rb = g.round if g.task_order == "myth_game" else g.round - 1
        i = idx.get((g.run_id, rb, g.agent))
        own_i.append(i if i is not None else -1)
        j = -1
        ea, er = None, np.nan
        if i is not None and isinstance(myths.at[i, "exposed_author"], str):
            ea, er = myths.at[i, "exposed_author"], myths.at[i, "exposed_round"]
            j = idx.get((g.run_id, int(er), ea), -1)
        shown_i.append(j)
        shown_author.append(ea)
        shown_round.append(er)
    dec["own_i"], dec["shown_i"], dec["shown_author"], dec["shown_round"] = own_i, shown_i, shown_author, shown_round

    fcols = [c for c in feat.columns if c not in ("run_id", "round", "agent", "family")]
    F = feat[fcols].to_numpy(float)
    for src in ("own", "shown"):
        ii = dec[f"{src}_i"].to_numpy()
        vals = np.where(ii[:, None] >= 0, F[np.clip(ii, 0, None)], np.nan)
        dec = pd.concat([dec, pd.DataFrame(vals, columns=[f"{src}_{c}" for c in fcols], index=dec.index)], axis=1)
    dec["shown_family"] = [myths.at[j, "family"] if j >= 0 else None for j in dec["shown_i"]]

    mg, r = dec["task_order"] == "myth_game", dec["round"]
    dec["split"] = "none"
    dec.loc[(dec["task_order"] == "game_myth") & (r >= 3), "split"] = "discovery"
    dec.loc[mg & (dec["size"] == 2) & (r >= 2), "split"] = "discovery"
    dec.loc[mg & (dec["size"] == 8) & (r >= 2), "split"] = "R3"
    dec.loc[mg & (r == 1), "split"] = "R4"
    dec.to_csv(OUT / "decision_table.csv", index=False)
    print(dec.groupby(["split", "role"]).size())
    print(dec.groupby(["split", "family", "role"])["coop"].agg(["mean", "std", "size"]).round(3))


if __name__ == "__main__":
    main()

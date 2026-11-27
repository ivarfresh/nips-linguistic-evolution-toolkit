#!/usr/bin/env python3
"""No-self-line dyads (2026-11-27): full Sonnet 4.5 dyad runs without the self-inspiration sentence.

Compares the 10 runs of data/json/noise_experiments/no_self_line_dyads_20261127 with their
paired September baseline runs (same replicate, same noise seed; no-defector cell of
negative_only_reasoning_rerun_dyad_{game_myth,myth_game}_claude_n5). Measures per run:

  resources      round-10 cumulative resources averaged over the two agents (as Figure 2)
  send           mean amount sent (of $5)
  return_share   mean returned / received
  self_copy      share of the agent's previous myth's content words that its next myth reuses
  partner_uptake among the shown partner myth's words the agent never used before, the share
                 its next myth uses
  unseen_uptake  the same share for the same seat's myth of that round in the other runs of
                 the same condition (never shown)
  borrowing      partner_uptake - unseen_uptake

Word definitions follow analyses/linguistic_uptake.py. Differences are paired by replicate
(n = 5 per task order) with a 95% t interval.

  python3 analyses/no_self_line_dyads.py --data-root <checkout holding the September runs>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses.linguistic_uptake import MIN_WORDS, content_words  # noqa: E402

NEW = "data/json/noise_experiments/no_self_line_dyads_20261127/no_self_line_dyad_{o}_claude_n5"
OLD = "data/json/noise_experiments/negative_only_crossmodel_reasoning_rerun_20260909/negative_only_reasoning_rerun_dyad_{o}_claude_n5"
CELL = "claude-sonnet-4.5/{o}/noisy2_crossmodel_negative_twotask_r3"
LINE = "Use the myth you wrote in the previous round as inspiration"
OUT = ROOT / "docs/research/no_self_line_dyads_20261127"
METRICS = ["resources", "send", "return_share", "self_copy", "partner_uptake", "unseen_uptake", "borrowing"]


def load(arm: str, order: str, root: Path) -> list[dict]:
    files = sorted((root / (NEW if arm == "no_line" else OLD).format(o=order) / CELL.format(o=order)).glob("*_anything.json"))
    runs = [json.loads(f.read_text()) for f in files]
    assert sorted(r["run_metadata"]["replicate_id"] for r in runs) == [0, 1, 2, 3, 4], (arm, order, len(runs))
    for r in runs:
        m = r["run_metadata"]
        assert m["defector_count"] == 0 and m["random_defection_probability"] == 0 and m["num_agents"] == 2
        sent = [str(x.get("content")) for a in r["agents"].values() for e in a["interaction_history"]
                for x in e.get("messages_sent") or []]
        assert any(LINE in s for s in sent) == (arm == "line"), (arm, order)
    return runs


def measures(runs: list[dict]) -> pd.DataFrame:
    words = {}  # (replicate, round, agent) -> content words of a valid myth
    for r in runs:
        rep = r["run_metadata"]["replicate_id"]
        for e in r["conversation_history"]:
            for agent, text in (e.get("myths") or {}).items():
                if len(str(text).split()) >= MIN_WORDS:
                    words[(rep, int(e["round"]), agent)] = content_words(str(text))
    rows = []
    for r in runs:
        m = r["run_metadata"]
        rep = m["replicate_id"]
        hist = r["conversation_history"]
        assert [e["round"] for e in hist] == list(range(1, 11))
        dy = [d for e in hist for d in e["dyads"]]
        row = {"replicate": rep, "noise_seed": m["noise_seed"], "pairing_seed": m["pairing_seed"],
               "resources": float(np.mean(list(hist[-1]["balances"].values()))),
               "send": float(np.mean([d["sent"] for d in dy])),
               "return_share": float(np.mean([d["returned"] / d["received"] for d in dy if d["received"] > 0]))}
        own, par, null = [], [], []
        for e in hist[1:]:
            rnd = int(e["round"])
            for agent, exp in (e.get("myth_exposures") or {}).items():
                new, prev = words.get((rep, rnd, agent)), words.get((rep, rnd - 1, agent))
                shown = words.get((rep, exp.get("source_round"), exp.get("original_author_id")))
                if new is None or prev is None or shown is None:
                    continue
                seen = frozenset().union(*[words.get((rep, k, agent), frozenset()) for k in range(1, rnd)])
                own.append(len(new & prev) / len(prev))
                fresh = new - seen

                def uptake(pool):
                    pool = pool - seen
                    return len(fresh & pool) / len(pool) if pool else np.nan

                par.append(uptake(shown))
                null.append(np.nanmean([uptake(w) for (q, k, a), w in words.items()
                                        if q != rep and k == exp["source_round"] and a == exp["original_author_id"]]))
        row.update(self_copy=np.mean(own), partner_uptake=np.nanmean(par), unseen_uptake=np.nanmean(null), myths=len(own))
        row["borrowing"] = row["partner_uptake"] - row["unseen_uptake"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=ROOT, help="checkout holding the September baseline runs")
    args = ap.parse_args()
    per_run, summary = [], []
    for order in ("game_myth", "myth_game"):
        a = measures(load("line", order, args.data_root)).assign(arm="line", task_order=order)
        b = measures(load("no_line", order, ROOT)).assign(arm="no_line", task_order=order)
        per_run += [a, b]
        pair = a.merge(b, on="replicate", suffixes=("_line", "_no_line"))
        assert len(pair) == 5 and (pair["noise_seed_line"] == pair["noise_seed_no_line"]).all()
        for mt in METRICS:
            d = pair[f"{mt}_no_line"] - pair[f"{mt}_line"]
            half = stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
            summary.append({"task_order": order, "metric": mt, "n": 5,
                            "line_mean": a[mt].mean(), "line_sd": a[mt].std(ddof=1),
                            "no_line_mean": b[mt].mean(), "no_line_sd": b[mt].std(ddof=1),
                            "diff": d.mean(), "ci_lo": d.mean() - half, "ci_hi": d.mean() + half,
                            "p_paired_t": stats.ttest_1samp(d, 0).pvalue if d.std(ddof=1) > 0 else np.nan,
                            "replicates_higher": int((d > 0).sum())})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(per_run).to_csv(OUT / "per_run.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(3).to_string(index=False))
    print(pd.concat(per_run)[["task_order", "arm", "replicate", "resources", "send", "return_share", "self_copy", "borrowing", "myths"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()

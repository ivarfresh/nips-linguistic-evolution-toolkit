#!/usr/bin/env python3
"""No-self-line mixed dyads (2026-11-27): Sonnet 4.5 with GPT-5 Nano / Gemini 3.7 Flash.

Compares the 20 runs of data/json/noise_experiments/no_self_line_mixed_dyads_20261127 with
their paired September mixed-dyad runs (same replicate 0-4, same seeds, same first sender).
Run-level measure: round-10 resources per agent. Per-family measures (the Sonnet agent and
its partner separately): mean send, return share, self-copying, partner-word uptake, and
borrowing (uptake minus uptake from a never-shown myth: the same family's myth of that round
in the other runs of the same condition and arm). Definitions as analyses/no_self_line_dyads.py.
Differences are paired by replicate with a 95% t interval.

  python3 analyses/no_self_line_mixed_dyads.py --data-root <checkout holding the September runs>
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

NEW = "data/json/noise_experiments/no_self_line_mixed_dyads_20261127/no_self_line_mixed_dyad_{o}_{a}_{b}"
OLD = "data/json/noise_experiments/mixed_model_dyads_20260917/mixed_dyad_{o}_{a}_{b}_n3"
LINE = "Use the myth you wrote in the previous round as inspiration"
OUT = ROOT / "docs/research/no_self_line_dyads_20261127"
FAMILY = {"anthropic/claude-sonnet-4.5": "sonnet", "openai/gpt-5-nano": "gpt", "google/gemini-3.7-flash": "gemini"}
METRICS = ["send", "return_share", "self_copy", "partner_uptake", "borrowing"]


def load(arm: str, order: str, other: str, root: Path) -> list[dict]:
    runs = []
    for a, b in (("sonnet", other), (other, "sonnet")):
        base = root / (NEW if arm == "no_line" else OLD).format(o=order, a=a, b=b)
        runs += [json.loads(f.read_text()) for f in sorted(base.rglob("*_anything.json")) if "quarantine" not in f.parts]
    runs = [r for r in runs if r["run_metadata"]["replicate_id"] <= 4]
    assert sorted(r["run_metadata"]["replicate_id"] for r in runs) == [0, 1, 2, 3, 4], (arm, order, other, len(runs))
    for r in runs:
        m = r["run_metadata"]
        assert m["defector_count"] == 0 and m["random_defection_probability"] == 0 and m["num_agents"] == 2
        sent = [str(x.get("content")) for a in r["agents"].values() for e in a["interaction_history"]
                for x in e.get("messages_sent") or []]
        assert any(LINE in s for s in sent) == (arm == "line"), (arm, order, other)
    return runs


def measures(runs: list[dict]) -> pd.DataFrame:
    words, fam = {}, {}  # (replicate, round, agent) -> words; (replicate, agent) -> family
    for r in runs:
        rep = r["run_metadata"]["replicate_id"]
        for agent, a in r["agents"].items():
            fam[(rep, agent)] = FAMILY[a["model"]]
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
        resources = float(np.mean(list(hist[-1]["balances"].values())))
        for agent in r["agents"]:
            own, par, null = [], [], []
            for e in hist[1:]:
                rnd = int(e["round"])
                exp = (e.get("myth_exposures") or {}).get(agent)
                if not exp:
                    continue
                new, prev = words.get((rep, rnd, agent)), words.get((rep, rnd - 1, agent))
                shown = words.get((rep, exp.get("source_round"), exp.get("original_author_id")))
                if new is None or prev is None or shown is None:
                    continue
                seen = frozenset().union(*[words.get((rep, k, agent), frozenset()) for k in range(1, rnd)])
                fresh = new - seen

                def uptake(pool):
                    pool = pool - seen
                    return len(fresh & pool) / len(pool) if pool else np.nan

                own.append(len(new & prev) / len(prev))
                par.append(uptake(shown))
                author_family = fam[(rep, exp["original_author_id"])]
                unseen = [uptake(w) for (q, k, a), w in words.items()
                          if q != rep and k == exp["source_round"] and fam[(q, a)] == author_family]
                null.append(np.nanmean(unseen) if unseen else np.nan)
            sends = [d["sent"] for d in dy if d["investor"] == agent]
            rets = [d["returned"] / d["received"] for d in dy if d["trustee"] == agent and d["received"] > 0]
            rows.append({"replicate": rep, "noise_seed": m["noise_seed"], "first_sender": fam[(rep, "Agent_1")],
                         "family": fam[(rep, agent)], "resources": resources,
                         "send": np.mean(sends), "return_share": np.mean(rets) if rets else np.nan,
                         "self_copy": np.mean(own) if own else np.nan, "partner_uptake": np.nanmean(par) if par else np.nan,
                         "borrowing": (np.nanmean(par) - np.nanmean(null)) if par else np.nan, "myths": len(own)})
    return pd.DataFrame(rows)


def paired(a: pd.DataFrame, b: pd.DataFrame, metric: str, keys: list[str]) -> dict:
    pair = a.merge(b, on=keys, suffixes=("_line", "_no_line")).dropna(subset=[f"{metric}_line", f"{metric}_no_line"])
    d = pair[f"{metric}_no_line"] - pair[f"{metric}_line"]
    n = len(d)
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {"n": n, "line_mean": pair[f"{metric}_line"].mean(), "line_sd": pair[f"{metric}_line"].std(ddof=1),
            "no_line_mean": pair[f"{metric}_no_line"].mean(), "no_line_sd": pair[f"{metric}_no_line"].std(ddof=1),
            "diff": d.mean(), "ci_lo": d.mean() - half, "ci_hi": d.mean() + half,
            "p_paired_t": stats.ttest_1samp(d, 0).pvalue if n > 1 and d.std(ddof=1) > 0 else np.nan,
            "pairs_higher": int((d > 0).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=ROOT, help="checkout holding the September baseline runs")
    args = ap.parse_args()
    per_run, summary = [], []
    for other in ("gpt", "gemini"):
        A, B = [], []
        for order in ("game_myth", "myth_game"):
            a = measures(load("line", order, other, args.data_root)).assign(task_order=order)
            b = measures(load("no_line", order, other, ROOT)).assign(task_order=order)
            chk = a.merge(b, on=["replicate", "family"])
            assert len(chk) == 10 and (chk["noise_seed_x"] == chk["noise_seed_y"]).all() and (chk["first_sender_x"] == chk["first_sender_y"]).all()
            A.append(a), B.append(b)
            per_run += [a.assign(arm="line", pair=f"sonnet+{other}"), b.assign(arm="no_line", pair=f"sonnet+{other}")]
        A, B = pd.concat(A), pd.concat(B)
        for scope in ("both", "game_myth", "myth_game"):
            a = A if scope == "both" else A[A["task_order"] == scope]
            b = B if scope == "both" else B[B["task_order"] == scope]
            keys = ["task_order", "replicate"]
            runs_a, runs_b = a.drop_duplicates(keys), b.drop_duplicates(keys)
            summary.append({"pair": f"sonnet+{other}", "task_order": scope, "family": "run", "metric": "resources",
                            **paired(runs_a, runs_b, "resources", keys)})
            for family in ("sonnet", other):
                for mt in METRICS:
                    summary.append({"pair": f"sonnet+{other}", "task_order": scope, "family": family, "metric": mt,
                                    **paired(a[a["family"] == family], b[b["family"] == family], mt, keys)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(per_run).to_csv(OUT / "mixed_per_run.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "mixed_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Self-anchor replay: does the "use your previous myth as inspiration" line hold back copying?

Every later-round myth prompt in the September runs ends with "Write your own myth. Use
the myth you wrote in the previous round as inspiration, but adapt it in your own way."
This replays logged Sonnet 4.5 myth calls (homogeneous September no-defector runs) twice:
exactly as logged (arm ``line``) and with only that second sentence deleted (arm
``noline``). The partner's myth stays in the prompt; the agent's own previous myth stays in
its chat memory, so this tests the instruction, not removing the own myth.

Measures (fixed before the run, same definitions as analyses/linguistic_uptake.py):
  primary   word adoption excess: among the shown partner myth's words the agent had never
            used in its own earlier myths, the share the new myth uses, minus the same share
            for unseen same-family myths from the same round (8-agent: other agents in the
            run; dyads: the same seat in other runs of the cell). Effect = noline - line.
  check     own retention: share of the agent's previous-round myth's content words that the
            new myth reuses. If the line matters, this should fall without it.
Decision rule: "the line holds back borrowing" if the noline - line difference in adoption
excess has a run-clustered 95% CI above 0 (t over runs, at most 20 clusters).

Contexts: one later-round myth call per run x agent (round drawn uniformly from 2-10 with a
fixed seed): all 10 seats in each dyad cell, 25 per 8-agent cell. 2 samples per arm.

  python3 analyses/self_anchor_replay.py --plan
  python3 analyses/self_anchor_replay.py --run --workers 8
  python3 analyses/self_anchor_replay.py --analyze
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses import linguistic_corpus  # noqa: E402
from analyses.linguistic_uptake import MIN_WORDS, content_words, null_candidates, own_history  # noqa: E402
from analyses.myth_replay_probe import DEFAULT_DATA_ROOT, append, call_cost, done_keys, replay  # noqa: E402

OUT = ROOT / "data/analysis/self_anchor_replay_20261002"
FIGS = ROOT / "docs/research/self_anchor_replay_20261002"
SEED = 20261002
LINE = "Use the myth you wrote in the previous round as inspiration, but adapt it in your own way. "
ARMS = ("line", "noline")
SAMPLES = 2
PER_CELL = {2: 10, 8: 25}
EST_CALL_USD = 0.02  # Sonnet $3/$15 per 1M (analyses/myth_replay_probe.py PRICE); ~2-5k in, ~550 out incl. thinking


def corpus(data_root: Path) -> pd.DataFrame:
    myths, _ = linguistic_corpus.load(data_root)
    myths["valid"] = myths["n_words"] >= MIN_WORDS
    myths["text"] = myths["text"].fillna("")
    return myths.reset_index(drop=True)


def contexts(data_root: Path) -> list[dict]:
    rng = random.Random(SEED)
    selected = []
    specs = [s for size in (2, 8) for s in linguistic_corpus.run_list(size, data_root)]
    for size in (2, 8):
        for order in ("myth_game", "game_myth"):
            pool = []
            for spec in specs:
                if spec["size"] != size or spec["task_order"] != order:
                    continue
                d = json.loads(Path(spec["abs_path"]).read_text())
                req = d["run_metadata"].get("llm_request") or {}
                if req.get("provider") in (None, "mixed") or "sonnet" not in req.get("model", ""):
                    continue
                for agent, a in sorted(d["agents"].items()):
                    calls = {x["metadata"]["round"]: x for x in a["interaction_history"]
                             if (x.get("metadata") or {}).get("task") == "myth" and x["metadata"].get("round", 0) >= 2}
                    pool.append((spec, req, agent, calls))
            rng.shuffle(pool)
            for spec, req, agent, calls in pool[:PER_CELL[size]]:
                rnd = rng.choice(sorted(calls))
                x = calls[rnd]
                msgs = x["messages_sent"]
                last = msgs[-1]["content"]
                assert msgs[-1]["role"] == "user" and last.count(LINE) == 1, (spec["path"], agent, rnd)
                edited = [*msgs[:-1], {**msgs[-1], "content": last.replace(LINE, "")}]
                assert "Write your own myth." in edited[-1]["content"] and "how the game should be played" in edited[-1]["content"]
                selected.append({"run": spec["path"], "run_id": Path(spec["path"]).stem, "size": size, "task_order": order,
                                 "agent": agent, "round": rnd, "plan": req, "temperature": x.get("temperature", "default"),
                                 "messages": {"line": msgs, "noline": edited},
                                 "original": (x.get("response") or {}).get("content", "")})
    return selected


def jobs_for(ctxs: list[dict]) -> list[dict]:
    jobs = [{"ctx": c, "arm": arm, "sample": s, "key": f"{c['run_id']}|{c['agent']}|r{c['round']}|{arm}|s{s}"}
            for c in ctxs for arm in ARMS for s in range(SAMPLES)]
    random.Random(SEED + 1).shuffle(jobs)  # interleave arms so drift over time hits both
    return jobs


def run(data_root: Path, workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "replays.jsonl"
    done = done_keys(path)
    pending = [j for j in jobs_for(contexts(data_root)) if j["key"] not in done]

    def work(j):
        c = j["ctx"]
        res = replay({"plan": c["plan"], "temperature": c["temperature"]}, c["messages"][j["arm"]])
        rec = {"key": j["key"], "run_id": c["run_id"], "size": c["size"], "task_order": c["task_order"],
               "agent": c["agent"], "round": c["round"], "arm": j["arm"], "sample": j["sample"],
               "text": res["text"], "error": res["error"], "finish_reason": res.get("finish_reason"),
               "usage": res["usage"], "cost": call_cost("Sonnet", res["usage"] or {})}
        append(path, rec)
        return rec

    spent, errors = 0.0, 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for n, rec in enumerate(pool.map(work, pending), 1):
            spent += rec["cost"]
            errors += bool(rec["error"])
            if n % 20 == 0 or n == len(pending):
                print(f"{n}/{len(pending)} done, ${spent:.2f}, errors {errors}", flush=True)


# ----------------------------------------------------------------------------- analysis

def analyze(data_root: Path) -> None:
    myths = corpus(data_root)
    words = [content_words(t) for t in myths["text"]]
    hist = own_history(myths, words)
    cands = null_candidates(myths)
    idx = {(r, t, a): i for i, (r, t, a) in enumerate(zip(myths["run_id"], myths["round"], myths["agent"]))}
    rows = [json.loads(line) for line in (OUT / "replays.jsonl").read_text().splitlines() if line.strip()]
    rows = list({r["key"]: r for r in rows}.values())  # a retried key keeps its latest record
    for c in contexts(data_root):  # the logged myth: a sanity row, not an arm
        rows.append({"key": f"{c['run_id']}|{c['agent']}|r{c['round']}|logged", "run_id": c["run_id"], "size": c["size"],
                     "task_order": c["task_order"], "agent": c["agent"], "round": c["round"], "arm": "logged",
                     "sample": 0, "text": c["original"], "error": None})

    out = []
    for r in rows:
        i = idx[(r["run_id"], r["round"], r["agent"])]
        text = "" if r["error"] else str(r["text"])
        rec = {k: r[k] for k in ("run_id", "size", "task_order", "agent", "round", "arm", "sample", "error")}
        rec.update(n_words=len(text.split()), refused_or_empty=bool(r["error"]) or len(text.split()) < MIN_WORDS)
        if not rec["refused_or_empty"] and i in cands:
            new = content_words(text) - hist[i]
            p, nulls = cands[i][0], cands[i][1:]

            def adoption(j):
                pool = words[j] - hist[i]
                return len(new & pool) / len(pool) if pool else np.nan

            prev = idx.get((r["run_id"], r["round"] - 1, r["agent"]))
            y = content_words(text)
            rec.update(adopt_parent=adoption(p), adopt_null=np.nanmean([adoption(j) for j in nulls]),
                       own_retention=len(y & words[prev]) / len(words[prev]) if prev is not None and words[prev] else np.nan)
            rec["adopt_excess"] = rec["adopt_parent"] - rec["adopt_null"]
        out.append(rec)
    df = pd.DataFrame(out)
    FIGS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "replay_measures.csv", index=False)

    metrics = ["adopt_parent", "adopt_null", "adopt_excess", "own_retention", "n_words"]
    ok = df[~df["refused_or_empty"]]
    wide = ok.groupby(["run_id", "size", "task_order", "agent", "round", "arm"])[metrics].mean().unstack("arm")
    summary = []
    for label in ("all", "dyads", "8-agent"):
        w = wide if label == "all" else wide[wide.index.get_level_values("size") == (2 if label == "dyads" else 8)]
        for m in metrics:
            for a, b in (("noline", "line"), ("logged", "line")):
                d = (w[(m, a)] - w[(m, b)]).dropna()
                per_run = d.groupby(level="run_id").mean()
                n = len(per_run)
                mean = per_run.mean()
                half = stats.t.ppf(0.975, n - 1) * per_run.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
                summary.append({"subset": label, "metric": m, "contrast": f"{a} - {b}", "n_contexts": len(d),
                                "n_runs": n, "mean_b": w[(m, b)].mean(), "mean_a": w[(m, a)].mean(),
                                "diff": mean, "ci_lo": mean - half, "ci_hi": mean + half,
                                "runs_positive": int((per_run > 0).sum())})
    s = pd.DataFrame(summary)
    s.to_csv(FIGS / "summary.csv", index=False)
    counts = df.groupby("arm").agg(calls=("arm", "size"), refused_or_empty=("refused_or_empty", "sum"))
    counts.to_csv(FIGS / "call_counts.csv")
    pd.set_option("display.width", 220)
    print(counts)
    print(s[s["contrast"] == "noline - line"].round(4).to_string(index=False))
    print(s[(s["contrast"] == "logged - line") & (s["subset"] == "all")].round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.plan or args.run:
        ctxs = contexts(args.data_root)
        jobs = jobs_for(ctxs)
        cells = pd.Series([f"{c['size']}-agent {c['task_order']}" for c in ctxs]).value_counts().to_dict()
        print(f"contexts {len(ctxs)} {cells}; runs {len({c['run_id'] for c in ctxs})}")
        print(f"MODEL=anthropic/claude-sonnet-4.5 (logged plan, thinking on) N={len(jobs)} WORKERS={args.workers} "
              f"EST_COST=${len(jobs) * EST_CALL_USD:.2f}", flush=True)
    if args.run:
        run(args.data_root, args.workers)
    if args.analyze:
        analyze(args.data_root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Code every myth's Lévi-Strauss structure: mythemes, oppositions, mediator, transformation.

Rubric: analyses/rubrics/myth_structure_rubric.txt. Per myth the judge returns
3-6 mythemes (each in the myth's own words and as abstract roles, so that
"columns" can later be clustered without matching on character names), 1-3
binary oppositions with a category, how the central one is resolved, whether a
mediator stands between its poles, and whether the ending inverts the opening
(arc type, and the stricter role inversion that stands in for the
canonical formula).

The "Myth: [ ... ]" wrapper that rounds 2-10 carry is stripped before judging,
so the judge cannot tell round 1 from later rounds by format.

Judge settings and cache are those of analyses/myth_moral_judge.py (GLM-5.2 via
OpenRouter, temperature 0, reasoning off, cached by exact prompt; only valid
JSON is cached).

  python3 analyses/myth_structure_judge.py --dataset september_n10 --pilot      # 54 myths, stratified
  python3 analyses/myth_structure_judge.py --dataset september_n10 --preflight
  python3 analyses/myth_structure_judge.py --model deepseek/deepseek-v4-flash   # rounds 1,2,5,6,9,10 (default)
  python3 analyses/myth_structure_judge.py --dataset september_n10 --model deepseek/deepseek-v4-flash --sample 1000

Outputs in the dataset's data directory (gitignored):
  myth_structure_<tag>.csv     one row per myth: categories, resolution, mediator, transformation, status
  myth_structure_<tag>.jsonl   the full parsed structure per myth, mythemes included
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import linguistic_datasets
from myth_moral_judge import MIN_WORDS, PRICES, ROOT, Judge

RUBRIC = ROOT / "analyses/rubrics/myth_structure_rubric.txt"
CATEGORIES = ["giving_hoarding", "trust_fear", "cooperation_betrayal", "punishment_forgiveness",
              "individual_community", "scarcity_abundance", "clarity_distortion", "honesty_deception",
              "order_chaos", "life_death", "high_low", "light_dark", "human_divine", "nature_culture", "other"]
RESOLUTIONS = ["one_pole_wins", "mediated", "balanced", "unresolved"]
ARCS = ["improves", "declines", "restored", "stable", "mixed"]
MEDIATOR_TYPES = ["spirit_or_deity", "trickster", "natural_element", "object_or_artifact",
                  "law_or_ritual", "human_character", "none"]
WRAPPER = re.compile(r"^\s*Myth:\s*\[?(.*?)\]?\.?\s*$", re.S)


def strip_wrapper(text: str) -> str:
    m = WRAPPER.match(text)
    return m.group(1).strip() if m else text.strip()


def render(text: str) -> str:
    return RUBRIC.read_text().replace("{text}", strip_wrapper(text))


def _bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return True if s == "true" else False if s == "false" else None


def parse(raw: str) -> tuple[dict | None, str]:
    s = raw.strip()
    try:
        obj = json.loads(s[s.find("{"): s.rfind("}") + 1])
    except Exception:  # noqa: BLE001
        return None, "bad_json"
    if not isinstance(obj, dict):
        return None, "bad_json"
    mythemes = obj.get("mythemes")
    if not isinstance(mythemes, list) or not 1 <= len(mythemes) <= 8:
        return None, "bad_mythemes"
    mythemes = [{"text": str(m.get("text", "")).strip(), "roles": str(m.get("roles", "")).strip()}
                for m in mythemes if isinstance(m, dict)]
    if not mythemes or any(not m["roles"] for m in mythemes):
        return None, "bad_mythemes"
    opps = obj.get("oppositions")
    if not isinstance(opps, list) or not 1 <= len(opps) <= 4:
        return None, "bad_oppositions"
    clean_opps = []
    for o in opps:
        cat = str((o or {}).get("category", "")).strip().lower() if isinstance(o, dict) else ""
        if cat not in CATEGORIES:
            return None, "bad_category"
        clean_opps.append({"pole_a": str(o.get("pole_a", "")).strip(), "pole_b": str(o.get("pole_b", "")).strip(),
                           "category": cat})
    res = str(obj.get("main_resolution", "")).strip().lower()
    if res not in RESOLUTIONS:
        return None, "bad_resolution"
    med = obj.get("mediator")
    if not isinstance(med, dict):
        return None, "bad_mediator"
    mtype = str(med.get("type", "")).strip().lower()
    present = _bool(med.get("present"))
    if mtype not in MEDIATOR_TYPES or present is None:
        return None, "bad_mediator"
    if not present:
        mtype = "none"
    tr = obj.get("transformation")
    if not isinstance(tr, dict):
        return None, "bad_transformation"
    arc, inv = str(tr.get("arc", "")).strip().lower(), _bool(tr.get("role_inversion"))
    if arc not in ARCS or inv is None:
        return None, "bad_transformation"
    return {
        "mythemes": mythemes,
        "oppositions": clean_opps,
        "main_resolution": res,
        "mediator": {"present": present, "figure": med.get("figure") if present else None, "type": mtype},
        "transformation": {"opening": str(tr.get("opening", "")), "closing": str(tr.get("closing", "")),
                           "arc": arc, "role_inversion": inv},
    }, "ok"


def corpus(ds) -> pd.DataFrame:
    myths = pd.read_csv(ds.data / "myths.csv")
    myths = myths[myths["n_words"] >= MIN_WORDS]
    return myths[["run_id", "size", "mixed", "composition", "task_order", "round", "agent", "family",
                  "exposed_author", "exposed_round", "text"]].reset_index(drop=True)


def pilot_sample(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """3 myths per family x task order x round in {1, 2, 10}; single-model runs only."""
    pool = frame[(~frame["mixed"]) & frame["round"].isin([1, 2, 10])]
    return pool.groupby(["family", "task_order", "round"]).sample(n=3, random_state=seed).reset_index(drop=True)


def flat(rec: dict | None) -> dict:
    if not rec:
        return {}
    opps = rec["oppositions"]
    out = {
        "n_mythemes": len(rec["mythemes"]),
        "main_category": opps[0]["category"],
        "main_pole_a": opps[0]["pole_a"], "main_pole_b": opps[0]["pole_b"],
        "categories": "|".join(o["category"] for o in opps),
        "main_resolution": rec["main_resolution"],
        "mediator_present": rec["mediator"]["present"],
        "mediator_type": rec["mediator"]["type"],
        "mediator_figure": rec["mediator"]["figure"],
        "arc": rec["transformation"]["arc"],
        "role_inversion": rec["transformation"]["role_inversion"],
    }
    for c in CATEGORIES:
        out[f"has_{c}"] = any(o["category"] == c for o in opps)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="z-ai/glm-5.2")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dataset", choices=sorted(linguistic_datasets.DATASETS), default="september_n10")
    ap.add_argument("--pilot", type=int, nargs="?", const=0, help="stratified 54-myth pilot (optional seed)")
    ap.add_argument("--sample", type=int, help="random sample of N myths (second-judge agreement)")
    ap.add_argument("--rounds", default="1,2,5,6,9,10",
                    help="rounds to code (default: three adjacent pairs, so every coded child from round 2, 6 "
                         "and 10 has its shown round-1, 5 and 9 parent coded); 'all' for every round")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    ds = linguistic_datasets.get(args.dataset)
    frame = corpus(ds)
    tag = f"{ds.name}_{args.model.replace('/', '__')}"
    if args.pilot is None and args.rounds != "all":
        frame = frame[frame["round"].isin([int(r) for r in args.rounds.split(",")])].reset_index(drop=True)
        tag += "_r" + args.rounds.replace(",", "-")
    if args.pilot is not None:
        frame, tag = pilot_sample(frame, args.pilot), tag + f"_pilot{args.pilot}"
    elif args.sample:
        frame, tag = frame.sample(args.sample, random_state=0).reset_index(drop=True), tag + f"_sample{args.sample}"

    pin, pout = PRICES[args.model]
    n_in = sum(len(render(t)) for t in frame["text"]) / 4 + 40 * len(frame)
    est = (n_in / 1e6 * pin + 450 * len(frame) / 1e6 * pout) * 1.5
    print(f"MODEL={args.model} DATASET={ds.name} N={len(frame)} WORKERS={args.workers} "
          f"EST_COST=${est:.2f} (upper bound; cached prompts are free)")
    if args.preflight:
        return

    judge = Judge(args.model)
    prompts = [render(t) for t in frame["text"]]
    valid = lambda raw: parse(raw)[1] == "ok"  # noqa: E731
    results: list[dict | None] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge, p, valid): i for i, p in enumerate(prompts)}
        for n, fut in enumerate(as_completed(futures), 1):
            results[futures[fut]] = fut.result()
            if n % 500 == 0:
                spent = sum(r.get("cost") or 0 for r in results if r and not r.get("cached"))
                print(f"  {n}/{len(prompts)} done, ${spent:.2f} billed this session", flush=True)

    parsed = [parse(r.get("raw", "")) for r in results]
    keys = frame.drop(columns=["text"]).reset_index(drop=True)
    rows = []
    jsonl = ds.data / f"myth_structure_{tag}.jsonl"
    with jsonl.open("w") as fh:
        for i, ((rec, status), r) in enumerate(zip(parsed, results)):
            key = keys.iloc[i].to_dict()
            rows.append({**key, **flat(rec), "status": status if not r.get("error") else "error",
                         "cost": r.get("cost") or 0, "cached": bool(r.get("cached"))})
            fh.write(json.dumps({**{k: (v.item() if hasattr(v, "item") else v) for k, v in key.items()},
                                 "structure": rec, "status": status}, ensure_ascii=False) + "\n")
    out = pd.DataFrame(rows)
    path = ds.data / f"myth_structure_{tag}.csv"
    out.to_csv(path, index=False)
    bad = (out["status"] != "ok").sum()
    print(f"{len(out) - bad} ok, {bad} unparsed/errored; ${out.loc[~out['cached'], 'cost'].sum():.2f} billed "
          f"-> {path} (+ {jsonl.name})")


if __name__ == "__main__":
    main()

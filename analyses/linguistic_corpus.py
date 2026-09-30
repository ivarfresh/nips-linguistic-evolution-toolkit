#!/usr/bin/env python3
"""Myth and decision tables for the September linguistic analysis.

One loader shared by every linguistic analysis of the September informed
negative-only runs: the homogeneous controls (negative_only_crossmodel_
reasoning_rerun_20260909) and the mixed-model dyads and populations.

The run list is taken from the validated tables behind the mixed-model figures,
not from a directory glob. The 20260909 folders also hold defector variants in
sibling subfolders; the figure tables list only the no-defector finals
(99 dyads, 135 populations), so reading their `path` column keeps the inclusion
rule identical to Figures 7 and 8.

Outputs (data/analysis/linguistic_20260923/, gitignored):
  myths.csv      one row per myth: run, size, composition, task order, round,
                 author, author family, text, and the myth the author was shown
                 before writing it (from the run's own `myth_exposures` record)
  decisions.csv  one row per agent per round: role, partner, partner family,
                 amount sent / return proportion, and a 0-1 cooperation score
                 (sent / 5 for senders, returned / received for receivers)

Run from the repo root (or pass --data-root to a checkout that holds data/json):
  python3 analyses/linguistic_corpus.py

Frontier corpus (--dataset frontier, or LINGUISTIC_DATASET=frontier): the main
frontier set, Opus 5 / Gemini 3.1 Pro / GPT-5.6 Sol, homogeneous runs of
2026-09-18 and mixed runs of 2026-09-28. The run list is the receipt-audited
list of scripts/analyze_frontier_main_mixed_20260928.py (which already leaves
out Opus 5.5, Sol 6, the Sol-none smoke run and quarantined finals); game-only
runs are dropped. Outputs go to data/analysis/linguistic_frontier_20260930/.
  python3 analyses/linguistic_corpus.py --dataset frontier
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from analyses import linguistic_datasets
except ImportError:  # run as a script from analyses/
    import linguistic_datasets

ROOT = Path(__file__).resolve().parents[1]
RUN_TABLES = {
    2: ROOT / "docs/figures/mixed_model_dyads_20260917/decisions.csv",
    8: ROOT / "docs/figures/mixed_model_populations_20260918/games.csv",
}
OUT = ROOT / "data/analysis/linguistic_20260923"  # September default; see linguistic_datasets
ENDOWMENT = 5.0
FAMILY_OF = {"claude-sonnet-4.5": "Sonnet", "gpt-5-nano": "GPT", "gemini-3.7-flash": "Gemini"}
FAMILY_ORDER = ["Sonnet", "Gemini", "GPT"]  # the figure labels: Sonnet+GPT, Sonnet+Gemini, Gemini+GPT


FRONTIER_EXPECTED = {(2, False): 30, (8, False): 30, (2, True): 36, (8, True): 10}  # myth-bearing runs


def family(model: str) -> str:
    if DS.name != "september":
        return DS.family(model)
    for key, fam in FAMILY_OF.items():
        if key in model:
            return fam
    raise ValueError(f"unknown model {model!r}")


def agent_families(run: dict) -> dict[str, str]:
    meta = run["run_metadata"]
    per_agent = (meta.get("llm_request") or {}).get("agents")
    if per_agent:
        return {a: family(v["model"]) for a, v in per_agent.items()}
    return {a: family(meta["model"]) for a in run["agents"]}


def composition_label(families: dict[str, str]) -> str:
    counts = pd.Series(list(families.values())).value_counts()
    if DS.name != "september":  # deterministic order for tied counts (2/3/3); September kept as built
        order = list(DS.families)
        if len(counts) == 1:
            return f"{len(families)} {counts.index[0]}" if len(families) > 2 else f"{counts.index[0]}+{counts.index[0]}"
        if len(families) == 2:
            return "+".join(sorted(families.values(), key=order.index))
        return " + ".join(f"{n} {f}" for f, n in sorted(counts.items(), key=lambda kv: (kv[1], order.index(kv[0]))))
    if len(counts) == 1:
        return f"{len(families)} {counts.index[0]}" if len(families) > 2 else f"{counts.index[0]}+{counts.index[0]}"
    if len(families) == 2:
        return "+".join(sorted(families.values(), key=FAMILY_ORDER.index))
    return " + ".join(f"{n} {f}" for f, n in counts.sort_values().items())


def run_list(size: int, data_root: Path) -> list[dict]:
    table = pd.read_csv(RUN_TABLES[size])
    runs = table.drop_duplicates("path")[["path", "task_order", "replicate_id"]]
    runs = runs[runs["task_order"] != "game"]  # game-only runs have no myths
    return [
        {"path": p, "abs_path": data_root / p, "task_order": t, "replicate_id": int(r), "size": size}
        for p, t, r in runs.itertuples(index=False)
    ]


def frontier_run_list(data_root: Path) -> list[dict]:
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.analyze_frontier_main_mixed_20260928 import load as frontier_load
    df, _, _ = frontier_load()  # receipt-audited: sha256 of every final checked
    runs = df.drop_duplicates("path")
    runs = runs[runs["task_order"] != "game"]  # game-only runs have no myths
    out = []
    for p, t, r, n, setting in runs[["path", "task_order", "replicate_id", "num_agents", "setting"]].itertuples(index=False):
        rel = Path("data/json" + p.split("/data/json", 1)[1])  # loader resolves the data/json symlinks
        out.append({"path": str(rel), "abs_path": data_root / rel, "task_order": t, "replicate_id": int(r),
                    "size": int(n), "mixed": setting == "mixed"})
    got = pd.Series([(s["size"], s["mixed"]) for s in out]).value_counts().to_dict()
    if got != FRONTIER_EXPECTED:
        raise RuntimeError(f"frontier myth runs {got}, expected {FRONTIER_EXPECTED}")
    return out


def run_specs(data_root: Path) -> list[dict]:
    if DS.name == "frontier":
        return frontier_run_list(data_root)
    return [spec for size in (2, 8) for spec in run_list(size, data_root)]


def load(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    myth_rows, decision_rows = [], []
    for spec in run_specs(data_root):
        size = spec["size"]
        run = json.loads(Path(spec["abs_path"]).read_text())
        fams = agent_families(run)
        comp = composition_label(fams)
        mixed = len(set(fams.values())) > 1
        run_id = Path(spec["path"]).stem
        base = {"run_id": run_id, "path": spec["path"], "size": size, "composition": comp, "mixed": mixed,
                "task_order": spec["task_order"], "replicate_id": spec["replicate_id"]}
        for entry in run["conversation_history"]:
            rnd = int(entry["round"])
            partner = {}
            for p in entry["pairings"]:
                partner[p["investor"]] = p["trustee"]
                partner[p["trustee"]] = p["investor"]
            exposures = entry.get("myth_exposures") or {}
            for agent, text in (entry.get("myths") or {}).items():
                exp = exposures.get(agent) or {}
                src = exp.get("original_author_id")
                myth_rows.append({**base, "round": rnd, "agent": agent, "family": fams[agent],
                                  "partner_this_round": partner.get(agent),
                                  "exposed_author": src, "exposed_family": fams.get(src) if src else None,
                                  "exposed_round": exp.get("source_round"),
                                  "text": text, "n_words": len(str(text).split())})
            for d in entry.get("dyads") or []:
                inv, tru = d["investor"], d["trustee"]
                sent, received, returned = float(d["sent"]), float(d["received"]), float(d["returned"])
                ret_prop = returned / received if received > 0 else float("nan")
                for agent, role, other in ((inv, "investor", tru), (tru, "trustee", inv)):
                    decision_rows.append({
                        **base, "round": rnd, "agent": agent, "family": fams[agent], "role": role,
                        "partner": other, "partner_family": fams[other],
                        "sent": sent, "received": received, "returned": returned,
                        "return_proportion": ret_prop,
                        "coop": sent / ENDOWMENT if role == "investor" else ret_prop,
                    })
    return pd.DataFrame(myth_rows), pd.DataFrame(decision_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=ROOT, help="checkout whose data/json holds the runs")
    ap.add_argument("--dataset", choices=sorted(linguistic_datasets.DATASETS),
                    help=f"corpus to build (default: ${linguistic_datasets.ENV} or september)")
    ap.add_argument("--out", type=Path, help="output directory (default: the dataset's data directory)")
    args = ap.parse_args()
    global DS
    DS = linguistic_datasets.get(args.dataset)
    out = args.out or DS.data
    if DS.name != "september" and out.resolve() == linguistic_datasets.DATASETS["september"].data.resolve():
        raise SystemExit("refusing to write a non-September corpus into the September data directory")
    print(f"dataset {DS.name} -> {out}")
    myths, decisions = load(args.data_root)
    if DS.name != "september":
        check_frontier(myths, decisions)
    out.mkdir(parents=True, exist_ok=True)
    myths.to_csv(out / "myths.csv", index=False)
    decisions.to_csv(out / "decisions.csv", index=False)
    print(f"myths: {len(myths)} from {myths.run_id.nunique()} runs -> {out / 'myths.csv'}")
    print(myths.groupby(["size", "mixed", "task_order"]).run_id.nunique().to_string())
    print(f"decisions: {len(decisions)} rows -> {out / 'decisions.csv'}")
    missing = myths[(myths["round"] > 1) & myths["exposed_author"].isna()]
    print(f"round>1 myths without an exposure record: {len(missing)}")


def check_frontier(myths: pd.DataFrame, decisions: pd.DataFrame) -> None:
    """Hard checks the September build only printed: unique run ids, and a resolvable shown myth."""
    if myths.drop_duplicates("path").run_id.duplicated().any():
        raise RuntimeError("run_id (file stem) is not unique across runs")
    later = myths[myths["round"] > 1]
    if later["exposed_author"].isna().any():
        raise RuntimeError(f"{later['exposed_author'].isna().sum()} round>1 myths without an exposure record")
    keys = set(zip(myths.run_id, myths["round"], myths.agent))
    missing = [k for k in zip(later.run_id, later.exposed_round, later.exposed_author) if k not in keys]
    if missing:
        raise RuntimeError(f"{len(missing)} shown myths not found in the corpus, e.g. {missing[:3]}")
    if myths.family.isin(["Sonnet", "Gemini", "GPT"]).any() or decisions.family.isin(["Sonnet", "Gemini", "GPT"]).any():
        raise RuntimeError("September family names in the frontier corpus")


DS = linguistic_datasets.get()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Agreement between the two judges on the myth-structure coding (no API calls).

The corpus was coded in full by DeepSeek-V4-Flash. GLM-5.2 coded a subset under the same
committed rubric (pilots and a partial run). This script finds every myth whose exact prompt
has a valid cached answer from both judges and reports, per field, the share on which they
agree. It only reads data/judge_cache/ (gitignored), so it needs the cache of the machine the
judges ran on; a myth missing from either cache is skipped, never sent to a model.

  python3 analyses/myth_structure_judge_agreement.py            # september_n10
  python3 analyses/myth_structure_judge_agreement.py --dataset frontier

Writes docs/figures/myth_structure_20261010/judge_agreement_<dataset>.csv. Afterwards rewrite
the folder's provenance.json: python3 analyses/myth_structure_analysis.py --provenance-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analyses"))
from analyses import linguistic_datasets  # noqa: E402
from analyses.myth_moral_judge import CACHE, SYSTEM  # noqa: E402
from analyses.myth_structure_judge import corpus, parse, render  # noqa: E402

JUDGES = ("deepseek/deepseek-v4-flash", "z-ai/glm-5.2")
OUT = ROOT / "docs/figures/myth_structure_20261010"
FIELDS = {  # name -> value taken from one parsed structure
    "central opposition (category)": lambda s: s["oppositions"][0]["category"],
    "resolution": lambda s: s["main_resolution"],
    "mediator present": lambda s: s["mediator"]["present"],
    "arc": lambda s: s["transformation"]["arc"],
    "role inversion": lambda s: s["transformation"]["role_inversion"],
}


def cached(model: str, prompt: str) -> dict | None:
    """The valid cached structure for this exact prompt, as analyses/myth_moral_judge.Judge keys it."""
    h = hashlib.sha256(f"{model}|T=0|{SYSTEM}|{prompt}".encode()).hexdigest()
    path = CACHE / model.replace("/", "__") / h[:2] / f"{h}.json"
    if not path.exists():
        return None
    structure, status = parse(json.loads(path.read_text()).get("raw", ""))
    return structure if status == "ok" else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=sorted(linguistic_datasets.DATASETS), default="september_n10")
    args = ap.parse_args()
    ds = linguistic_datasets.get(args.dataset)
    frame = corpus(ds)
    pairs = []
    for row in frame.itertuples():
        prompt = render(row.text)
        both = [cached(model, prompt) for model in JUDGES]
        if all(both):
            pairs.append((row.family, *both))
    if not pairs:
        raise SystemExit(f"{ds.name}: no myth has a cached answer from both judges")
    rows = []
    for name, value in FIELDS.items():
        same = pd.Series([value(a) == value(b) for _, a, b in pairs])
        fam = pd.Series([f for f, _, _ in pairs])
        rows.append({"field": name, "n_myths": len(pairs), "agreement": round(same.mean(), 3),
                     **{f"agreement_{f}": round(same[fam == f].mean(), 3) for f in ds.families if (fam == f).any()}})
    table = pd.DataFrame(rows)
    path = OUT / f"judge_agreement_{ds.name}.csv"
    table.to_csv(path, index=False)
    counts = pd.Series([f for f, _, _ in pairs]).value_counts().to_dict()
    print(f"{ds.name}: {len(pairs)} of {len(frame)} myths coded by both judges {counts}")
    print(table.to_string(index=False))
    print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

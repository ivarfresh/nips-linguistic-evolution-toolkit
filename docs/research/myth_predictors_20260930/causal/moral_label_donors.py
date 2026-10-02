#!/usr/bin/env python3
"""Label the 30 transplant donor texts with the same moral rubric the other lenses use.

Reuses Judge/render/parse from analyses/myth_moral_judge.py (Arabella's 3-moral rubric and the
one-sentence moral summary), GLM-5.2 and DeepSeek V4 Flash via OpenRouter, temperature 0.
The cache is redirected into this folder so nothing in the repo is written.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WT = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-spread-viz")
sys.path.insert(0, str(WT / "analyses"))
import myth_moral_judge as mmj  # noqa: E402

mmj.CACHE = HERE / "judge_cache"
from dotenv import load_dotenv  # noqa: E402
load_dotenv("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-linguistic-evolution-toolkit/.env")


def main() -> None:
    don = pd.read_csv(HERE / "donors.csv")
    don = don[don["seed_type"] != "baseline"].reset_index(drop=True)
    frame = don[["seed_type", "rep", "text"]]
    out = don[["seed_type", "rep"]].copy()
    total = 0.0
    for model in ("z-ai/glm-5.2", "deepseek/deepseek-v4-flash"):
        est = mmj.preflight(model, ["label", "summary"], frame)
        print(f"MODEL={model} TASKS=label,summary N={len(frame)} WORKERS=8 EST_COST=${est:.3f}")
        judge = mmj.Judge(model)
        tag = model.split("/")[0].split("-")[0]
        for task in ("label", "summary"):
            res = mmj.run_task(judge, task, frame, workers=8)
            out[f"{task}_{tag}"] = res[task].values
            total += res[f"{task}_cost"].sum()
    out.to_csv(HERE / "donor_moral_labels.csv", index=False)
    print(out.to_string())
    print(f"billed ${total:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""May 2026 prompt-arm experiments: the myth-writing INSTRUCTION was manipulated (control / game-directive /
normative), which changes self-written myth content (manipulation check in
data/analysis/myth_game_manipulation_check). Does it change play? 2-agent Sonnet 4.5, informed negative U(0,2),
OpenRouter era (thinking on). Pre-September protocol. Writes prompt_arm_outcomes.csv.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-linguistic-evolution-toolkit")
ROOTS = [MAIN / "data/json/noise_experiments/neutral_2026_05_27/myth_causal_neutral_noise_negative2_informed_claude_r10_n10",
         MAIN / "data/json/noise_experiments/neutral_2026_05_28/myth_causal_normative_noise_negative2_informed_claude_r10_n5"]


def outcome(p: Path) -> dict:
    run = json.loads(p.read_text())
    h = run["conversation_history"]
    sends = []
    for e in h:
        for d in e.get("dyads") or []:
            sends.append(float(d["sent"]))
        if not e.get("dyads") and e.get("sent") is not None:
            sends.append(float(e["sent"]))
    bal = h[-1].get("balances") or run.get("game_data", {}).get("balances") or {}
    return {"send_mean": np.mean(sends) if sends else np.nan, "joint": float(sum(bal.values())) if bal else np.nan,
            "n_rounds": len(h)}


def main() -> None:
    rows = []
    for root in ROOTS:
        for p in root.rglob("*.json"):
            if re.search(r"checkpoint|results|error", p.name):
                continue
            m = re.search(r"_(myth_control|myth_game_directive|myth_game_normative)_", p.name)
            order = p.parent.parent.name
            arm = m.group(1) if m else ("game_only" if order == "game" else "unknown")
            rows.append({"set": root.name, "arm": arm, "task_order": order, "file": p.name, **outcome(p)})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "prompt_arm_runs.csv", index=False)
    s = df.groupby(["set", "arm"]).agg(n=("joint", "size"), joint_mean=("joint", "mean"), joint_sd=("joint", "std"),
                                       send_mean=("send_mean", "mean"), send_sd=("send_mean", "std")).round(2)
    s.to_csv(HERE / "prompt_arm_outcomes.csv")
    print(s.to_string())


if __name__ == "__main__":
    main()

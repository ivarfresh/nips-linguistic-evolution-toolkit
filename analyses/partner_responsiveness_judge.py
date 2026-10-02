"""Code what Sonnet 4.5's game rationales rest on: the partner, the myth, or both.

For each rationale an LLM judge records whether the decision is justified by the
co-player's moves, whether it invokes a myth, and, when it does, whether the
myth's lesson is conditional on the partner (reciprocate, match, respond to
betrayal) or unconditional (give / trust regardless). The judge sees only the
rationale and the agent's role, not the task order.

Sample (seeded): every Sonnet 4.5 dyad decision with prose in the figure2 and
random-defection sets, plus 300 per task order from the eight-agent figure2 runs.

Usage (from repo root; cached, so reruns are free):
  python analyses/partner_responsiveness_judge.py [--judge-model M] [--workers N]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from _llm_judge import judge, stats

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "figures" / "partner_responsiveness_20260930"
PER_ORDER_8AGENT = 300

SYSTEM = """You code the written reasoning of an AI agent playing a repeated trust game \
(the sender chooses how much of $5 to send; it is tripled; the receiver chooses how much to \
return). In some sessions the agents also write short myths. Read the reasoning and answer \
strictly in JSON with these keys:

"partner_basis": true if the amount chosen is justified, even partly, by what the co-player \
(or co-players) did or sent/returned in the game, including this round's amount received. \
false otherwise.
"myth_basis": true if the reasoning invokes a myth, story, tale or its moral as a reason for \
the amount. false otherwise.
"myth_rule": the lesson drawn from the myth. "conditional" if it says to respond to what the \
other does (reciprocate, match, mirror, reward trust, protect yourself after betrayal); \
"unconditional" if it says to give, trust or be generous regardless of what the other does \
(keep giving despite losses, forgive always, the giver goes first); "vague" if a myth is \
invoked without a behavioural rule; "none" if no myth is invoked.
"primary_driver": which consideration mainly sets the amount: "partner", "myth", "both", or \
"other" (own fixed rule, payoff arithmetic, fairness norm with no reference to either).
Return only the JSON object."""


def sample(dec):
    s = dec[(dec.model == "Sonnet 4.5") & (dec.own_source == "llm")].copy()
    s = s[s.text.fillna("").str.split().str.len() > 8]
    dyads = s[(s.num_agents == 2) & s.run_set.isin(["figure2", "random"])]
    pops = s[(s.num_agents == 8) & (s.run_set == "figure2")]
    pops = pops.groupby("task_order").sample(n=PER_ORDER_8AGENT, random_state=20260930)
    return pd.concat([dyads, pops])


def code_one(row, model):
    role = "SENDER" if row.role == "investor" else "RECEIVER"
    user = f"Role this round: {role}\n\nReasoning:\n{row.text}"
    return judge(SYSTEM, user, model=model, temperature=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="anthropic/claude-sonnet-4.5")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dec = pd.read_csv(OUT_DIR / "decisions.csv", low_memory=False)
    s = sample(dec).reset_index(drop=True)
    print(f"MODEL={args.judge_model} N={len(s)} WORKERS={args.workers} "
          f"EST_COST=${len(s) * (900 * 3 + 80 * 15) / 1e6:.2f}", flush=True)
    results = [None] * len(s)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(code_one, row, args.judge_model): i for i, row in s.iterrows()}
        for k, f in enumerate(as_completed(futs), 1):
            results[futs[f]] = f.result()
            if k % 200 == 0:
                print(f"  {k}/{len(s)}", flush=True)
    coded = pd.DataFrame(results)
    out = pd.concat([s.drop(columns=["text"]), coded], axis=1)
    out.to_csv(OUT_DIR / "sonnet_rationale_judge.csv", index=False)
    stats().print_summary()
    print(f"-> {(OUT_DIR / 'sonnet_rationale_judge.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""R5 lens, step 1: one row per transplant run, with the donor text's features.

Sources (read-only):
  - slide678 8-agent rerun and dyad rerun plan.json + repNN.json finals (main repo data/json)
  - the historical Phase 3/5/6 final each rerun combo points at (same donor text, old apparatus)
  - GLM-5.2 donor rule / amount-check tables (shared linguistic data dir)
Writes runs.csv and donors.csv next to this script.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-linguistic-evolution-toolkit")
LING = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-spread-viz/data/analysis/linguistic_20260923")
RERUNS = {8: MAIN / "data/json/noise_experiments/slide678_rerun_20260916",
          2: MAIN / "data/json/noise_experiments/slide678_dyad_rerun_20260917"}
MIDPOINT = {"all": 5.0, "most": 4.25, "moderate": 2.75, "little": 1.25, "none": 0.0}  # myth_rules_analysis.py:56
WRITER = {"s_start": "Sonnet", "s_end_plus": "Sonnet", "s_end_minus": "Sonnet", "s_end_plus_gemini": "Gemini",
          "s_end_plus_gpt": "GPT", "s_filler": "Wikipedia", "baseline": "none"}

# judge-free themes, verbatim from myth-rules worktree analyses/myth_rules_analysis.py:295
KEYWORDS = {
    "consistency": r"\bconsisten(?:t|cy|tly)\b|\bsteady\b|\bsteadiness\b|\breliab",
    "measured": r"\bmeasured\b|\bprudent|\bprudence|within (?:one's|their|his|her|your|its) means|\brestraint\b|\bmoderat",
    "uncertainty": r"\buncertain|\bnois|\bstatic\b|\bfog\b|\bmist\b|\bturbulen|\bdistort|\bmisheard|\bwhisper",
}
# cooperative / uncooperative word sets, verbatim from analyses/cooperativity_analysis.py:94-112
COOP = {'together', 'shared', 'mutual', 'partnership', 'community', 'collaboration', 'unity', 'collective', 'bond',
        'alliance', 'cooperation', 'cooperate', 'collaborative', 'relationship', 'connection', 'reciprocity', 'trust',
        'friendship', 'companion', 'ally', 'partner', 'kindness', 'generosity', 'compassion', 'solidarity', 'give',
        'giving', 'gave', 'given', 'offered', 'offer', 'returned', 'return', 'share', 'generous', 'gift', 'bestow',
        'granted'}
UNCOOP = {'alone', 'solitary', 'independent', 'self', 'individual', 'isolated', 'separate', 'isolation', 'lonely',
          'solo', 'betrayal', 'division', 'conflict', 'rivalry', 'enemy', 'hostility', 'antagonism', 'opposition',
          'discord', 'took', 'taken', 'take', 'kept', 'keep', 'withheld', 'withhold', 'refused', 'refuse', 'denied',
          'deny', 'hoarded', 'hoard'}
NUM = r"\b(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|half|all|nothing)\b"
FIVE = r"\b(?:5|five|all (?:five|of it|you have|their|his|her))\b|\bfull (?:amount|endowment|five)\b|\beverything\b"
GAME = r"\bsend|\bsent\b|\breturn|\btripl|\bthree ?fold|\bmultipl|\bendow|\bmeasure"


def text_features(t: str) -> dict:
    low = t.lower()
    words = re.findall(r"[a-z']+", low)
    n = max(len(words), 1)
    f = {"n_words": len(words),
         "coop_pct": 100 * sum(w in COOP for w in words) / n,
         "uncoop_pct": 100 * sum(w in UNCOOP for w in words) / n,
         "num_per100": 100 * len(re.findall(NUM, low)) / n,
         "digit_per100": 100 * len(re.findall(r"\b\d+(?:\.\d+)?\b", low)) / n,
         "five_mentions": len(re.findall(FIVE, low)),
         "game_terms_per100": 100 * len(re.findall(GAME, low)) / n}
    for k, pat in KEYWORDS.items():
        f[f"kw_{k}"] = len(re.findall(pat, low))
    f["coop_minus_uncoop"] = f["coop_pct"] - f["uncoop_pct"]
    return f


def run_outcomes(path: Path) -> dict:
    """Sends are decisions (dyads[*].sent, as myth_rules_analysis.py reads them). Return share is taken
    against what the trustee was TOLD it received (uninformed -$5 noise shrinks the told amount)."""
    run = json.loads(path.read_text())
    hist = run["conversation_history"]
    sends, rets_told, rets_true, r1 = [], [], [], []
    for e in hist:
        for d in e.get("dyads") or []:
            s = float(d["sent"])
            sends.append(s)
            told = d.get("received_communicated")
            if told is None:  # older finals: rebuild from the sender's communicated amount
                told = 3 * float(e["actions"][d["investor"]].get("communicated", s))
            if told and told > 0:
                rets_told.append(min(float(d["returned"]) / float(told), 1.0))
            if s > 0:
                rets_true.append(float(d["returned"]) / (3 * s))
            if e["round"] == 1:
                r1.append(s)
    bal = hist[-1]["balances"]
    n_agents = len(bal)
    joint = float(sum(bal.values()))
    ceiling = 15.0 * len(hist) * n_agents / 2
    return {"n_rounds": len(hist), "n_agents": n_agents, "joint": joint, "joint_frac": joint / ceiling,
            "send_mean": float(np.mean(sends)), "send_r1": float(np.mean(r1)),
            "return_share_told": float(np.mean(rets_told)) if rets_told else np.nan,
            "return_share_true": float(np.mean(rets_true)) if rets_true else np.nan,
            "full_send_share": float(np.mean(np.array(sends) == 5.0))}


def host_r1_citations(path: Path) -> dict:
    """Round-1 sender responses: do hosts cite the myth, and quote a number from it?"""
    run = json.loads(path.read_text())
    e = run["conversation_history"][0]
    inv = [d["investor"] for d in (e.get("dyads") or [])]
    resp = e.get("game_responses") or {}
    txt = [(resp.get(a) or {}).get("content") or "" for a in inv]
    return {"r1_cite_myth": float(np.mean([bool(re.search(r"\bmyth", t, re.I)) for t in txt])) if txt else np.nan}


def main() -> None:
    rows = []
    donor_text = {}
    for size, root in RERUNS.items():
        plan = json.loads((root / "plan.json").read_text())
        for c in plan["combos"]:
            st, rep = c["seed_type"], int(c["rep"])
            key = (st, rep)
            txt = c.get("seed_text") or ""
            if key in donor_text and donor_text[key]["text"] != txt:
                raise SystemExit(f"donor text differs between reruns for {key}")
            meta = c.get("seed_meta") or {}
            donor_text[key] = {"seed_type": st, "rep": rep, "text": txt, "source_run": meta.get("source_run"),
                               "source_agent": meta.get("agent_id"), "source_round": meta.get("round"),
                               "joint_at_source": meta.get("joint_at_source"),
                               "historical_final": c.get("historical_final")}
            final = root / st / f"rep{rep:02d}.json"
            rows.append({"apparatus": f"rerun_{size}", "size": size, "seed_type": st, "rep": rep,
                         "path": str(final.relative_to(MAIN)), **run_outcomes(final), **host_r1_citations(final)})
    # historical finals (8-agent Phase 3/5/6, old apparatus); one per donor
    for key, d in donor_text.items():
        hp = MAIN / d["historical_final"]
        rows.append({"apparatus": "historical_8", "size": 8, "seed_type": key[0], "rep": key[1],
                     "path": d["historical_final"], **run_outcomes(hp), **host_r1_citations(hp)})
    runs = pd.DataFrame(rows)

    # donor features: GLM rule pass (two independent passes: the size-8 and size-2 rows judged the same text)
    rules = pd.read_csv(LING / "myth_rules_donors_z-ai__glm-5.2.csv")
    chk = pd.read_csv(LING / "myth_amount_check_donors_z-ai__glm-5.2.csv")[["size", "seed_type", "rep", "amount_status"]]
    rules = rules.merge(chk, on=["size", "seed_type", "rep"], how="left")
    rules["prescribed"] = rules["send_amount"].where(rules["send_amount"].notna(), rules["send_rule"].map(MIDPOINT))
    cols = ["send_rule", "return_rule", "after_letdown", "send_amount", "test_first", "consistency",
            "noise_mentioned", "amount_status", "prescribed"]
    r8 = rules[rules["size"] == 8].set_index(["seed_type", "rep"])[cols]
    r2 = rules[rules["size"] == 2].set_index(["seed_type", "rep"])[cols].add_suffix("_pass2")
    don = pd.DataFrame(donor_text.values()).set_index(["seed_type", "rep"])
    don = don.join(r8).join(r2)
    feats = pd.DataFrame([text_features(t) if t else {} for t in don["text"]], index=don.index)
    don = don.join(feats).reset_index()
    don["writer"] = don["seed_type"].map(WRITER)
    don.loc[don["seed_type"] == "baseline", "text"] = ""
    # endorsed-only prescription: narrated named amount -> fall back to band midpoint
    don["prescribed_endorsed"] = np.where(don["amount_status"] == "narrated", don["send_rule"].map(MIDPOINT),
                                          don["prescribed"])
    don.to_csv(HERE / "donors.csv", index=False)
    runs.to_csv(HERE / "runs.csv", index=False)
    print(runs.groupby(["apparatus", "seed_type"])[["joint", "send_mean", "send_r1", "return_share_told"]].mean().round(2))
    seeded = don[don["seed_type"] != "baseline"]
    agree = []
    for c in ["send_rule", "return_rule", "after_letdown", "test_first", "consistency", "noise_mentioned",
              "prescribed", "amount_status"]:
        a, b = seeded[c].astype(object).fillna("NA").astype(str), seeded[f"{c}_pass2"].astype(object).fillna("NA").astype(str)
        both = seeded[c].notna() & seeded[f"{c}_pass2"].notna()
        agree.append({"field": c, "n_texts": len(seeded), "agree_all": (a == b).mean(),
                      "n_both_nonmissing": int(both.sum()), "agree_both_nonmissing": (a[both] == b[both]).mean()})
    agree = pd.DataFrame(agree).round(3)
    agree.to_csv(HERE / "judge_test_retest.csv", index=False)
    print("GLM pass-1 vs pass-2 on the same 30 seeded texts:\n", agree.to_string())


if __name__ == "__main__":
    main()

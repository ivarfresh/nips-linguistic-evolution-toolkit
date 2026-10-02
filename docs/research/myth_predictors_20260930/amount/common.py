"""Shared loaders for the send-amount lens (H2). Read-only on the shared data dir."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/linguistic-mixed/data/analysis/linguistic_20260923")
OUT = Path(__file__).resolve().parent
MIDPOINT = {"all": 5.0, "most": 4.25, "moderate": 2.75, "little": 1.25, "none": 0.0}
# return_rule -> share of the tripled amount; match_partner / unspecified -> missing
RETURN_SHARE = {"more_than_half": 0.6, "half": 0.5, "at_least_sent": 1 / 3, "little": 0.15}

# ------------------------------------------------------------------ regex extractor
NUM = {"zero": 0, "none": 0, "one": 1, "a single": 1, "two": 2, "three": 3, "four": 4, "five": 5,
       "half": 2.5, "two and a half": 2.5, "one and a half": 1.5, "three and a half": 3.5,
       "four and a half": 4.5}
NUMRE = r"(two and a half|one and a half|three and a half|four and a half|zero|one|a single|two|three|four|five|[0-5](?:\.\d+)?)"
UNIT = r"(?:[a-z-]+\s){0,2}?(?:coins?|seeds?|measures?|embers?|sparks?|stones?|tokens?|drops?|sacks?|crystals?|gems?|lights?|grains?|pieces?|portions?|gifts?|stars?|dollars?|units?|parts?|shares?|jewels?|pearls?|loaves|breads?|fish|flames?|petals?|beads?|feathers?|gold|silver|dollars)"
FIVE = r"(?:five|5|\$5)"
SEND_V = r"(?:send|sends|sent|sending|give|gives|gave|giving|given|cast|casts|casting|pour|pours|poured|pouring|offer|offers|offered|offering|release|releases|released|releasing|place|places|placed|entrust|entrusts|entrusted|toss|tossed|throw|threw|plant|planted|sow|sowed|sown|risk|risked|share|shared|send forth|let go of|hand|handed|pass|passed|lay|laid|drop|dropped|float|floated)"

PAT_OF_FIVE = re.compile(rf"\b{NUMRE}\s+(?:{UNIT}\s+)?(?:out\s+)?of\s+(?:the\s+|her\s+|his\s+|their\s+|its\s+|my\s+|your\s+|our\s+)?{FIVE}\b", re.I)
PAT_ALL_FIVE = re.compile(rf"\b(?:all|every one of(?: the| her| his| their)?|the full|the whole|the entire|entire|full|whole)\s+(?:{FIVE})\b", re.I)
PAT_VERB_NUM = re.compile(rf"\b{SEND_V}\s+(?:only\s+|just\s+|but\s+|a mere\s+|nearly\s+|almost\s+)?{NUMRE}\s+{UNIT}", re.I)
PAT_DOLLAR = re.compile(rf"\b{SEND_V}\s+(?:only\s+|just\s+)?\$([0-5](?:\.\d+)?)", re.I)
# abstract phrasings
PAT_ALL_ABS = re.compile(r"\b(gave|give|gives|giving|send|sent|sends|sending|pour(?:ed|s|ing)?|cast(?:s|ing)?|offer(?:ed|s|ing)?|release[ds]?|releasing)\s+(?:(?:it|them)\s+)?(everything|all (?:she|he|they|it|you|we) (?:had|has|have|held|holds|hold)|(?:her|his|their|its|your|the) (?:whole|entire|full) [a-z]+|without (?:holding back|reserve|hesitation)|fully|freely and fully|the full measure|all of (?:it|them))", re.I)
PAT_NO_HOLD = re.compile(r"\b(held nothing back|holding nothing back|hold nothing back|without holding (?:anything )?back|without reserve|gave everything|give everything|gives everything|everything (?:she|he|they) (?:had|held)|all (?:she|he|they) (?:had|held|possessed))", re.I)
PAT_PART = re.compile(r"\b(held (?:some|a little|a portion|part) back|hold (?:some|a little|a portion|part) back|kept (?:some|a portion|a little|part|a share) (?:back|for)|keep(?:ing)? (?:some|a portion|a little|part) (?:back|in reserve)|not all|a portion|part of (?:her|his|their|its|the)|a measured|a modest|a careful|a small|a share of|some of (?:her|his|their|its|the))", re.I)
PAT_NOTHING = re.compile(r"\b(sent nothing|send nothing|gave nothing|give nothing|kept (?:everything|all five|all of it)|keep (?:everything|all five)|hoarded (?:all|everything)|withheld (?:all|everything))", re.I)


def _val(tok: str) -> float:
    t = tok.lower()
    return float(NUM[t]) if t in NUM else float(t)


def regex_amount(text: str) -> dict:
    """Sender amount a myth names or implies, out of 5, from surface patterns.

    concrete: mean of explicit numbers tied to sending ("three of the five", "all five",
    "sent four coins", "$3"). abstract: 'all' (gave everything / held nothing back),
    'part' (held some back / a portion / measured), 'none' (sent nothing), or ''.
    regex_amount = concrete if found, else 5 / 2.75 / 0 for all / part / none.
    """
    if not isinstance(text, str):
        return {"rx_concrete": np.nan, "rx_abstract": "", "rx_amount": np.nan, "rx_n": 0}
    vals = []
    for m in PAT_OF_FIVE.finditer(text):
        vals.append(_val(m.group(1)))
    vals += [5.0] * len(PAT_ALL_FIVE.findall(text))
    for m in PAT_VERB_NUM.finditer(text):
        v = _val(m.group(1))
        if v <= 5:
            vals.append(v)
    for m in PAT_DOLLAR.finditer(text):
        vals.append(float(m.group(1)))
    concrete = float(np.mean(vals)) if vals else np.nan
    n_all = len(PAT_ALL_ABS.findall(text)) + len(PAT_NO_HOLD.findall(text))
    n_part = len(PAT_PART.findall(text))
    n_none = len(PAT_NOTHING.findall(text))
    abstract = ""
    if n_all or n_part or n_none:
        abstract = max([("all", n_all), ("part", n_part), ("none", n_none)], key=lambda kv: kv[1])[0]
    amt = concrete if vals else {"all": 5.0, "part": 2.75, "none": 0.0}.get(abstract, np.nan)
    return {"rx_concrete": concrete, "rx_abstract": abstract, "rx_amount": amt, "rx_n": len(vals)}


# ------------------------------------------------------------------ loaders
def load_myths() -> pd.DataFrame:
    m = pd.read_csv(DATA / "myths.csv")
    r = pd.read_csv(DATA / "myth_rules_september_z-ai__glm-5.2.csv")
    r = r[r["status"] == "ok"]
    chk = pd.read_csv(DATA / "myth_amount_check_september_z-ai__glm-5.2.csv")[["run_id", "round", "agent", "amount_status"]]
    keys = ["run_id", "round", "agent"]
    m = m.merge(r[keys + ["send_rule", "send_amount", "return_rule", "after_letdown", "consistency"]], on=keys, how="left")
    m = m.merge(chk, on=keys, how="left")
    band = m["send_rule"].map(MIDPOINT)
    named = m["send_amount"].notna()
    endorsed = m["amount_status"] == "endorsed"
    m["judge_amount"] = m["send_amount"].where(named, band)            # any named, else band
    m["judge_amount_strict"] = m["send_amount"].where(named & endorsed, band)  # endorsed, else band
    m["judge_return"] = m["return_rule"].map(RETURN_SHARE)
    rx = pd.DataFrame([regex_amount(t) for t in m["text"]], index=m.index)
    m = pd.concat([m, rx], axis=1)
    m["setting"] = m["size"].astype(str) + "-agent " + np.where(m["mixed"], "mixed", "homogeneous")
    return m


def load_decisions() -> pd.DataFrame:
    d = pd.read_csv(DATA / "decisions.csv")
    d["setting"] = d["size"].astype(str) + "-agent " + np.where(d["mixed"], "mixed", "homogeneous")
    return d

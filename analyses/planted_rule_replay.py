#!/usr/bin/env python3
"""Planted-rule replay: does a rule read in another agent's myth get played and passed on?

A saved moment from an Opus 5 population with two defectors (frontier defector pilot,
2026-10-01) is replayed with a shared myth board in the myth prompt. One myth on the board
is replaced by a planted story that is identical in three groups except for one sentence:

  none     "... send none of your five coins."
  neutral  "... settle on your amount before you reach the lamp."   (no amount named)
  all      "... send all five coins."

Four settings cross Ivar's two ideas:

  own myths kept / removed   removed = the agent's own myth exchanges are deleted from its
                             chat memory and its own myths are taken off the board
  line self / others         self   = "Use the myth you wrote in the previous round as inspiration ..."
                             others = "Use the myths written by other agents on the board as inspiration ..."

Three calls per case (context x group x setting), one sample each:
  myth     the agent writes its next myth                      (does the rule enter the myth?)
  send     the same agent then decides how much to send        (does the reader play by it?)
  reader   a fresh agent reads only that new myth and sends    (does the rule survive retelling?)
plus a check: fresh agents read the planted story itself and send (``direct``).

Contexts: per run (10 runs: 5 game->myth, 5 myth->game), two ordinary agents whose myth call
is followed directly by a SENDER decision, one against an ordinary co-player (logged sends
~$5, room to fall) and one against a forced-zero defector (logged sends ~$1, room to rise).
The round nearest 5 is taken. The runs had no board; the board is built from each run's own
earlier myths (all eight agents, rounds 1..r-1, shuffled within round, no author names) with
the board template of the 2026-11 board pilot.

Verdict rules, fixed before the run (group contrasts in dollars, clustered by run):
  direct readers do not differ between groups           -> test invalid, stop
  reader sends differ between groups by >= $0.50        -> rules spread through myths
  send differs but reader does not                      -> the rule reaches the reader and stops
  neither differs                                       -> rules do not spread through myths
  the gap is larger in a changed setting than in kept/self -> that idea helps

  python3 analyses/planted_rule_replay.py --plan
  python3 analyses/planted_rule_replay.py --run myth --limit 6      # smoke
  python3 analyses/planted_rule_replay.py --run myth,send,reader,direct --workers 8
  python3 analyses/planted_rule_replay.py --judge
  python3 analyses/planted_rule_replay.py --analyze
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import random
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses.myth_replay_probe import DEFAULT_DATA_ROOT, append, done_keys, parse_send, replay  # noqa: E402
from src.myth_writer import MYTH_BOARD_MEMORY_NOTE, MythWriter  # noqa: E402

OUT = ROOT / "data/analysis/planted_rule_replay_20261110"
FIGS = ROOT / "docs/research/planted_rule_replay_20261110"
RECEIPT = "data/json/noise_experiments/frontier_defector_pilot_20261001/all_receipt.json"
SEED = 20261110
PRICE = (5.0, 25.0)  # Opus 5 USD per 1M in/out: scripts/run_frontier_defector_populations.py RATES
GROUPS = ("none", "neutral", "all")
RULE = {"none": "send none of your five coins",
        "neutral": "settle on your amount before you reach the lamp",
        "all": "send all five coins"}
SETTINGS = {"kept_self": (True, "self"), "removed_self": (False, "self"),
            "kept_others": (True, "others"), "removed_others": (False, "others"),
            # comparison without a board (added after the board result): the plant is the one partner myth
            # the logged prompt shows, as in the original runs; memory and wording untouched
            "noboard_self": (True, "self")}
SHOWN_PREFIX = "Here is the myth the other agent wrote in the previous round:\n"
SHOWN_SUFFIX = "\n\nWrite your own myth."
LINE = "Use the myth you wrote in the previous round as inspiration, but adapt it in your own way."
LINE_OTHERS = "Use the myths written by other agents on the board as inspiration, but adapt them in your own way."
DIRECT_SAMPLES = 10
MARKERS = ("oskeld", "hollow crossing", "lamplighter")  # names only the planted story uses
FACT = re.compile(r"green (flame|fire|light|lamp)|(flame|fire|lamp|light)[^.]{0,40}\bgreen\b|salt from the sea|sea[- ]salt", re.I)  # the arbitrary fact
PLANT = """Myth: **The Lamp of Oskeld**

Before the valley had roads, travelers met at the Hollow Crossing after dark, and none could see another's face. Each carried five coins, and each wondered what to place in the stranger's hand.

Oskeld the lamplighter kept the only flame at the crossing, and it burned green, for she fed it salt from the sea. For forty winters she watched who came and who went, who returned fed and who returned hungry. She said little, but she counted everything.

In her last winter the travelers gathered and asked her: "You have seen every exchange made beneath your lamp. Tell us what you learned."

Oskeld trimmed the wick and spoke one sentence, and the travelers carved it on the lamp-post so that it would outlast her:

*When you hold five coins and it is your turn to send, {rule}.*

Some grumbled that the rule was too simple. Oskeld shrugged. "I did not invent it. I only watched for forty winters, and this is what the ones who came back every year had done."

The lamp still burns at the Hollow Crossing. Those who keep Oskeld's rule say they have never regretted it, and the post has been carved anew seven times, always with the same words."""
READER_INTRO = "Here is a myth another agent wrote in this session:\n{myth}\n\n"


def board_template() -> str:
    cfg = yaml.safe_load((ROOT / "config/experiments_noisy.yaml").read_text())
    t = cfg["prompt_templates"]["myth_writing_later_rounds_board_memory_primary"]
    assert t.count(LINE) == 1 and "{myth_board}" in t
    return t


def is_myth_prompt(text: str) -> bool:
    return "Format exactly:\nMyth:" in text


# ----------------------------------------------------------------------------- contexts

def contexts(data_root: Path) -> list[dict]:
    receipt = json.loads((data_root / RECEIPT).read_text())
    finals = [f for f in receipt["finals"] if "sol" not in json.dumps(f["agent_models"]) and f["task_order"] != "game"]
    assert len(finals) == 10, len(finals)
    template = board_template()
    out = []
    for f in sorted(finals, key=lambda f: (f["task_order"], f["replicate_id"])):
        d = json.loads((data_root / f["path"]).read_text())
        meta = d["run_metadata"]
        defectors = set(meta["comparison_inputs"]["game_params"]["defector_agent_ids"])
        ordinary = sorted(a for a in d["agents"] if a not in defectors)
        myths = {e["round"]: e["myths"] for e in d["conversation_history"]}
        round1 = next(h["messages_sent"][-1]["content"] for a in ordinary for h in d["agents"][a]["interaction_history"]
                      if h["metadata"]["task"] == "game" and h["metadata"]["round"] == 1 and h["metadata"]["role"] == "investor")
        run_id = f"{f['task_order']}_rep{f['replicate_id']}"
        rng = random.Random(f"{SEED}|{run_id}")
        cands = {"ordinary": [], "defector": []}
        for a in ordinary:
            hist = d["agents"][a]["interaction_history"]
            for i in range(len(hist) - 1):
                m, g = hist[i], hist[i + 1]
                if (m["metadata"]["task"] == "myth" and 3 <= m["metadata"]["round"] <= 7 and g["metadata"]["task"] == "game"
                        and g["metadata"]["role"] == "investor" and g.get("response_source", "llm") == "llm"):
                    kind = "defector" if g["metadata"]["opponent_id"] in defectors else "ordinary"
                    cands[kind].append((abs(m["metadata"]["round"] - 5), rng.random(), a, m, g))
        used = set()
        for kind in ("defector", "ordinary"):  # defector contexts are scarcer: pick them first
            pool = sorted(c for c in cands[kind] if c[2] not in used) or sorted(cands[kind])
            assert pool, (run_id, kind)
            _, _, a, m, g = pool[0]
            used.add(a)
            rnd = m["metadata"]["round"]
            msgs = m["messages_sent"]
            assert msgs[0]["role"] == "system" and is_myth_prompt(msgs[-1]["content"]) and LINE in msgs[-1]["content"]
            logged_myth = m["response"]["content"]
            # the logged send call must be the myth context minus its oldest exchange, plus the myth exchange
            gm = g["messages_sent"]
            assert gm[-2]["content"] == logged_myth and gm[-3]["content"] == msgs[-1]["content"], (run_id, a, rnd)
            assert [x["content"] for x in gm[1:-3]] == [x["content"] for x in msgs[3:-1]], (run_id, a, rnd)
            plant_author = random.Random(f"{SEED}|{run_id}|{a}|plant").choice([x for x in ordinary if x != a])
            items = []
            for r in range(1, rnd):
                authors = sorted(myths[r])
                random.Random(f"{SEED}|{run_id}|{a}|{r}").shuffle(authors)
                items += [(r, x, myths[r][x]) for x in authors]
            out.append({"run_id": run_id, "task_order": f["task_order"], "agent": a, "round": rnd, "partner": kind,
                        "plan": meta["llm_request"], "temperature": m.get("temperature", "default"),
                        "system": msgs[0], "memory": msgs[1:-1], "items": items, "plant_author": plant_author,
                        "template": template, "game_prompt": gm[-1]["content"], "round1_prompt": round1,
                        "logged_myth": logged_myth, "logged_prompt": msgs[-1]["content"], "logged_send": parse_send(g["response"]["content"]),
                        "ctx_id": f"{run_id}|{a}|r{rnd}"})
    return out


def myth_prompt(c: dict, group: str, setting: str, remembered: bool = False) -> str:
    keep_own, line = SETTINGS[setting]
    if setting == "noboard_self":
        head, rest = c["logged_prompt"].split(SHOWN_SUFFIX, 1)
        assert head.startswith(SHOWN_PREFIX)
        return SHOWN_PREFIX + PLANT.format(rule=RULE[group]) + SHOWN_SUFFIX + rest
    items = [(r, a, PLANT.format(rule=RULE[group]) if (r == c["round"] - 1 and a == c["plant_author"]) else t)
             for r, a, t in c["items"] if keep_own or a != c["agent"]]
    board = (MYTH_BOARD_MEMORY_NOTE.format(count=len(items), last_round=c["round"] - 1) if remembered
             else MythWriter._format_board(items))
    text = c["template"].format(myth_board=board)
    return text if line == "self" else text.replace(LINE, LINE_OTHERS)


def memory(c: dict, setting: str) -> list[dict]:
    if SETTINGS[setting][0]:
        return list(c["memory"])
    mem = c["memory"]
    assert len(mem) % 2 == 0
    return [x for i in range(0, len(mem), 2) if not is_myth_prompt(mem[i]["content"]) for x in mem[i:i + 2]]


def myth_messages(c: dict, group: str, setting: str) -> list[dict]:
    return [c["system"], *memory(c, setting), {"role": "user", "content": myth_prompt(c, group, setting)}]


def send_messages(c: dict, group: str, setting: str, new_myth: str) -> list[dict]:
    mem = memory(c, setting)
    if SETTINGS[setting][0]:
        mem = mem[2:]  # capacity six exchanges: the oldest drops when the myth exchange is added
    return [c["system"], *mem, {"role": "user", "content": myth_prompt(c, group, setting, remembered=True)},
            {"role": "assistant", "content": new_myth}, {"role": "user", "content": c["game_prompt"]}]


def reader_messages(c: dict, myth: str) -> list[dict]:
    return [c["system"], {"role": "user", "content": READER_INTRO.format(myth=myth) + c["round1_prompt"]}]


# ----------------------------------------------------------------------------- run

def load(stage: str) -> dict:
    path = OUT / f"{stage}.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {r["key"]: r for r in rows if not r.get("error")}


def cells(ctxs):
    return [(c, g, s) for c in ctxs for g in GROUPS for s in SETTINGS]


def jobs_for(stage: str, ctxs: list[dict]) -> list[dict]:
    jobs = []
    if stage == "direct":
        c = ctxs[0]
        jobs = [{"key": f"direct|{g}|s{i}", "ctx": c, "group": g, "setting": "direct",
                 "messages": reader_messages(c, PLANT.format(rule=RULE[g]))} for g in GROUPS for i in range(DIRECT_SAMPLES)]
    else:
        myths = load("myth") if stage != "myth" else {}
        for c, g, s in cells(ctxs):
            key = f"{c['ctx_id']}|{g}|{s}"
            if stage == "myth":
                msgs = myth_messages(c, g, s)
            else:
                new = (myths.get(key) or {}).get("text", "")
                if len(new.split()) < 50:
                    continue  # no usable myth yet: retried after the myth stage is complete
                msgs = send_messages(c, g, s, new) if stage == "send" else reader_messages(c, new)
            jobs.append({"key": key, "ctx": c, "group": g, "setting": s, "messages": msgs})
    random.Random(SEED + 1).shuffle(jobs)  # interleave groups so drift over time hits all of them
    return jobs


def cost(usage: dict) -> float:
    return ((usage or {}).get("input_tokens") or 0) * PRICE[0] / 1e6 + ((usage or {}).get("output_tokens") or 0) * PRICE[1] / 1e6


def run(stage: str, ctxs: list[dict], workers: int, limit: int | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stage}.jsonl"
    done = done_keys(path)
    pending = [j for j in jobs_for(stage, ctxs) if j["key"] not in done][:limit]

    def work(j):
        c = j["ctx"]
        res = replay({"plan": c["plan"], "temperature": c["temperature"]}, j["messages"])
        rec = {"key": j["key"], "stage": stage, "run_id": c["run_id"], "task_order": c["task_order"], "agent": c["agent"],
               "round": c["round"], "partner": c["partner"], "group": j["group"], "setting": j["setting"],
               "text": res["text"], "send": res["send"], "error": res["error"], "finish_reason": res.get("finish_reason"),
               "usage": res["usage"], "cost": cost(res["usage"])}
        append(path, rec)
        return rec

    spent, errors = 0.0, 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for n, rec in enumerate(pool.map(work, pending), 1):
            spent += rec["cost"]
            errors += bool(rec["error"])
            if n % 20 == 0 or n == len(pending):
                print(f"{stage}: {n}/{len(pending)} done, ${spent:.2f}, errors {errors}", flush=True)


# ----------------------------------------------------------------------------- judge (blind: sees only the myth text)

def judge() -> None:
    sys.path.insert(0, str(ROOT / "analyses"))
    from myth_moral_judge import Judge
    from myth_rule_judge import parse as rparse, render as rrender
    j = Judge("z-ai/glm-5.2")
    path = OUT / "judge.jsonl"
    done = done_keys(path)
    rows = [r for r in load("myth").values() if r["key"] not in done]

    def work(r):
        res = j(rrender(r["text"]), lambda raw: rparse(raw)[1] == "ok")
        obj, status = rparse(res.get("raw", ""))
        rec = {"key": r["key"], "status": status, "error": None if obj else status, "cost": float(res.get("cost") or 0),
               **({k: obj.get(k) for k in ("send_rule", "send_amount")} if obj else {})}
        append(path, rec)
        return rec

    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        recs = list(pool.map(work, rows))
    print(f"judge: {len(recs)} myths, ${sum(r['cost'] for r in recs):.2f}, failed {sum(bool(r['error']) for r in recs)}")


# ----------------------------------------------------------------------------- analysis

def contrast(df: pd.DataFrame, value: str, a: str, b: str) -> dict:
    """Mean of (group a - group b) within context x setting, then a t interval over runs."""
    w = df.pivot_table(index=["run_id", "ctx", "setting"], columns="group", values=value)
    d = (w[a] - w[b]).dropna()
    per_run = d.groupby(level="run_id").mean()
    n = len(per_run)
    half = stats.t.ppf(0.975, n - 1) * per_run.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    p = stats.ttest_1samp(per_run, 0).pvalue if n > 1 and per_run.std(ddof=1) > 0 else np.nan
    return {"contrast": f"{a} - {b}", "n_cases": len(d), "n_runs": n, "mean_a": w[a].mean(), "mean_b": w[b].mean(),
            "diff": per_run.mean(), "ci_lo": per_run.mean() - half, "ci_hi": per_run.mean() + half, "p": p}


def analyze() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    myth, send, reader, direct, jd = (load(s) for s in ("myth", "send", "reader", "direct", "judge"))
    rows = []
    for key, m in myth.items():
        low = m["text"].lower()
        j = jd.get(key, {})
        rows.append({"key": key, "run_id": m["run_id"], "ctx": f"{m['run_id']}|{m['agent']}", "task_order": m["task_order"],
                     "partner": m["partner"], "group": m["group"], "setting": m["setting"],
                     "own": m["setting"].split("_")[0], "line": m["setting"].split("_")[1],
                     "marker": float(any(k in low for k in MARKERS)), "fact": float(bool(FACT.search(low))), "n_words": len(m["text"].split()),
                     "judged_rule": j.get("send_rule"), "judged_amount": j.get("send_amount"),
                     "rule_none": float(j.get("send_rule") == "none") if j else np.nan,
                     "rule_all": float(j.get("send_rule") == "all") if j else np.nan,
                     "send": (send.get(key) or {}).get("send"), "reader_send": (reader.get(key) or {}).get("send")})
    df = pd.DataFrame(rows)
    df.to_csv(FIGS / "cases.csv", index=False)
    pd.set_option("display.width", 250)

    d = pd.DataFrame([{"group": r["group"], "send": r["send"]} for r in direct.values()])
    dsum = d.groupby("group")["send"].agg(["count", "mean", "std"]).reindex(GROUPS)
    dsum.to_csv(FIGS / "direct_reader.csv")
    print("DIRECT READERS (read the planted story itself)\n", dsum.round(2), "\n")

    means = df.groupby(["setting", "group"]).agg(n=("send", "size"), marker=("marker", "mean"), fact=("fact", "mean"), rule_none=("rule_none", "mean"),
                                                 rule_all=("rule_all", "mean"), send=("send", "mean"), send_sd=("send", "std"),
                                                 reader=("reader_send", "mean"), reader_sd=("reader_send", "std"))
    means = means.reindex([(s, g) for s in SETTINGS for g in GROUPS])
    means.to_csv(FIGS / "means_by_setting_group.csv")
    print(means.round(2), "\n")
    bypartner = df[df["setting"] != "noboard_self"].groupby(["partner", "group"]).agg(n=("send", "size"), send=("send", "mean"), send_sd=("send", "std"),
                                                     reader=("reader_send", "mean"), reader_sd=("reader_send", "std"))
    bypartner.to_csv(FIGS / "means_by_partner_group.csv")
    print(bypartner.round(2), "\n")

    out = []
    board = df[df["setting"] != "noboard_self"]
    subsets = [("all settings", board)] + [(s, df[df["setting"] == s]) for s in SETTINGS] + \
              [(f"partner {p}", board[board["partner"] == p]) for p in ("ordinary", "defector")]
    for label, sub in subsets:
        for value in ("marker", "fact", "rule_none", "rule_all", "send", "reader_send"):
            for a, b in (("all", "none"), ("none", "neutral"), ("all", "neutral")):
                if sub[value].notna().sum():
                    out.append({"subset": label, "measure": value, **contrast(sub, value, a, b)})
    s = pd.DataFrame(out)
    s.to_csv(FIGS / "contrasts.csv", index=False)
    print(s[s["subset"] == "all settings"].round(3).to_string(index=False), "\n")
    print(s[(s["contrast"] == "all - none") & s["measure"].isin(["send", "reader_send", "marker"])].round(3).to_string(index=False))
    total = sum(r.get("cost", 0) for st in (myth, send, reader, direct, jd) for r in st.values())
    print(f"\ncalls: myth {len(myth)}, send {len(send)}, reader {len(reader)}, direct {len(direct)}, judged {len(jd)}; cost ${total:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", default="")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--est-cost", type=float, default=35.0)
    args = ap.parse_args()
    stages = [s for s in args.run.split(",") if s]
    if args.plan or stages:
        ctxs = contexts(args.data_root)
        print(pd.DataFrame([{k: c[k] for k in ("run_id", "agent", "round", "partner", "logged_send")} for c in ctxs]).to_string(index=False))
        n = len(cells(ctxs)) * 3 + len(GROUPS) * DIRECT_SAMPLES
        print(f"MODEL={ctxs[0]['plan']['model']} (logged plan: adaptive thinking, effort high) N={n} "
              f"WORKERS={args.workers} EST_COST=${args.est_cost:.2f}", flush=True)
        if args.plan:
            c = ctxs[0]
            for st, msgs in (("myth", myth_messages(c, "none", "kept_self")), ("myth removed_others", myth_messages(c, "none", "removed_others")),
                             ("send", send_messages(c, "none", "removed_others", "Myth: <new myth>")), ("reader", reader_messages(c, "Myth: <new myth>"))):
                print(f"--- {st}: {len(msgs)} messages, {sum(len(m['content']) for m in msgs)} chars")
            print(myth_messages(c, "none", "removed_others")[-1]["content"][-900:])
            print("--- send tail"); print(json.dumps(send_messages(c, "none", "removed_others", "Myth: <new myth>")[-3:], indent=1)[:1800])
            print("--- reader"); print(reader_messages(c, "Myth: <new myth>")[-1]["content"])
    for st in stages:
        run(st, ctxs, args.workers, args.limit)
    if args.judge:
        judge()
    if args.analyze:
        analyze()


if __name__ == "__main__":
    main()

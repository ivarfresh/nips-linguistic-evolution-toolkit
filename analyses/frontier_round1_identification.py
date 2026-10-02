#!/usr/bin/env python3
"""Round-1 identification check on every frontier myth->game run.
For each round-1 sender: the messages of its send call, with its own round-1 myth masked.
Within a composition cell, everything except the own myth should be byte-identical."""
import json, hashlib, collections
import pandas as pd
from pathlib import Path
R = str(Path(__file__).resolve().parents[1]) + "/"
m = pd.read_csv(R + "data/analysis/linguistic_frontier_20260930/myths.csv")
runs = m[m.task_order == "myth_game"].drop_duplicates("path")[["path", "composition", "size"]]
fam = dict(zip(zip(m.run_id, m.agent), m.family))
H = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:12]
sig = collections.defaultdict(collections.Counter)      # cell -> masked-message hash counts (senders)
recv = collections.defaultdict(collections.Counter)     # cell -> masked hash w/o the partner-send line
settings = collections.defaultdict(set)                 # (cell, family) -> request settings hash
extra_keys = collections.Counter(); reasoning_nonnull = collections.Counter(); own_match = collections.Counter()
myth_call = collections.defaultdict(collections.Counter)
n_senders = collections.Counter()
for p, comp, size in runs.itertuples(index=False):
    r = json.load(open(R + p)); rid = p.rsplit("/", 1)[1][:-5]
    md = r["run_metadata"]; plan = md.get("llm_request") or {}
    per_agent = plan.get("agents") or {a: plan for a in r["agents"]}
    myths1 = r["conversation_history"][0]["myths"]
    for a, ag in r["agents"].items():
        f = fam[(rid, a)]
        settings[(comp, f)].add(H({k: v for k, v in per_agent[a].items() if k != "agents"}) + "|T=" + str(ag.get("temperature")))
        for ih in ag["interaction_history"]:
            md_ = ih["metadata"]
            for msg in ih["messages_sent"]:
                for k in msg:
                    if k not in ("role", "content"): extra_keys[k] += 1
                if msg.get("reasoning"): reasoning_nonnull[(f, msg["role"])] += 1
            if md_["round"] != 1: continue
            if md_["task"] == "myth":
                myth_call[comp][H(ih["messages_sent"])] += 1
                continue
            msgs = [(x["role"], x["content"]) for x in ih["messages_sent"]]
            asst = [i for i, (ro, _) in enumerate(msgs) if ro == "assistant"]
            own_match[len(asst) == 1 and msgs[asst[0]][1] == myths1[a]] += 1
            masked = [(ro, "<OWN MYTH>" if ro == "assistant" else c) for ro, c in msgs]
            if md_["role"] == "investor":
                sig[comp][H(masked)] += 1; n_senders[(comp, f)] += 1
            else:
                recv[comp][H(masked[:-1])] += 1
print("extra keys in messages_sent (stored, dropped by the transport):", dict(extra_keys))
print("non-empty stored reasoning by (family, role):", dict(reasoning_nonnull))
print("own myth is the only assistant turn and equals the recorded round-1 myth:", dict(own_match))
for comp in sorted(sig):
    print(f"{comp}: sender masked-message variants {len(sig[comp])} over {sum(sig[comp].values())} senders; "
          f"receiver (minus last prompt) variants {len(recv[comp])}; myth-call variants {len(myth_call[comp])}")
for k in sorted(settings): print("settings", k, len(settings[k]), "variant(s);", "senders:", n_senders[k])

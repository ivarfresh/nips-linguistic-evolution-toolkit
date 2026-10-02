#!/usr/bin/env python3
"""0-10 giving score for every September myth (send, return, conditional).

Same request settings as analyses/myth_moral_judge.py (OpenRouter, temperature 0,
reasoning off, JSON mode, same system prompt), but responses are cached in this
lens folder, not the repo.

  python3 giving_judge.py --model z-ai/glm-5.2 --preflight
  python3 giving_judge.py --model z-ai/glm-5.2 --limit 100
  python3 giving_judge.py --model z-ai/glm-5.2
  # frontier corpus: read and write its data dir, cache under the gitignored judge cache
  python3 giving_judge.py --model z-ai/glm-5.2 --data-dir <repo>/data/analysis/linguistic_frontier_20260930 \
      --out-dir <same> --cache-dir <repo>/data/judge_cache/moral/giving
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
WT = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-spread-viz")
DATA = WT / "data/analysis/linguistic_20260923"
ENV = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-linguistic-evolution-toolkit/.env")
RUBRIC = HERE / "giving_rubric.txt"
SYSTEM = ("You are an impartial LLM-as-judge. Return only valid JSON unless the "
          "user explicitly requests another format.")
PRICES = {"z-ai/glm-5.2": (0.65, 2.042), "deepseek/deepseek-v4-flash": (0.089, 0.177)}
FIELDS = ["send", "return", "conditional"]


def render(text):
    return RUBRIC.read_text().replace("{text}", text)


def parse(raw):
    s = (raw or "").strip()
    try:
        obj = json.loads(s[s.find("{"): s.rfind("}") + 1])
        out = {f: float(obj[f]) for f in FIELDS}
    except Exception:  # noqa: BLE001
        return None
    return out if all(0 <= v <= 10 for v in out.values()) else None


class Judge:
    def __init__(self, model, cache_root=None):
        from openai import OpenAI
        from dotenv import dotenv_values
        key = dotenv_values(ENV).get("OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        self.model = model
        self.cache = (cache_root or HERE / "cache") / model.replace("/", "__")

    def __call__(self, user, max_retries=8):
        h = hashlib.sha256(f"{self.model}|T=0|{SYSTEM}|{user}".encode()).hexdigest()
        cp = self.cache / h[:2] / f"{h}.json"
        if cp.exists():
            c = json.loads(cp.read_text())
            if parse(c.get("raw")):
                return {**c, "cached": True}
        delay = 2.0
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=0, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    extra_body={"usage": {"include": True}, "reasoning": {"enabled": False}}, timeout=120)
                u = resp.usage.model_dump() if resp.usage else {}
                out = {"raw": resp.choices[0].message.content or "", "cost": u.get("cost"),
                       "model_served": resp.model}
                if parse(out["raw"]):
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    cp.write_text(json.dumps(out))
                    return {**out, "cached": False}
            except Exception as err:  # noqa: BLE001
                out = {"raw": "", "error": str(err), "cost": 0}
            time.sleep(delay)
            delay = min(delay * 2, 60)
        return {**out, "cached": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=DATA, help="directory holding myths.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE, help="where giving_scores_<model>.csv goes")
    ap.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    a = ap.parse_args()
    m = pd.read_csv(a.data_dir / "myths.csv")
    m = m[m.n_words >= 20]
    if a.limit:
        m = m.sample(n=a.limit, random_state=0)
    prompts = [render(t) for t in m.text]
    pin, pout = PRICES[a.model]
    est = (sum(len(p) for p in prompts) / 4 + 40 * len(m)) / 1e6 * pin + 30 * len(m) / 1e6 * pout
    print(f"MODEL={a.model} N={len(m)} WORKERS={a.workers} EST_COST=${est * 1.5:.2f} (x1.5 margin)")
    if a.preflight:
        return
    j = Judge(a.model, a.cache_dir)
    res = [None] * len(prompts)
    with ThreadPoolExecutor(a.workers) as pool:
        fut = {pool.submit(j, p): i for i, p in enumerate(prompts)}
        for n, f in enumerate(as_completed(fut), 1):
            res[fut[f]] = f.result()
            if n % 1000 == 0:
                print(n, f"${sum((r or {}).get('cost') or 0 for r in res if r and not r.get('cached')):.2f}", flush=True)
    out = m[["run_id", "round", "agent"]].copy()
    parsed = [parse(r.get("raw")) or {} for r in res]
    for f in FIELDS:
        out[f"g_{f}"] = [p.get(f) for p in parsed]
    out["cost"] = [r.get("cost") or 0 for r in res]
    out["cached"] = [bool(r.get("cached")) for r in res]
    tag = a.model.replace("/", "__") + (f"_sample{a.limit}" if a.limit else "")
    out.to_csv(a.out_dir / f"giving_scores_{tag}.csv", index=False)
    print(f"parsed {out.g_send.notna().sum()}/{len(out)}; billed ${out.loc[~out.cached, 'cost'].sum():.2f}")


if __name__ == "__main__":
    main()

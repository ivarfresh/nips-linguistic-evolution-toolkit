"""Replay recorded Gemini 3.7 Flash calls through Vertex AI and through the Gemini API.

Question: does Vertex AI still serve the Gemini 3.7 Flash the paper used, and do its
answers look like the recorded ones? Google redirected ``gemini-3.7-flash`` on the
Gemini API to Gemini 3.8 Flash on 2026-10-10, so the Gemini API column is a 3.8 baseline.

For each sampled call we resend the exact recorded ``messages_sent`` with the recorded
generation parameters (thinking level high, temperature, output cap) to:
  vertex  https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3.7-flash
  gemini  https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash
and store the full text, the reported modelVersion and token counts.

Usage:
  python scripts/gemini_route_replay.py --plan            # sample prompts, print preflight, no calls
  python scripts/gemini_route_replay.py --run --workers 8 # make the calls (resumable)
"""
import argparse
import glob
import json
import random
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import _gemini_messages, _gemini_api_key, _env  # noqa: E402

import os  # noqa: E402
# Run data is not in git; point at the checkout that holds it (default: this one).
DROOT = Path(os.environ.get("RUN_DATA_ROOT", str(ROOT)))
DATA = DROOT / "data/json/noise_experiments"
from dotenv import load_dotenv  # noqa: E402
load_dotenv(DROOT / ".env", override=True)  # the project's own keys, not a parent folder's .env
SETS = ["negative_only_crossmodel_reasoning_rerun_20260909", "table1_n10_extension_20261001",
        "mixed_model_populations_20260918", "mixed_model_dyads_20260917"]
ROUTES = {
    "vertex": "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3.7-flash:generateContent",
    "gemini_api": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent",
}
PRICE_IN, PRICE_OUT = 0.75, 3.75  # $/M; output includes thinking
SEED = 20261126
_lock = threading.Lock()


def finals(set_name):
    for f in glob.glob(str(DATA / set_name / "**/*.json"), recursive=True):
        if any(s in f for s in (".results.", "checkpoint", "receipt", "quarantine")):
            continue
        yield f


def sample(n_founding, n_later, n_game):
    rng = random.Random(SEED)
    founding, later, game = [], [], []
    for s in SETS:
        for f in finals(s):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            order = "_".join(d.get("task_order", []))
            for aid, a in d.get("agents", {}).items():
                for k, h in enumerate(a.get("interaction_history", [])):
                    u = h["response"].get("usage") or {}
                    rs = u.get("request_settings") or {}
                    if "gemini-3.7" not in rs.get("model", "") or u.get("outcome") != "complete":
                        continue
                    md = h.get("metadata") or {}
                    rec = {"file": str(Path(f).relative_to(DROOT)), "agent": aid, "index": k, "set": s,
                           "order": order, "task": md.get("task"), "round": md.get("round"),
                           "role": md.get("role")}
                    if md.get("task") == "myth" and md.get("round") == 1 and order == "myth_game":
                        founding.append(rec)
                    elif md.get("task") == "myth" and (md.get("round") or 0) >= 3:
                        later.append(rec)
                    elif md.get("task") == "game":
                        game.append(rec)
    out = []
    for kind, pool, n in (("founding_myth", founding, n_founding), ("later_myth", later, n_later), ("game", game, n_game)):
        rng.shuffle(pool)
        seen = set()
        for r in pool:
            key = (r["file"], r["agent"])
            if key in seen:
                continue
            seen.add(key)
            out.append({**r, "kind": kind, "id": f"{kind}-{len([x for x in out if x['kind'] == kind]):03d}"})
            if sum(1 for x in out if x["kind"] == kind) >= n:
                break
    return out


def load_call(rec):
    d = json.load(open(DROOT / rec["file"]))
    h = d["agents"][rec["agent"]]["interaction_history"][rec["index"]]
    u = h["response"]["usage"]
    return h["messages_sent"], u["request_settings"]["parameters"], {
        "text": h["response"].get("content"), "model_version": u.get("response_model"),
        "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
        "thinking_tokens": u.get("reasoning_tokens")}


def call(route, messages, params):
    system, contents = _gemini_messages(messages)
    payload = {"contents": contents, "generationConfig": params}
    if system:
        payload["system_instruction"] = system
    key = (_env("VERTEX_API_KEY") or _gemini_api_key()) if route == "vertex" else _gemini_api_key()
    req = urllib.request.Request(ROUTES[route], data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 503) and attempt < 4:
                import time
                time.sleep(2 ** attempt * 5)
                continue
            return {"error": f"HTTP {e.code}: {body}"}
    parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    um = data.get("usageMetadata") or {}
    return {"text": "".join(p.get("text", "") for p in parts if not p.get("thought")),
            "model_version": data.get("modelVersion"), "input_tokens": um.get("promptTokenCount"),
            "output_tokens": um.get("candidatesTokenCount"), "thinking_tokens": um.get("thoughtsTokenCount"),
            "finish_reason": (data.get("candidates") or [{}])[0].get("finishReason")}


def cost(r):
    if not r or r.get("error"):
        return 0.0
    return ((r.get("input_tokens") or 0) * PRICE_IN + ((r.get("output_tokens") or 0) + (r.get("thinking_tokens") or 0)) * PRICE_OUT) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DROOT / "data/analysis/gemini_route_replay_20261126"))
    ap.add_argument("--founding", type=int, default=60)
    ap.add_argument("--later", type=int, default=30)
    ap.add_argument("--game", type=int, default=30)
    ap.add_argument("--routes", default="vertex,gemini_api")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "prompts.json"
    if plan_path.exists():
        prompts = json.load(open(plan_path))
    else:
        prompts = sample(a.founding, a.later, a.game)
        json.dump(prompts, open(plan_path, "w"), indent=1)
    if a.limit:
        prompts = prompts[: a.limit]
    routes = a.routes.split(",")
    est = 0.0
    for p in prompts:
        _, _, orig = load_call(p)
        est += cost(orig) * (2.0 if "gemini_api" in routes and len(routes) == 1 else (1 + 2.0) if "gemini_api" in routes else 1)
    print(f"MODEL=gemini-3.7-flash routes={','.join(routes)} N={len(prompts)} prompts x {len(routes)} routes "
          f"WORKERS={a.workers} EST_COST=${est:.2f} (3.8 priced at 2x recorded thinking)")
    if not a.run:
        return
    res_path = out / "responses.jsonl"
    done = set()
    if res_path.exists():
        for line in open(res_path):
            r = json.loads(line)
            if not r["result"].get("error"):
                done.add((r["id"], r["route"]))
    jobs = [(p, r) for p in prompts for r in routes if (p["id"], r) not in done]

    def work(job):
        p, route = job
        messages, params, _ = load_call(p)
        res = call(route, messages, params)
        with _lock, open(res_path, "a") as fh:
            fh.write(json.dumps({"id": p["id"], "route": route, "result": res}) + "\n")
        return route, res

    spent = 0.0
    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(work, j) for j in jobs]):
            route, res = fut.result()
            spent += cost(res)
            print(route, res.get("model_version") or res.get("error", "")[:120])
    print(f"DONE spent≈${spent:.2f}")


if __name__ == "__main__":
    main()

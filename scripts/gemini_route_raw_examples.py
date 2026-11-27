"""Save full raw request/response objects from the Gemini API and Vertex AI for a few recorded prompts.

One prompt per kind (first myth, later myth, game move) from the replay sample is sent to both
routes with the exact recorded body. Files land in <replay out>/raw_pair/:
  <id>_request.json             the body both routes received (identical), key redacted
  <id>_response_gemini_api.json the full JSON the Gemini API returned
  <id>_response_vertex.json     the full JSON Vertex AI returned
  <id>_recorded.json            the September call as stored in our run file

  python scripts/gemini_route_raw_examples.py
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gemini_route_replay import DROOT, load_call  # noqa: E402  (also loads the project .env)
from src.utils import _gemini_messages  # noqa: E402

URLS = {
    "gemini_api": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent",
    "vertex": "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3.7-flash:generateContent",
}
KEYS = {"gemini_api": "GEMINI_API_KEY", "vertex": "VERTEX_API_KEY"}


def main():
    base = DROOT / "data/analysis/gemini_route_replay_20261126"
    out = base / "raw_pair"
    out.mkdir(exist_ok=True)
    prompts = json.load(open(base / "prompts.json"))
    picks = [next(p for p in prompts if p["kind"] == k) for k in ("founding_myth", "later_myth", "game")]
    for p in picks:
        messages, params, rec = load_call(p)
        system, contents = _gemini_messages(messages)
        payload = {"contents": contents, "generationConfig": params}
        if system:
            payload["system_instruction"] = system
        json.dump({"urls": URLS, "headers": {"Content-Type": "application/json", "x-goog-api-key": "<redacted>"},
                   "body": payload}, open(out / f"{p['id']}_request.json", "w"), indent=1)
        json.dump({"source": p, "recorded": rec}, open(out / f"{p['id']}_recorded.json", "w"), indent=1)
        for route, url in URLS.items():
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "x-goog-api-key": os.environ[KEYS[route]]})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    raw = json.loads(r.read())
            except urllib.error.HTTPError as e:
                raw = {"http_error": e.code, "body": json.loads(e.read() or b"{}")}
            json.dump(raw, open(out / f"{p['id']}_response_{route}.json", "w"), indent=1)
            print(p["id"], route, raw.get("modelVersion") or raw.get("http_error"))


if __name__ == "__main__":
    main()

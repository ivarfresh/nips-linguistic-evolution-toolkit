"""Local prompt inspector for the mid-tier and frontier runs.

    python tools/prompt_inspector_server.py            # opens http://127.0.0.1:8765

Lists every final run in the batches below and serves one run at a time to
tools/prompt_inspector.html, which shows each agent's system prompt and, per
call, the new prompt, the full message list sent to the API, the reasoning,
the response and token usage. Read-only; binds to localhost only.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading
import urllib.parse
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
NOISE = ROOT / "data/json/noise_experiments"
# Group label -> run folders under data/json/noise_experiments.
BATCHES = {
    "Mid-tier": [
        ("Single-model (Sept, n=5)", NOISE / "negative_only_crossmodel_reasoning_rerun_20260909"),
        ("Mixed dyads", NOISE / "mixed_model_dyads_20260917"),
        ("Mixed 8-agent", NOISE / "mixed_model_populations_20260918"),
        ("Table 1 n=10 extension", NOISE / "table1_n10_extension_20261001"),
    ],
    "Frontier": [
        ("Frontier single-model", NOISE / "frontier_rerun_20260918"),
        ("Frontier mixed dyads", NOISE / "frontier_mixed_main_20260928"),
        ("Frontier mixed 8-agent", NOISE / "frontier_mixed_populations_20260928"),
    ],
}
NON_FINAL = re.compile(r"(\.results|\.checkpoint|\.error|receipt|provenance|manifest)", re.I)
FINAL_NAME = re.compile(r"_\d{3}_.*\.json$")


def build_index():
    runs, seen = [], set()
    for group, folders in BATCHES.items():
        for label, folder in folders:
            if not folder.is_dir():
                continue
            for p in sorted(folder.rglob("*.json")):
                if {"worker_logs", "quarantine"} & set(p.parts) or NON_FINAL.search(p.name) or not FINAL_NAME.search(p.name):
                    continue
                rel = p.relative_to(folder).parts
                if len(rel) < 5 or p.name in seen:
                    continue
                seen.add(p.name)
                rep = re.search(r"_rep(\d+)", p.name)
                runs.append({
                    "id": len(runs), "group": group, "batch": label, "set": rel[0],
                    "models": rel[1], "task_order": rel[2], "condition": rel[3],
                    "replicate": int(rep.group(1)) if rep else None, "file": p.name, "_path": str(p),
                })
    return runs


class Handler(BaseHTTPRequestHandler):
    index: list = []

    def log_message(self, *args):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._send((ROOT / "tools/prompt_inspector.html").read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/index":
            Handler.index = build_index()
            public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in Handler.index]
            return self._send(json.dumps(public).encode(), "application/json")
        if url.path == "/api/run":
            q = urllib.parse.parse_qs(url.query)
            try:
                run = Handler.index[int(q["id"][0])]
            except (KeyError, ValueError, IndexError):
                return self._send(b'{"error":"unknown run"}', "application/json", 404)
            return self._send(Path(run["_path"]).read_bytes(), "application/json")
        self._send(b"not found", "text/plain", 404)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Prompt inspector at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()

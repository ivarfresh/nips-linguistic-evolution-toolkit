"""Shared LLM-judge helper for post-hoc coding of experiment transcripts.

Provides:
  - `judge(system, user, model, temperature)`: cached call that returns parsed JSON.
  - Content-addressable disk cache keyed on hash(system + user + model + temperature).
  - OpenRouter client reusing `src/utils.py` retry logic.
  - Simple cost tracker printed at end.

The judge MUST be called with a prompt that instructs the model to return strict
JSON — the helper parses the response. If parsing fails, the raw string is returned
under the `_parse_error` key.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

# Reuse the repo's retry/error handling.
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import call_llm, OPENROUTER_API_KEY  # noqa: E402


CACHE_ROOT = Path(__file__).parent.parent / "data" / "judge_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


# Rough OpenRouter per-1M token prices for quick cost tracking. Update as needed.
PRICING_USD_PER_M = {
    "anthropic/claude-haiku-4.5": {"in": 1.00, "out": 5.00},
    "anthropic/claude-sonnet-4.5": {"in": 3.00, "out": 15.00},
    "openai/gpt-5-nano": {"in": 0.05, "out": 0.40},
    "openai/gpt-5-mini": {"in": 0.25, "out": 2.00},
    "google/gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    "meta-llama/llama-3.3-70b-instruct": {"in": 0.13, "out": 0.40},
}


@dataclass
class JudgeStats:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    est_cost_usd: float = 0.0
    per_model: dict[str, dict] = field(default_factory=dict)

    def add(self, model: str, usage: dict | None) -> None:
        self.calls += 1
        if not usage:
            return
        it = usage.get("input_tokens", 0) or 0
        ot = usage.get("output_tokens", 0) or 0
        self.input_tokens += it
        self.output_tokens += ot
        price = PRICING_USD_PER_M.get(model)
        if price:
            cost = (it / 1_000_000) * price["in"] + (ot / 1_000_000) * price["out"]
            self.est_cost_usd += cost
        pm = self.per_model.setdefault(model, {"calls": 0, "in": 0, "out": 0})
        pm["calls"] += 1
        pm["in"] += it
        pm["out"] += ot

    def print_summary(self) -> None:
        print(f"\n--- Judge usage ---")
        print(f"Calls: {self.calls} (cache hits: {self.cache_hits})")
        print(f"Tokens: in={self.input_tokens:,}  out={self.output_tokens:,}")
        print(f"Estimated cost: ~${self.est_cost_usd:.3f}")
        for m, stats in self.per_model.items():
            print(f"  {m}: {stats['calls']} calls, {stats['in']:,} in / {stats['out']:,} out")


_STATS = JudgeStats()


def _cache_key(system: str, user: str, model: str, temperature: float) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(f"|T={temperature}|".encode())
    h.update(b"SYS:")
    h.update(system.encode())
    h.update(b"USR:")
    h.update(user.encode())
    return h.hexdigest()


def _cache_path(key: str) -> Path:
    # Two-level sharding to keep directory sizes reasonable.
    return CACHE_ROOT / key[:2] / f"{key}.json"


def _get_client() -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing. Set it in .env at repo root.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


_CLIENT: OpenAI | None = None


def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _get_client()
    return _CLIENT


def _parse_json_block(content: str) -> tuple[Any, str | None]:
    """Extract JSON from response content. Handles code fences and leading prose."""
    s = content.strip()
    # Strip markdown fence if present
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # Heuristic: find first "{" and last "}".
    if "{" in s and "}" in s:
        first = s.find("{")
        last = s.rfind("}")
        s_core = s[first:last + 1]
    else:
        s_core = s
    try:
        return json.loads(s_core), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def judge(
    system: str,
    user: str,
    model: str = "anthropic/claude-haiku-4.5",
    temperature: float = 0.0,
    use_cache: bool = True,
    reasoning_effort: str | None = None,
) -> dict:
    """Run the judge and return parsed JSON (or `{"_parse_error": ..., "_raw": ...}`).

    Cached by (system, user, model, temperature).
    """
    key = _cache_key(system, user, model, temperature)
    cp = _cache_path(key)

    if use_cache and cp.exists():
        try:
            cached = json.loads(cp.read_text())
            _STATS.cache_hits += 1
            _STATS.calls += 1
            return cached
        except Exception:
            pass  # fall through to live call

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # Pass "" to disable reasoning entirely (call_llm checks `if reasoning_effort`).
    resp = call_llm(_client(), model, temperature, messages, reasoning_effort=reasoning_effort or "")
    content = resp["content"]
    parsed, err = _parse_json_block(content)
    if parsed is None:
        parsed = {"_parse_error": err, "_raw": content}

    _STATS.add(model, resp.get("usage"))

    if use_cache:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(parsed, ensure_ascii=False, indent=2))

    return parsed


def stats() -> JudgeStats:
    return _STATS


def reset_stats() -> None:
    global _STATS
    _STATS = JudgeStats()

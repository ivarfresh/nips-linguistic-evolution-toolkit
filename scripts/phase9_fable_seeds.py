"""Phase 9 — regenerate gowith and jabberwocky seed pools with Claude Fable 5
as the translator (previous pools were translated by Sonnet 4.5).

Pools produced (candidates first, registered into the manifest only with
per-seed verification status attached):
  s_end_plus_gowith_fable   gowith translations of seeds.s_end_plus
  s_start_jab_fable         jabberwocky translations of seeds.s_start
  s_end_plus_jab_fable      jabberwocky translations of seeds.s_end_plus

Verification per seed:
  - gowith: Sonnet readback of {send, return_fraction} from the translation alone
  - jabberwocky: word-count ratio vs original (structure preservation proxy)
  - ALL: refusal probe with the EXACT runtime-built game context (captured in
    data/phase7/debug_failing_call.json) against the Sonnet 4.5 HOST, 4 samples.
    Template-approximated probes are worthless — see researchlog 2026-07-02.

Usage: python scripts/phase9_fable_seeds.py
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import src.utils  # noqa: F401  (loads .env)
import anthropic

from phase5_jabberwocky_seeds import JABBERWOCKY_PROMPT, word_count  # noqa: E402

FABLE = "claude-fable-5"
HOST = "claude-sonnet-4-5-20250929"  # the model that plays the games
MANIFEST_PATH = REPO_ROOT / "data/phase3/seed_manifest.json"
GOWITH_SPEC = REPO_ROOT / "data/phase7/gowith_spec.md"
GAME_CONTEXT = REPO_ROOT / "data/phase7/debug_failing_call.json"
OUT_PATH = REPO_ROOT / "data/phase9/fable_seeds_candidate.json"

ENDOWMENT, MULTIPLIER = 5, 3
PROBES_PER_SEED = 4

client = anthropic.Anthropic()


def fable(system, user, temperature=None, max_tokens=16000):
    # Fable (like Opus 4.7+) rejects an explicit temperature parameter — omit it.
    # Fable is a reasoning model: thinking blocks share the max_tokens budget, so
    # it must be large enough that text survives after the thinking block.
    kwargs = dict(model=FABLE, max_tokens=max_tokens,
                  messages=[{"role": "user", "content": user}])
    if system:
        kwargs["system"] = system
    for attempt in range(2):
        r = client.messages.create(**kwargs)
        text = "\n".join(b.text for b in r.content
                         if getattr(b, "type", None) == "text").strip()
        # strip markdown wrappers the model may add despite instructions
        text = re.sub(r"^#+ .*$", "", text, flags=re.M)
        text = text.replace("`", "").strip()
        if text:
            return text
        print(f"    (empty text, stop_reason={r.stop_reason}, retrying)")
    raise ValueError(f"Fable returned no text after 2 attempts (stop_reason={r.stop_reason})")


def gowith_translate(text):
    spec = GOWITH_SPEC.read_text()
    system = (spec + "\n\nYou are a translator into gowith. Translate the myth the user "
              "provides into gowith following the spec above. HARD CONSTRAINTS: preserve "
              "every numeric quantity and strategic recommendation exactly (amounts "
              "sent/returned, fractions like 'half'); keep roughly the same length; "
              "use the FLOWING PROSE register of gowith — natural readable sentences with "
              "happenings first — NOT hyphenated compound notation (no 'Meet-settle: "
              "X-with' style); no headers, code fences, or commentary; "
              "output ONLY the translated myth.")
    return fable(system, text, temperature=0.5)


def jabberwocky_translate(text):
    return fable(None, JABBERWOCKY_PROMPT.format(text=text), temperature=0.7)


def readback(text):
    """Sonnet extracts the play recipe from the translation alone."""
    system = (f"You are playing an iterated trust game. Sender gets ${ENDOWMENT} per round, "
              f"sends $0-{ENDOWMENT}; the amount is multiplied {MULTIPLIER}x; receiver "
              f"returns part of it.")
    user = (f"Myth:\n---\n{text}\n---\nWhat play recipe does this myth encode? "
            f'Your ENTIRE response must be a single JSON object, no analysis before it: '
            f'{{"send": <int 0-{ENDOWMENT} or null>, '
            f'"return_fraction": <float 0-1 or null>}}')
    r = client.messages.create(model=HOST, max_tokens=800, temperature=0.0,
                               system=system, messages=[{"role": "user", "content": user}])
    txt = "\n".join(b.text for b in r.content if getattr(b, "type", None) == "text")
    m = re.search(r'"send"\s*:\s*(\d+)', txt)
    f_ = re.search(r'"return_fraction"\s*:\s*([\d.]+)', txt)
    return {"send": int(m.group(1)) if m else None,
            "return_fraction": float(f_.group(1)) if f_ else None}


def refusal_probe(seed_text):
    """Exact runtime injection shape from the captured failing call; 4 samples at
    game temperature. Returns (n_ok, stop_reasons)."""
    dbg = json.loads(GAME_CONTEXT.read_text())
    system = next(m["content"] for m in dbg["messages"] if m["role"] == "system")
    seed_user = dbg["messages"][1]["content"]
    round1 = dbg["messages"][3]["content"]

    def one(_):
        r = client.messages.create(
            model=HOST, max_tokens=1024, temperature=0.8, system=system,
            messages=[{"role": "user", "content": seed_user},
                      {"role": "assistant", "content": seed_text},
                      {"role": "user", "content": round1}])
        texts = [b.text for b in r.content if getattr(b, "type", None) == "text"]
        return r.stop_reason if (r.stop_reason == "refusal" or not texts) else "ok"
    with ThreadPoolExecutor(max_workers=4) as ex:
        outcomes = list(ex.map(one, range(PROBES_PER_SEED)))
    return sum(o == "ok" for o in outcomes), outcomes


def build_pool(source_pool, transform, manifest):
    out = []
    for i, src in enumerate(manifest["seeds"][source_pool]):
        original = src["text"]
        print(f"  [{source_pool}[{i}]] translating ({transform})...", flush=True)
        try:
            text = gowith_translate(original) if transform == "gowith" else jabberwocky_translate(original)
        except ValueError as exc:
            print(f"    TRANSLATOR REFUSED: {exc}")
            out.append({
                "source_pool": source_pool, "source_index": i,
                "transform": transform, "translator_model": FABLE,
                "original_text": original, "text": None,
                "status": "translator_refused", "probe_pass": False,
            })
            continue
        entry = {
            "source_run": src.get("source_run"),
            "source_pool": source_pool,
            "source_index": i,
            "agent_id": src.get("agent_id"),
            "round": src.get("round"),
            "joint_at_source": src.get("joint_at_source"),
            "text": text,
            "tokens": word_count(text),
            "transform": transform,
            "translator_model": FABLE,
            "original_text": original,
        }
        if transform == "gowith":
            entry["readback"] = readback(text)
        else:
            ratio = word_count(text) / max(1, word_count(original))
            entry["word_count_ratio"] = round(ratio, 3)
        n_ok, outcomes = refusal_probe(text)
        entry["probe_ok"] = n_ok
        entry["probe_outcomes"] = outcomes
        entry["probe_pass"] = n_ok == PROBES_PER_SEED
        flag = "PASS" if entry["probe_pass"] else f"REFUSED {PROBES_PER_SEED - n_ok}/{PROBES_PER_SEED}"
        print(f"    probe: {flag}" + (f" | readback: {entry.get('readback')}" if transform == "gowith" else f" | wc ratio {entry.get('word_count_ratio')}"))
        out.append(entry)
    return out


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    n_translate = 15
    n_probe = 15 * PROBES_PER_SEED
    print(f"PREFLIGHT: MODEL={FABLE}(translate)+{HOST}(verify) "
          f"N={n_translate}+{n_probe + 5} EST_COST=$6")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pools = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}

    plan = [("s_end_plus_gowith_fable", "s_end_plus", "gowith"),
            ("s_start_jab_fable", "s_start", "jabberwocky"),
            ("s_end_plus_jab_fable", "s_end_plus", "jabberwocky")]
    def entry_done(s):
        return bool(s.get("text")) or s.get("status") == "translator_refused"

    for key, source, transform in plan:
        existing = pools.get(key, [])
        if len(existing) == len(manifest["seeds"][source]) and all(map(entry_done, existing)):
            print(f"  {key}: already complete in candidates file, skipping")
            continue
        pools[key] = build_pool(source, transform, manifest)
        OUT_PATH.write_text(json.dumps(pools, indent=2))  # incremental save

    print(f"\nCandidates -> {OUT_PATH}")
    for key, seeds in pools.items():
        registrable = [s for s in seeds if s.get("text")]
        manifest["seeds"][key] = registrable
        n_pass = sum(s.get("probe_pass", False) for s in registrable)
        refused = len(seeds) - len(registrable)
        print(f"registered {key}: {len(registrable)} seeds ({refused} translator-refused), "
              f"{n_pass}/{len(registrable)} pass the host refusal probe")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"manifest now has {len(manifest['seeds'])} pools")


if __name__ == "__main__":
    main()

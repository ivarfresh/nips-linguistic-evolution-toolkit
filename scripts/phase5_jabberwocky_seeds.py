"""Phase 5 — produce jabberwocky'd versions of s_start and s_end_plus seeds.

For each of the 5 seeds in each pool, asks Sonnet 4.5 to perform a
"jabberwocky translation": replace every content word (noun, verb, adjective)
with an English-sounding nonsense word while keeping every function word,
punctuation mark, paragraph break, and sentence structure intact.

This isolates semantic content from syntactic/structural form. If the
jabberwocky'd seed still lifts cooperation, structure carries the effect.
If it falls to baseline, semantics carry it.

Writes pools into data/phase3/seed_manifest.json under `seeds.s_start_jab`
and `seeds.s_end_plus_jab`. Stores the original text alongside the
translated text for traceability and reproducibility.
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils import create_llm_client, call_llm

MANIFEST_PATH = REPO_ROOT / "data/phase3/seed_manifest.json"
JUDGE_MODEL = "anthropic/claude-sonnet-4.5"

JABBERWOCKY_PROMPT = """\
You will perform a "jabberwocky translation" of a passage of English text.

RULES:
1. Replace every content word — nouns, verbs (including auxiliary use), adjectives, adverbs — with an English-sounding nonsense word. Use pronounceable, Lewis-Carroll-style invented words (e.g. "brillig", "slithy", "gimble", "wabe"). Each invented word should be plausible as English morphologically (use English-style suffixes like -ed, -ing, -ly where appropriate).
2. Keep every function word EXACTLY as in the original: articles (a, an, the), prepositions (in, on, of, to, with, from, etc.), conjunctions (and, but, or, etc.), pronouns (he, she, it, they, etc.), auxiliaries when not the main verb (will, would, can, could, may, might, must, do, did), and similar grammatical glue words.
3. Keep every punctuation mark, paragraph break, and line break IDENTICAL to the original.
4. Preserve the same sentence structure and approximate word count for each sentence.
5. Be consistent: if you invent a name (e.g. "Brillig"), reuse the same invention if the same word appears multiple times in the original.
6. Do not add any commentary, header, or wrapper around the translated text. Output only the jabberwocky translation.

ORIGINAL TEXT:
\"\"\"
{text}
\"\"\"

JABBERWOCKY TRANSLATION:"""


def word_count(text):
    return len(re.findall(r"\b[a-zA-Z']+\b", text))


def translate(client, text):
    messages = [{"role": "user", "content": JABBERWOCKY_PROMPT.format(text=text)}]
    response = call_llm(client, JUDGE_MODEL, 0.7, messages, max_retries=2)
    content = response.get("content", "").strip()
    # Trim any wrapping quotes or labels the model might have added despite instructions.
    content = re.sub(r'^"""\s*', '', content)
    content = re.sub(r'\s*"""$', '', content)
    content = re.sub(r'^JABBERWOCKY TRANSLATION:\s*', '', content, flags=re.IGNORECASE)
    return content


def jabberwocky_pool(client, source_pool, target_pool_key, manifest):
    """Read source_pool from manifest, translate each, write to target_pool_key."""
    print(f"\n=== Translating {source_pool} → {target_pool_key} ===")
    seeds = manifest["seeds"].get(source_pool, [])
    out = []
    for i, src in enumerate(seeds):
        original = src["text"]
        print(f"  [{i+1}/{len(seeds)}] {src.get('agent_id', '?'):8} R{src.get('round', '?'):>2} ({word_count(original)} words) ...", end=" ", flush=True)
        translated = translate(client, original)
        out_words = word_count(translated)
        print(f"-> {out_words} words")
        out.append({
            "source_run": src.get("source_run"),
            "agent_id": src.get("agent_id"),
            "round": src.get("round"),
            "joint_at_source": src.get("joint_at_source"),
            "text": translated,
            "tokens": out_words,
            "jabberwocky_source_pool": source_pool,
            "jabberwocky_source_index": i,
            "jabberwocky_original_text": original,
            "jabberwocky_translator_model": JUDGE_MODEL,
        })
    return out


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    client = create_llm_client(JUDGE_MODEL)

    cost_per_seed = 0.05
    n = len(manifest["seeds"].get("s_start", [])) + len(manifest["seeds"].get("s_end_plus", []))
    print(f"Preflight: MODEL={JUDGE_MODEL} N={n} EST_COST≈${n * cost_per_seed:.2f}")

    manifest["seeds"]["s_start_jab"] = jabberwocky_pool(client, "s_start", "s_start_jab", manifest)
    manifest["seeds"]["s_end_plus_jab"] = jabberwocky_pool(client, "s_end_plus", "s_end_plus_jab", manifest)

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(manifest['seeds']['s_start_jab'])} s_start_jab seeds")
    print(f"Wrote {len(manifest['seeds']['s_end_plus_jab'])} s_end_plus_jab seeds")


if __name__ == "__main__":
    main()

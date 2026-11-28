"""Render the donor manifest of the myth-injection rerun as a page for reading every text."""
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/phase3/seed_manifest_main_figures_20261127.json"
OUTPUT = ROOT / "docs/research/myth_injection_noise1_20261127/donor_myths.html"
CELLS = [
    ("s_end_plus", "Late Sonnet 4.5 myth, high cooperation", "Round-10 myths from the five highest-ending Sonnet 4.5 myth→game populations."),
    ("s_start", "Early Sonnet 4.5 myth", "Round-1 myths by the same agents in the same five populations."),
    ("s_end_minus", "Late Sonnet 4.5 myth, low cooperation", "Round-10 myths from the five lowest-ending Sonnet 4.5 populations over both task orders."),
    ("s_end_plus_gemini", "Late Gemini 3.7 Flash myth", "Round-10 myths from Gemini 3.7 Flash myth→game populations. All ten ended at 75, so ties are broken by replicate number."),
    ("s_end_plus_gpt", "Late GPT-5 Nano myth, highest cooperation", "Round-10 myths from the five highest-ending GPT-5 Nano myth→game populations."),
    ("s_end_minus_gpt", "Late GPT-5 Nano myth, low cooperation", "Round-10 myths from the five lowest-ending GPT-5 Nano populations over both task orders."),
    ("s_filler", "Filler (not a myth)", "Wikipedia paraphrases of matched length, unchanged from the September batch."),
]
ORDER = {"myth_game": "myth→game", "game_myth": "game→myth"}
STYLE = """:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#66645e;--line:#dcd9d0;--card:#fff;--acc:#8a4b12}
@media(prefers-color-scheme:dark){:root{--bg:#171715;--fg:#e9e7e1;--mut:#a09d94;--line:#35342f;--card:#1f1f1c;--acc:#e0a468}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 Georgia,serif}main{max-width:820px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:1.6rem;margin:.2em 0}h2{font-size:1.25rem;margin:2.2em 0 .2em;padding-top:.6em;border-top:2px solid var(--line)}
p.note,.meta,nav{font-family:system-ui,sans-serif}p.note{color:var(--mut);font-size:.92rem;margin:.3em 0 1em}
nav{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);font-size:.85rem;display:flex;flex-wrap:wrap;gap:6px 14px}nav a{color:var(--acc);text-decoration:none}
article{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 18px;margin:14px 0}
.meta{font-size:.82rem;color:var(--mut);display:flex;flex-wrap:wrap;gap:4px 14px;margin-bottom:8px}.meta b{color:var(--fg)}
.text{white-space:pre-wrap;overflow-wrap:anywhere}code{font-size:.75rem;color:var(--mut);overflow-wrap:anywhere}"""


def main():
    manifest = json.loads(MANIFEST.read_text())
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           f"<title>Myths to inject</title><style>{STYLE}</style></head><body><main><h1>Myths to inject</h1>",
           '<p class="note">Myth-injection rerun at the main-figure noise setting (negative-only up to $1, agents told). '
           "Eight cells, five replicates; the eighth cell injects nothing and is not shown. Each replicate uses one text below, "
           "in both the dyad and the eight-agent batch. Selection rule fixed before any text was read: runs ranked by final "
           "resources per agent (25 to 75), one agent per run by seeded draw (seed 20261127).</p>",
           "<nav>" + "".join(f'<a href="#{k}">{html.escape(t)}</a>' for k, t, _ in CELLS) + "</nav>"]
    for key, title, description in CELLS:
        out.append(f'<h2 id="{key}">{html.escape(title)}</h2><p class="note">{html.escape(description)}</p>')
        for i, entry in enumerate(manifest["seeds"][key]):
            words = len(entry["text"].split())
            if key == "s_filler":
                meta = f'<span><b>Replicate {i}</b></span><span>topic: {html.escape(str(entry.get("topic")))}</span><span>{words} words</span>'
                source = ""
            else:
                meta = (f'<span><b>Replicate {i}</b></span><span>source run ended at <b>{entry["mean_final_per_agent_at_source"]:.1f}</b> per agent</span>'
                        f'<span>{ORDER[entry["source_task_order"]]}, source replicate {entry["source_replicate_id"]}</span>'
                        f'<span>{entry["agent_id"]}, round {entry["round"]}</span><span>{words} words</span>')
                source = f'<div><code>{html.escape(entry["source_run"])}</code></div>'
            out.append(f'<article><div class="meta">{meta}</div><div class="text">{html.escape(entry["text"])}</div>{source}</article>')
    out.append(f'<p class="note">Manifest: {MANIFEST.relative_to(ROOT)}, sha256 {hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}</p></main></body></html>')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(out))
    print(OUTPUT)


if __name__ == "__main__":
    main()

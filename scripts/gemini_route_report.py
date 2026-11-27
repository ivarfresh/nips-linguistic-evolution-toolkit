"""Build the side-by-side HTML report and classifier checks for gemini_route_replay.py output.

Columns: recorded answer (Gemini API, Sept 2026, Gemini 3.7 Flash) | Vertex AI now | Gemini API now.
Classifier check: can a text classifier tell two columns' myths apart? 0.50 = no.
Prompts are the CV groups, so the two answers to one prompt never sit on both sides of a split.

  python scripts/gemini_route_report.py   (reads RUN_DATA_ROOT/data/analysis/gemini_route_replay_20261126)
"""
import html
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from scripts.gemini_route_replay import DROOT, load_call  # noqa: E402

OUT = DROOT / "data/analysis/gemini_route_replay_20261126"
COLS = [("recorded", "Recorded, Sept 2026 (Gemini API)"), ("vertex", "Vertex AI, now"), ("gemini_api", "Gemini API, now")]


def decision(text):
    m = re.search(r'"(send|return)"\s*:\s*([0-9.]+)', text or "")
    return float(m.group(2)) if m else None


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    return f"{st.mean(xs):,.0f} (±{st.pstdev(xs):,.0f})" if len(xs) > 1 else f"{xs[0]:,.0f}"


def classifier(a_texts, b_texts, groups):
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    X = a_texts + b_texts
    y = [0] * len(a_texts) + [1] * len(b_texts)
    g = groups + groups
    cv = GroupKFold(5)
    words = cross_val_score(make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2)), LogisticRegression(max_iter=2000)), X, y, groups=g, cv=cv).mean()
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    E = model.encode(X, normalize_embeddings=True)
    emb = cross_val_score(LogisticRegression(max_iter=2000), E, y, groups=g, cv=cv).mean()
    n = len(a_texts)
    paired = float(sum((E[i] * E[n + i]).sum() for i in range(n)) / n)
    return emb, words, paired


USED_FIELDS = ["candidates[].content.parts[].text (the answer)", "candidates[].finishReason",
               "usageMetadata.promptTokenCount", "usageMetadata.candidatesTokenCount",
               "usageMetadata.thoughtsTokenCount", "modelVersion", "responseId"]


def raw_section(E):
    """Full JSON objects saved by gemini_route_raw_examples.py (and the first pair from 2026-11-26)."""
    d = OUT / "raw_pair"
    if not d.exists():
        return []
    load = lambda f: json.load(open(d / f)) if (d / f).exists() else None  # noqa: E731
    def shorten(o):
        """Replace opaque signatures (thousands of base64 chars) by a placeholder with their length."""
        if isinstance(o, dict):
            return {k: (f"<omitted: {len(v):,} characters>" if k == "thoughtSignature" and isinstance(v, str) else shorten(v))
                    for k, v in o.items()}
        if isinstance(o, list):
            return [shorten(x) for x in o]
        return o

    pretty = lambda o: E(json.dumps(shorten(o), indent=1, ensure_ascii=False))  # noqa: E731
    out = ["<h2 id=raw>Full response objects</h2>",
           "<p class=muted>Exactly what each provider returned, unedited. Both routes received the same request body, "
           "byte for byte (shown once per example). Our code reads only these fields, which both routes return: "
           + ", ".join(f"<code>{E(f)}</code>" for f in USED_FIELDS) + ". Everything else is ignored. "
           "The left column is the September call as stored in our run file (our own record, not Google's raw JSON).</p>"]
    examples = []
    if (d / "2_response_vertex.json").exists():
        examples.append(("Example 1: game move, FULL RESPONSES FROM BOTH ROUTES (captured 2026-11-26)", load("0_recorded_call.json"),
                         (load("1_request_vertex.json") or {}).get("body"),
                         load("2_response_gemini_api.json"), load("2_response_vertex.json")))
    for pid, title in (("founding_myth-000", "First myth"), ("later_myth-000", "Later myth"), ("game-000", "Game move")):
        if (d / f"{pid}_response_vertex.json").exists():
            examples.append((f"{title} ({pid}, captured 2026-11-27), full responses from both routes", load(f"{pid}_recorded.json"),
                             (load(f"{pid}_request.json") or {}).get("body"),
                             load(f"{pid}_response_gemini_api.json"), load(f"{pid}_response_vertex.json")))
    for title, rec, body, api, vtx in examples:
        out.append(f"<div class=item><div class=meta><b>{E(title)}</b></div>")
        out.append(f"<details><summary>Request body sent to both routes</summary><pre>{pretty(body)}</pre></details>")
        out.append("<div class=grid>")
        out.append(f"<div class=col><h4>Recorded, Sept 2026 (our run file)</h4><pre>{pretty(rec)}</pre></div>")
        for label, obj in (("Gemini API, full response", api), ("Vertex AI, full response", vtx)):
            note = ""
            if obj and obj.get("http_error") == 402:
                note = "<div class=tag><span class=bad>Not available: the AI Studio prepaid balance ran out on 2026-11-27.</span></div>"
            out.append(f"<div class=col><h4>{E(label)}</h4>{note}<pre>{pretty(obj)}</pre></div>")
        out.append("</div></div>")
    return out


def main():
    prompts = json.load(open(OUT / "prompts.json"))
    res = {}
    for line in open(OUT / "responses.jsonl"):
        r = json.loads(line)
        if not r["result"].get("error") or (r["id"], r["route"]) not in res:
            res[(r["id"], r["route"])] = r["result"]
    rows = []
    for p in prompts:
        messages, params, rec = load_call(p)
        rows.append({"p": p, "messages": messages, "recorded": rec,
                     "vertex": res.get((p["id"], "vertex")), "gemini_api": res.get((p["id"], "gemini_api"))})

    # summary
    summary = []
    for key, label in COLS:
        got = [r[key] for r in rows if r[key] and not r[key].get("error")]
        versions = {}
        for g in got:
            versions[g.get("model_version")] = versions.get(g.get("model_version"), 0) + 1
        line = {"label": label, "n": len(got), "errors": sum(1 for r in rows if r[key] and r[key].get("error")),
                "versions": versions}
        for kind in ("founding_myth", "later_myth", "game"):
            line[kind] = mean_sd([r[key].get("thinking_tokens") for r in rows if r["p"]["kind"] == kind and r[key] and not r[key].get("error")])
        line["words"] = mean_sd([len((r[key].get("text") or "").split()) for r in rows if "myth" in r["p"]["kind"] and r[key] and not r[key].get("error")])
        summary.append(line)

    game = [r for r in rows if r["p"]["kind"] == "game"]
    agree = {}
    for key in ("vertex", "gemini_api"):
        pairs = [(decision(r["recorded"]["text"]), decision(r[key]["text"])) for r in game if r[key] and not r[key].get("error")]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        agree[key] = (sum(1 for a, b in pairs if abs(a - b) < 0.01), len(pairs),
                      st.mean([abs(a - b) for a, b in pairs]) if pairs else None)

    clf = []
    myth = [r for r in rows if "myth" in r["p"]["kind"]]
    for a, b in (("recorded", "vertex"), ("recorded", "gemini_api"), ("vertex", "gemini_api")):
        both = [r for r in myth if r[a] and r[b] and not r[a].get("error") and not r[b].get("error") and r[a].get("text") and r[b].get("text")]
        if len(both) < 20:
            clf.append((a, b, len(both), None, None, None))
            continue
        emb, words, paired = classifier([r[a]["text"] for r in both], [r[b]["text"] for r in both], [r["p"]["id"] for r in both])
        clf.append((a, b, len(both), emb, words, paired))
    json.dump({"summary": summary, "game_agreement": agree, "classifier": clf}, open(OUT / "summary.json", "w"), indent=1, default=str)

    # html
    E = html.escape
    lab = dict(COLS)
    css = """
    body{background:#fff;color:#1a1a1a;font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;padding:24px 32px}
    h1{font-size:22px;margin:0 0 4px} h2{font-size:18px;margin:28px 0 8px} .muted{color:#666}
    table.s{border-collapse:collapse;margin:8px 0 16px} table.s td,table.s th{border:1px solid #ddd;padding:6px 10px;text-align:left;vertical-align:top}
    table.s th{background:#f6f6f6}
    .item{border:1px solid #ddd;border-radius:6px;margin:18px 0;padding:12px 14px}
    .meta{font-size:13px;color:#555;margin-bottom:8px}
    .grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
    .col{border:1px solid #e5e5e5;border-radius:4px;padding:8px 10px;background:#fff}
    .col h4{margin:0 0 4px;font-size:13px} .tag{font:12px monospace;color:#444;margin-bottom:6px}
    .bad{color:#b00020;font-weight:600} .ok{color:#0a6b2d;font-weight:600}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.45 ui-monospace,Menlo,monospace;margin:0}
    details summary{cursor:pointer;color:#333;font-size:13px} nav a{margin-right:12px}
    """
    out = [f"<!doctype html><html><head><meta charset=utf-8><title>Gemini route replay</title><style>{css}</style></head><body>"]
    out.append("<h1>Gemini 3.7 Flash: recorded vs Vertex AI vs Gemini API</h1>")
    out.append("<p class=muted>Each prompt is a call recorded in the paper's September runs, resent unchanged "
               "(same messages, thinking level high, same temperature and output cap) to both routes on 2026-11-26. "
               "Google redirects <code>gemini-3.7-flash</code> on the Gemini API to Gemini 3.8 Flash since 2026-10-10. "
               "Each column shows the model version the provider reported.</p>")
    out.append("<h2>Summary</h2><table class=s><tr><th>Column</th><th>Answers</th><th>Errors</th><th>Reported model</th>"
               "<th>Thinking tokens: first myth</th><th>later myth</th><th>game move</th><th>Myth words</th></tr>")
    for s in summary:
        v = ", ".join(f"{k}: {n}" for k, n in s["versions"].items())
        out.append(f"<tr><td>{E(s['label'])}</td><td>{s['n']}</td><td>{s['errors']}</td><td>{E(v)}</td>"
                   f"<td>{s['founding_myth']}</td><td>{s['later_myth']}</td><td>{s['game']}</td><td>{s['words']}</td></tr>")
    out.append("</table>")
    out.append("<table class=s><tr><th>Game moves</th><th>Same amount as recorded</th><th>Mean absolute difference ($)</th></tr>")
    for k, (same, n, mad) in agree.items():
        out.append(f"<tr><td>{E(lab[k])}</td><td>{same} of {n}</td><td>{'' if mad is None else f'{mad:.2f}'}</td></tr>")
    out.append("</table>")
    out.append("<h2>Can a classifier tell the myths apart?</h2><p class=muted>0.50 means indistinguishable. "
               "Five-fold cross-validation; the two answers to the same prompt are always in the same fold. "
               "Paired similarity is the mean cosine between the two answers to the same prompt (1 = identical meaning).</p>")
    out.append("<table class=s><tr><th>Comparison</th><th>Myths per side</th><th>Accuracy (meaning)</th><th>Accuracy (words)</th><th>Paired similarity</th></tr>")
    for a, b, n, emb, words, paired in clf:
        cells = ("–", "–", "–") if emb is None else (f"{emb:.2f}", f"{words:.2f}", f"{paired:.3f}")
        out.append(f"<tr><td>{E(lab[a])} vs {E(lab[b])}</td><td>{n}</td><td>{cells[0]}</td><td>{cells[1]}</td><td>{cells[2]}</td></tr>")
    out.append("</table>")
    out.extend(raw_section(E))
    out.append("<nav>Jump to: <a href='#raw'>full response objects</a><a href='#founding_myth'>first myths</a><a href='#later_myth'>later myths</a><a href='#game'>game moves</a></nav>")

    titles = {"founding_myth": "First myths (round 1, written before any game or partner myth)",
              "later_myth": "Later myths (round 3 and after)", "game": "Game moves"}
    for kind in ("founding_myth", "later_myth", "game"):
        out.append(f"<h2 id={kind}>{titles[kind]}</h2>")
        for r in [x for x in rows if x["p"]["kind"] == kind]:
            p = r["p"]
            last_user = next((m["content"] for m in reversed(r["messages"]) if m["role"] == "user"), "")
            sysmsg = next((m["content"] for m in r["messages"] if m["role"] == "system"), "")
            out.append(f"<div class=item><div class=meta><b>{E(p['id'])}</b> · {E(p['set'])} · {E(p['order'])} · round {p['round']}"
                       f"{' · ' + E(p['role']) if p.get('role') else ''} · {E(p['agent'])} · {len(r['messages'])} messages</div>")
            out.append(f"<details><summary>Prompt (last user message; system prompt and history below)</summary><pre>{E(last_user)}</pre>"
                       f"<details><summary>System prompt</summary><pre>{E(sysmsg)}</pre></details></details><div class=grid>")
            for key, label in COLS:
                c = r[key]
                if not c:
                    out.append(f"<div class=col><h4>{E(label)}</h4><span class=muted>not run</span></div>")
                    continue
                if c.get("error"):
                    out.append(f"<div class=col><h4>{E(label)}</h4><span class=bad>{E(c['error'])}</span></div>")
                    continue
                mv = c.get("model_version") or "?"
                cls = "ok" if mv.startswith("gemini-3.7") else "bad"
                out.append(f"<div class=col><h4>{E(label)}</h4><div class=tag><span class={cls}>{E(mv)}</span> · thinking {c.get('thinking_tokens')}"
                           f" · out {c.get('output_tokens')}</div><pre>{E(c.get('text') or '')}</pre></div>")
            out.append("</div></div>")
    out.append("</body></html>")
    path = OUT / "gemini_route_replay.html"
    path.write_text("\n".join(out))
    print(path)
    for s in summary:
        print(s)
    print("game agreement", agree)
    for row in clf:
        print("classifier", row)


if __name__ == "__main__":
    main()

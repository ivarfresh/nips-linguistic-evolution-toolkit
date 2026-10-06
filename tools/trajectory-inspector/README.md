# Myth Reading Room

The evidence inspector now opens with a guided, question-first explanation:
what changed in the story, where it might have come from, and whether it changed
play. Five examples explain borrowing, elaboration, ambiguous attribution and
behavior predating text. T40's clause timeline and T23's recommended return-share
schedule visualize text content, not internal beliefs or observed adherence.
Dialogs link back to original stories. Existing `:plots` links open this clearer
view; raw diagnostics remain available at `:diagnostics`. Notes are untouched.

The optional raw diagnostics retain three read-only plots: own sender/receiver
actions on separate axes with task-order-aware myth markers; contiguous-token
source-match bars for the selected quotation; and a clickable 48 × 10 map of
the 44 selected events. No complexity score, causal network, interpolated action
series or new scientific label is inferred. Missing role opportunities are not
zeros. Source-match bars are literal diagnostics, not semantic similarity.

The separate `followup.html` evidence inspector is **unblinded and read-only**:
48 full trajectories, 44 selected events, previous/exposed/unseen source panels,
quoted clauses and game timing/eligibility. It does not read or write annotation
storage. `build_followup.py` exports only curated texts, selected assessments and
game values; local source paths and raw call histories are excluded. Rebuild from
the research root with `python3 tools/trajectory-inspector/build_followup.py`.
The original packet view and all existing note keys remain unchanged. The
blinding statements below describe only that original packet, not this new view.

Free-form review: each round now has a **My thoughts** box, available beside
both the AI and structured-note views. Temporary browser drafts retain exact
wording. **Prepare all thoughts for chat** collects them with trajectory/round
IDs for copying or downloading; the user must paste/attach that text in chat.
No message, model call, or automatic classification occurs on the Site.
Export schema v3 includes `thoughts` per round. V1/v2 imports remain supported;
a separate v3 storage key preserves the original v2 draft from stale-tab writes.
Refresh older tabs before continuing. Export JSON for a portable backup.

Private static inspector for the twelve-trajectory human validation packet.
No models are called. Research data and machine coding are never modified.

`dist/packet.js` is generated **only** from the blinded `HUMAN_VALIDATION.md`.
The identity key, exposed myths and game outcomes are not bundled. Myths render
as text, never executable HTML, including original Markdown punctuation.
At the user's 2026-10-03 request, a separate `dist/ai-drafts.js` now supplies
AI interpretations of all 120 rounds. The interface is therefore AI-assisted
review, **not independent blinded human validation**.

The AI draft tab and trajectory summary are read-only and explicitly await human
review. Your notes remain separately editable and browser-local; no AI draft
sets human read/completion flags or replaces existing notes. AI downloads carry
provenance and cannot be imported as human notes. Human exports disclose that
AI assistance is available. A malformed/mismatched AI packet disables the AI
view without preventing access to human notes.

Rebuild the curated AI artifact with
`python3 docs/research/strategy_clause_audit_20261002/ai_reading_packet.py`.
The source contains manually authored AI readings of the full frozen packet,
not a keyword classifier or paid model pipeline. Its 841 entries include absence
notes; 923 linked quote instances are verified as exact source substrings.
Exact matching does not establish semantic correctness. Comparison notes also
link the preceding round's action evidence. Original audit labels remain frozen;
broader procedural/fractional refinements in this new reading can yield different
trajectory labels. No cross-round counts are scientific prevalence estimates.

Features: trajectory/round navigation, arbitrary-round comparison, repeatable
evidence entries in six categories (condition, action, exception/recovery,
rationale/expected consequence, change/uncertainty, other evidence), manual read
marks, trajectory interpretation, and JSON export/import tied to the packet
SHA256. Each entry has independent advice/narration/ambiguous checkboxes and
multiple exact quotations with source-round selectors. Advice and narration may
coexist in one entry. Quote matching checks wording, not interpretive validity.
Removal and import replacement require confirmation.

Schema v2 automatically reads v1 drafts/exports. Old field text becomes entries;
the former shared quote and type stay in a separate Other evidence entry, not
invented links on every old field. The original v1 note is also preserved under
`legacy` in exports. A separate v2 localStorage key leaves the original v1 draft
untouched and prevents stale v1 tabs from overwriting v2 notes. Refresh old tabs
before continuing; changes made later in v1 need explicit export/import.

Browser storage is a **temporary local draft**, not a cross-device database.
Exported files are the user's portable review record. Notes do not migrate
between the local preview and hosted origin; export then import to move them.
Storage failures leave the in-memory work intact and show an export warning.
Completing a review is a user-controlled status, not a claim of validated data.

From the research repository root:

```sh
python3 tools/trajectory-inspector/build_packet.py
python3 -m http.server 4318 --bind 127.0.0.1 --directory tools/trajectory-inspector/dist
# In another terminal:
uv run --with playwright python tools/trajectory-inspector/smoke_test.py
node tools/trajectory-inspector/schema_test.cjs
```

The generator needs the research checkout. The static `dist/` output is
self-contained, requires no dependencies/build/server code, and is the
publication artifact. `.openai/hosting.json` identifies the private Site.
Do not change the Site's audience without explicit user direction.

Optional browser WebMCP exposes navigation only, not note writing. Unsupported
browsers continue normally. The app imports no third-party scripts or fonts
and makes no annotation API requests.

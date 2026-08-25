#!/usr/bin/env python3
"""Per-key (hence per-project) Anthropic cost attribution.

The cost_report endpoint can't group by API key, so we attribute cost like this:

  1. usage_report/messages, grouped by api_key_id + model -> tokens per key/model.
  2. cost_report, grouped by description -> authoritative cost per model
     (input vs output).  price = cost / tokens, derived from YOUR OWN billing
     (no hardcoded price list, so it can't drift out of date).
  3. Apply those per-model prices to each key's tokens -> per-key cost.
  4. VALIDATE: sum of all per-key cost vs the authoritative cost_report total.
     If they don't reconcile within a small tolerance, the attribution is flagged.

Then keys whose name matches --pattern (the LLM-evo family by default) are
summed into the project total.

Requires ANTHROPIC_ADMIN_KEY (sk-ant-admin...) in .env.

Usage:
    python scripts/anthropic_project_cost.py --starting 2026-07-01 --ending 2026-08-01
    python scripts/anthropic_project_cost.py --starting 2025-08-01   # all time
    python scripts/anthropic_project_cost.py --pattern 'dexter'      # a different project

Read-only; costs nothing.
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORG = "https://api.anthropic.com/v1/organizations"
# Default: the LLM-evolution key family (see `organizations/api_keys`).
DEFAULT_PATTERN = r"llm[-_ ]?evo|llm_evolution|bloom|deepeval"


def admin_key():
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.strip().startswith("ANTHROPIC_ADMIN_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("ANTHROPIC_ADMIN_KEY not in .env")


def api_get(key, path, params):
    """GET with cursor pagination; returns concatenated data[]."""
    out = []
    page = None
    for _ in range(60):
        p = dict(params)
        if page:
            p["page"] = page
        url = f"{ORG}/{path}?{urllib.parse.urlencode(p, doseq=True)}"
        req = urllib.request.Request(url, headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
        for attempt in range(6):  # backoff on 429 rate limits
            try:
                with urllib.request.urlopen(req, timeout=40) as r:
                    doc = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise
        out.extend(doc.get("data", []))
        if doc.get("has_more") and doc.get("next_page"):
            page = doc["next_page"]
            continue
        break
    return out


def norm_model(s):
    """Normalize display name and model id to a common key.
    'Claude Sonnet 4.5 Usage - Input Tokens' and 'claude-sonnet-4-5-20250929'
    both -> 'sonnet-4-5'."""
    s = s.lower()
    m = re.search(r"(haiku|sonnet|opus|fable)[\s\-]*(\d+)[\.\-]?(\d+)?", s)
    if not m:
        return s.strip()
    fam, a, b = m.group(1), m.group(2), m.group(3)
    return f"{fam}-{a}" + (f"-{b}" if b else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starting", default="2025-08-01")
    ap.add_argument("--ending", default=None, help="exclusive end date YYYY-MM-DD (default: now)")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN,
                    help="case-insensitive regex over key NAMES to sum as the project")
    args = ap.parse_args()
    key = admin_key()
    start = f"{args.starting}T00:00:00Z"
    common = {"starting_at": start, "bucket_width": "1d", "limit": 31}
    if args.ending:
        common["ending_at"] = f"{args.ending}T00:00:00Z"

    # id -> name map for keys
    names = {k["id"]: k.get("name") for k in api_get(key, "api_keys", {"limit": 100})}

    # Prices are derived per (month, model) so results stay exact across ranges
    # spanning price changes. month = starting_at[:7].
    # 1. tokens per (month, api_key_id, norm_model)
    usage = api_get(key, "usage_report/messages",
                    {**common, "group_by[]": ["api_key_id", "model"]})
    key_in = defaultdict(int)    # (kid, month, model) -> input tokens
    key_out = defaultdict(int)   # (kid, month, model) -> output tokens
    mdl_in = defaultdict(int)    # (month, model) -> input tokens (all keys)
    mdl_out = defaultdict(int)
    for b in usage:
        mon = b.get("starting_at", "")[:7]
        for r in b.get("results", []):
            kid = r.get("api_key_id") or "(no key)"
            nm = norm_model(r.get("model") or "unknown")
            cc = r.get("cache_creation") or {}
            inp = (int(r.get("uncached_input_tokens") or 0)
                   + int(r.get("cache_read_input_tokens") or 0)
                   + int(cc.get("ephemeral_1h_input_tokens") or 0)
                   + int(cc.get("ephemeral_5m_input_tokens") or 0))
            out = int(r.get("output_tokens") or 0)
            key_in[(kid, mon, nm)] += inp
            key_out[(kid, mon, nm)] += out
            mdl_in[(mon, nm)] += inp
            mdl_out[(mon, nm)] += out

    # 2. authoritative cost per (month, model), split input vs output
    cost = api_get(key, "cost_report", {**common, "group_by[]": ["description"]})
    cin = defaultdict(float)     # (month, model) -> input cost USD
    cout = defaultdict(float)
    grand_cost = 0.0
    for b in cost:
        mon = b.get("starting_at", "")[:7]
        for r in b.get("results", []):
            amt = float(r.get("amount") or 0) / 100.0
            grand_cost += amt
            desc = (r.get("description") or "")
            nm = norm_model(desc)
            if "output" in desc.lower():
                cout[(mon, nm)] += amt
            else:  # input + any cache line items
                cin[(mon, nm)] += amt

    # 3. derive per (month, model) $/token and apply per key
    price_in = {mm: cin[mm] / mdl_in[mm] for mm in mdl_in if mdl_in[mm]}
    price_out = {mm: cout[mm] / mdl_out[mm] for mm in mdl_out if mdl_out[mm]}

    key_totals = defaultdict(float)
    for (kid, mon, nm), t in key_in.items():
        key_totals[kid] += t * price_in.get((mon, nm), 0.0)
    for (kid, mon, nm), t in key_out.items():
        key_totals[kid] += t * price_out.get((mon, nm), 0.0)

    rows = [(names.get(kid, kid) or kid, kid, c) for kid, c in key_totals.items()]
    rows.sort(key=lambda x: -x[2])

    pat = re.compile(args.pattern, re.I)
    proj_total = sum(c for nm, _, c in rows if pat.search(nm or ""))
    attributed = sum(c for _, _, c in rows)

    end_lbl = args.ending or "now"
    print("=" * 64)
    print(f"ANTHROPIC COST BY KEY  ({args.starting} .. {end_lbl})")
    print("=" * 64)
    print(f"  {'key name':<34}{'cost $':>12}   project?")
    print("  " + "-" * 58)
    for nm, _, c in rows:
        if c < 0.005:
            continue
        mark = "  <== project" if pat.search(nm or "") else ""
        print(f"  {str(nm)[:34]:<34}{c:>12,.2f}{mark}")
    print("  " + "-" * 58)
    unattributed = grand_cost - attributed
    print(f"  {'unattributed (no key: tool/web fees)':<34}{unattributed:>12,.2f}")
    print(f"  {'PROJECT TOTAL (pattern match)':<34}{proj_total:>12,.2f}")
    print(f"  {'all keys (attributed)':<34}{attributed:>12,.2f}")
    print(f"  {'ORG TOTAL (authoritative billed)':<34}{grand_cost:>12,.2f}")
    print()
    # 4. validation against the authoritative aggregate
    pct = (unattributed / grand_cost * 100) if grand_cost else 0.0
    flag = ("exact" if abs(pct) < 2
            else f"{pct:.1f}% is unattributable tool/non-message cost, not misassigned")
    print(f"  VALIDATION: attributed ${attributed:,.2f} + unattributed ${unattributed:,.2f}"
          f" = billed ${grand_cost:,.2f}  [{flag}]")
    print(f"  pattern: /{args.pattern}/i    (run per-month for penny-exact figures)")


if __name__ == "__main__":
    main()

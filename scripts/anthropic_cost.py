#!/usr/bin/env python3
"""Anthropic (Claude) org cost report — the source of truth for BYOK spend.

The OpenRouter BYOK figure only meters what routed through OpenRouter and
loses spend from deleted keys. This hits Anthropic's own Cost API, which
reports the whole organization's charges directly.

Requires an ADMIN key (sk-ant-admin...), created by an org Owner at
  https://platform.claude.com/settings/admin-keys
Put it in .env as:  ANTHROPIC_ADMIN_KEY=sk-ant-admin...

Usage:
    python scripts/anthropic_cost.py                       # since 2025-08-01, daily
    python scripts/anthropic_cost.py --starting 2026-01-01 # custom start
    python scripts/anthropic_cost.py --raw                 # dump raw JSON

Read-only GET; costs nothing.
"""
import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE = "https://api.anthropic.com/v1/organizations/cost_report"
ROOT = Path(__file__).resolve().parent.parent


def load_env_key(name):
    env = ROOT / ".env"
    if not env.exists():
        sys.exit(f"No .env at {env}")
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def fetch(key, starting, ending):
    params = {"starting_at": starting, "bucket_width": "1d", "limit": 31}
    if ending:
        params["ending_at"] = ending
    out = []
    page = None
    for _ in range(50):  # safety cap on pagination
        q = dict(params)
        if page:
            q["page"] = page
        req = urllib.request.Request(
            f"{BASE}?{urllib.parse.urlencode(q)}",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                doc = json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:400]
            sys.exit(f"HTTP {e.code} from cost_report:\n{body}")
        out.extend(doc.get("data", []))
        if doc.get("has_more") and doc.get("next_page"):
            page = doc["next_page"]
            continue
        break
    return out


def amount_of(result):
    """Return cost in USD. The cost_report `amount` field is in CENTS
    (verified against a Console CSV export: API 17455.62 == $174.56), so
    divide by 100."""
    for f in ("amount", "cost", "value", "total"):
        if f in result and result[f] is not None:
            try:
                return float(result[f]) / 100.0
            except (TypeError, ValueError):
                pass
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starting", default="2025-08-01")
    ap.add_argument("--ending", default=None)
    ap.add_argument("--raw", action="store_true", help="print raw JSON and exit")
    args = ap.parse_args()

    key = load_env_key("ANTHROPIC_ADMIN_KEY")
    if not key:
        sys.exit("ANTHROPIC_ADMIN_KEY not found in .env. Create an Admin key "
                 "(org Owner only) at https://platform.claude.com/settings/admin-keys")
    if not key.startswith("sk-ant-admin"):
        print("WARN: key does not look like an admin key (sk-ant-admin...); "
              "the cost endpoint will likely 401.", file=sys.stderr)

    start = f"{args.starting}T00:00:00Z" if "T" not in args.starting else args.starting
    end = None
    if args.ending:
        end = f"{args.ending}T00:00:00Z" if "T" not in args.ending else args.ending

    buckets = fetch(key, start, end)

    if args.raw:
        print(json.dumps(buckets, indent=2)[:8000])
        return

    grand = 0.0
    by_model = defaultdict(float)
    currency = "USD"
    print("=" * 60)
    print(f"ANTHROPIC ORG COST  (from {args.starting})")
    print("=" * 60)
    for b in buckets:
        for res in b.get("results", []):
            amt = amount_of(res)
            grand += amt
            label = (res.get("model") or res.get("description")
                     or res.get("workspace_id") or "unknown")
            by_model[label] += amt
            currency = res.get("currency", currency)
    if not buckets:
        print("  (no data returned for this range)")
        return
    print(f"  {'model / group':<40}{'cost':>14}")
    print("  " + "-" * 54)
    for label, amt in sorted(by_model.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {str(label)[:40]:<40}{amt:>12,.2f} {currency}")
    print("  " + "-" * 54)
    print(f"  {'TOTAL':<40}{grand:>12,.2f} {currency}")
    print("\n  Note: verify units — if this looks 100x off, amounts are in cents.")


if __name__ == "__main__":
    main()

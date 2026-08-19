#!/usr/bin/env python3
"""OpenRouter spend report for this project.

Reads OPENROUTER_MANAGEMENT_KEY from .env (a management key — the runtime
OPENROUTER_API_KEY cannot fetch account-wide data) and prints:

  1. Account credit balance (total_credits / total_usage).
  2. Per-key all-time spend for keys whose name matches --pattern
     (default "LLM_evo"), split into OpenRouter-billed credit usage and
     BYOK usage (routed through your own provider key), plus the combined
     all-time total.
  3. Recent per-day x per-model activity breakdown (OpenRouter's /activity
     endpoint only covers a recent window).

Usage:
    python scripts/openrouter_spend.py                 # LLM_evo keys
    python scripts/openrouter_spend.py --pattern ''    # all keys
    python scripts/openrouter_spend.py --no-activity   # skip daily breakdown

All calls are read-only GETs and cost nothing.
"""
import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE = "https://openrouter.ai/api/v1"
ROOT = Path(__file__).resolve().parent.parent


def load_env_key(name):
    """Read a single KEY=value from .env without importing dotenv."""
    env = ROOT / ".env"
    if not env.exists():
        sys.exit(f"No .env at {env}")
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def get(path, key):
    req = urllib.request.Request(
        f"{BASE}/{path}", headers={"Authorization": f"Bearer {key}"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def money(x):
    return f"${float(x or 0):>10,.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="LLM_evo",
                    help="case-insensitive substring to match key names (default: LLM_evo; '' = all)")
    ap.add_argument("--no-activity", action="store_true", help="skip the recent daily breakdown")
    args = ap.parse_args()

    mkey = load_env_key("OPENROUTER_MANAGEMENT_KEY")
    if not mkey:
        sys.exit("OPENROUTER_MANAGEMENT_KEY not found in .env. Create one at "
                 "https://openrouter.ai/settings/management-keys")

    # 1. Account credits (runtime key works too, but the management key is fine).
    credits = get("credits", mkey).get("data", {})
    tc = credits.get("total_credits")
    tu = credits.get("total_usage")
    print("=" * 66)
    print("ACCOUNT CREDITS")
    print("=" * 66)
    print(f"  purchased : {money(tc)}")
    print(f"  used      : {money(tu)}")
    if tc is not None and tu is not None:
        print(f"  remaining : {money(tc - tu)}")

    # 2. Per-key all-time spend.
    keys = get("keys?include_disabled=true", mkey).get("data", [])
    pat = args.pattern.lower()
    matched = [k for k in keys if pat in (k.get("name") or "").lower()]

    print()
    print("=" * 66)
    print(f"KEYS MATCHING {args.pattern!r}  ({len(matched)} of {len(keys)} total)")
    print("=" * 66)
    hdr = f"  {'name':<22}{'credit $':>12}{'byok $':>12}{'combined $':>14}  {'state':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    tot_c = tot_b = 0.0
    for k in sorted(matched, key=lambda r: (r.get("usage") or 0) + (r.get("byok_usage") or 0), reverse=True):
        c = float(k.get("usage") or 0)
        b = float(k.get("byok_usage") or 0)
        tot_c += c
        tot_b += b
        state = "disabled" if k.get("disabled") else "active"
        print(f"  {str(k.get('name'))[:22]:<22}{c:>12,.2f}{b:>12,.2f}{c + b:>14,.2f}  {state:>8}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL (all time)':<22}{tot_c:>12,.2f}{tot_b:>12,.2f}{tot_c + tot_b:>14,.2f}")
    print()
    print("  credit $ = billed from your OpenRouter credit balance")
    print("  byok $   = metered spend on your own provider key routed via OpenRouter")

    # 3. Recent daily x model breakdown.
    if not args.no_activity:
        act = get("activity", mkey).get("data", [])
        if act:
            dates = sorted({r.get("date", "")[:10] for r in act})
            span = f"{dates[0]} .. {dates[-1]}" if dates else "n/a"
            by = defaultdict(lambda: [0.0, 0.0, 0])  # model -> [usage, byok, requests]
            for r in act:
                m = r.get("model") or r.get("model_permaslug")
                by[m][0] += float(r.get("usage") or 0)
                by[m][1] += float(r.get("byok_usage_inference") or 0)
                by[m][2] += int(r.get("requests") or 0)
            print()
            print("=" * 66)
            print(f"RECENT ACTIVITY BY MODEL  (window: {span})")
            print("=" * 66)
            print(f"  {'model':<34}{'credit $':>10}{'byok $':>10}{'reqs':>7}")
            print("  " + "-" * 60)
            for m, (u, b, n) in sorted(by.items(), key=lambda kv: kv[1][0] + kv[1][1], reverse=True):
                print(f"  {str(m)[:34]:<34}{u:>10,.2f}{b:>10,.2f}{n:>7}")
            print("  (OpenRouter's /activity only covers a recent window, not all time)")


if __name__ == "__main__":
    main()

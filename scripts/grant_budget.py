"""How much of the Cooperative AI compute grant has the project spent, and what is left?

Pulls billed costs live from the Anthropic and OpenAI admin cost endpoints,
adds OpenRouter credit spend and the Google (Gemini) bill, converts USD to EUR
at each month's average ECB rate, and compares the total with the grant in
config/grant_budget.yaml. Spend is attributed per API key so keys from other
projects (Claude Code, Dexter, ...) are left out.

Secrets are read from the macOS Keychain (or the environment), never from the
repo: ANTHROPIC_ADMIN_KEY, OPENAI_ADMIN_KEY, OPENROUTER_MANAGEMENT_KEY. Store
one from the clipboard with
  security add-generic-password -U -a "$USER" -s <NAME> -w "$(pbpaste)"

Usage:
  python scripts/grant_budget.py                    # spent / remaining
  python scripts/grant_budget.py --plan-usd 150     # ... and after a planned run
  python scripts/grant_budget.py --snapshot-openrouter   # once, to start live OpenRouter tracking
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.parse
import urllib.request

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "grant_budget.yaml"


# ---------------------------------------------------------------- helpers

def secret(name):
    if os.environ.get(name):
        return os.environ[name]
    r = subprocess.run(["security", "find-generic-password", "-s", name, "-w"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def get_json(url, params=None, headers=None):
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers={"User-Agent": "grant-budget/1.0", **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def iso(d):
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def usd_to_eur_rates(start, end):
    """Average ECB USD->EUR rate per month (frankfurter.app), plus the latest rate."""
    body = get_json(f"https://api.frankfurter.app/{start.isoformat()}..{end.isoformat()}",
                    {"from": "USD", "to": "EUR"})
    by_month = defaultdict(list)
    for day, rates in body["rates"].items():
        by_month[day[:7]].append(rates["EUR"])
    monthly = {m: sum(v) / len(v) for m, v in by_month.items()}
    latest = get_json("https://api.frankfurter.app/latest", {"from": "USD", "to": "EUR"})["rates"]["EUR"]
    return monthly, latest


class Tally:
    """USD spend per (bucket, month), where bucket is project / other / unclassified."""

    def __init__(self):
        self.usd = defaultdict(lambda: defaultdict(float))
        self.labels = defaultdict(float)

    def add(self, bucket, label, month, usd):
        self.usd[bucket][month] += usd
        self.labels[(bucket, label)] += usd


def classify(key_id, cfg):
    if key_id in cfg.get("project_keys", {}):
        return "project", cfg["project_keys"][key_id]
    if key_id in cfg.get("other_keys", {}):
        return "other", cfg["other_keys"][key_id]
    return "unclassified", key_id or "(no key: console/playground)"


# ---------------------------------------------------------------- Anthropic

# cost_report token_type -> how to read that token count from a usage_report row
TOKEN_FIELDS = {
    "uncached_input_tokens": lambda r: r.get("uncached_input_tokens", 0),
    "cache_read_input_tokens": lambda r: r.get("cache_read_input_tokens", 0),
    "output_tokens": lambda r: r.get("output_tokens", 0),
    "cache_creation.ephemeral_5m_input_tokens": lambda r: (r.get("cache_creation") or {}).get("ephemeral_5m_input_tokens", 0),
    "cache_creation.ephemeral_1h_input_tokens": lambda r: (r.get("cache_creation") or {}).get("ephemeral_1h_input_tokens", 0),
}


def anthropic_pages(path, params, key):
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    params = dict(params)
    while True:
        body = get_json("https://api.anthropic.com/v1/organizations/" + path, params, headers)
        yield from body["data"]
        if not body.get("has_more"):
            return
        params["page"] = body["next_page"]


def anthropic_spend(cfg, start, end, tally):
    """The cost endpoint has no per-key split, so each day's cost for one
    (model, tier, context window, token type) is shared out over API keys in
    proportion to their tokens of that same type from the usage endpoint."""
    key = secret("ANTHROPIC_ADMIN_KEY")
    if not key:
        return "ANTHROPIC_ADMIN_KEY not in Keychain or environment"
    window = {"starting_at": iso(start), "ending_at": iso(end), "limit": 31}

    usage = defaultdict(lambda: defaultdict(float))   # (day, model, tier, ctx, type) -> key -> tokens
    searches = defaultdict(lambda: defaultdict(float))  # day -> key -> web searches
    for bucket in anthropic_pages("usage_report/messages", {
            **window, "bucket_width": "1d",
            "group_by[]": ["api_key_id", "model", "service_tier", "context_window"]}, key):
        day = bucket["starting_at"][:10]
        for row in bucket["results"]:
            ident = (day, row.get("model"), row.get("service_tier"), row.get("context_window"))
            for ttype, read in TOKEN_FIELDS.items():
                if read(row):
                    usage[ident + (ttype,)][row.get("api_key_id")] += read(row)
            n = (row.get("server_tool_use") or {}).get("web_search_requests", 0)
            if n:
                searches[day][row.get("api_key_id")] += n

    for bucket in anthropic_pages("cost_report", {**window, "group_by[]": ["description"]}, key):
        day = bucket["starting_at"][:10]
        for row in bucket["results"]:
            usd = float(row["amount"]) / 100  # cents -> dollars
            if row.get("cost_type") == "tokens":
                shares = usage.get((day, row.get("model"), row.get("service_tier"),
                                    row.get("context_window"), row.get("token_type")))
            elif row.get("cost_type") == "web_search":
                shares = searches.get(day)
            else:
                shares = None
            total = sum(shares.values()) if shares else 0
            if not total:
                tally.add("unclassified", f"unmatched: {row.get('description')}", day[:7], usd)
                continue
            for key_id, n in shares.items():
                bucket_name, label = classify(key_id, cfg)
                tally.add(bucket_name, label, day[:7], usd * n / total)
    return None


# ---------------------------------------------------------------- OpenAI

def openai_spend(cfg, start, end, tally):
    key = secret("OPENAI_ADMIN_KEY")
    if not key:
        return "OPENAI_ADMIN_KEY not in Keychain or environment"
    headers = {"Authorization": f"Bearer {key}"}
    params = {"start_time": int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp()),
              "end_time": int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp()),
              "group_by": ["api_key_id"], "limit": 180}
    while True:
        body = get_json("https://api.openai.com/v1/organization/costs", params, headers)
        for bucket in body["data"]:
            month = datetime.fromtimestamp(bucket["start_time"], timezone.utc).strftime("%Y-%m")
            for row in bucket["results"]:
                bucket_name, label = classify(row.get("api_key_id"), cfg)
                tally.add(bucket_name, label, month, float(row["amount"]["value"]))
        if not body.get("has_more"):
            return None
        params["page"] = body["next_page"]


# ---------------------------------------------------------------- OpenRouter

def openrouter_total_usage():
    key = secret("OPENROUTER_MANAGEMENT_KEY")
    if not key:
        return None
    body = get_json("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {key}"})
    return float(body["data"]["total_usage"])


def openrouter_spend(cfg, tally):
    for month, usd in cfg["fixed_usd_by_month"].items():
        tally.add("project", "credits (Activity page)", str(month), float(usd))
    snap = cfg.get("snapshot")
    if not snap:
        return (f"live tracking not started: counted only up to {cfg['fixed_until']}. "
                "Run --snapshot-openrouter once.")
    now = openrouter_total_usage()
    if now is None:
        return (f"OPENROUTER_MANAGEMENT_KEY missing: spend since {snap['taken_at']} not counted")
    tally.add("project", "credits (since snapshot)", date.today().strftime("%Y-%m"),
              now - float(snap["total_usage_usd"]))
    return None


def snapshot_openrouter():
    total = openrouter_total_usage()
    if total is None:
        sys.exit("OPENROUTER_MANAGEMENT_KEY not in Keychain or environment")
    taken = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = CONFIG.read_text()
    new = (f"snapshot:\n    taken_at: \"{taken}\"\n    total_usage_usd: {total:.6f}"
           f"   # lifetime credit usage at that moment")
    text, n = re.subn(r"snapshot: null[^\n]*", new, text)
    if n != 1:
        sys.exit("config already has an OpenRouter snapshot; edit it by hand to reset")
    CONFIG.write_text(text)
    print(f"OpenRouter snapshot stored: ${total:.2f} lifetime credit usage at {taken}")


# ---------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    plan_group = ap.add_mutually_exclusive_group()
    plan_group.add_argument("--plan-usd", type=float, help="estimated cost of a planned run, in USD")
    plan_group.add_argument("--plan-eur", type=float, help="estimated cost of a planned run, in EUR")
    ap.add_argument("--snapshot-openrouter", action="store_true",
                    help="record current OpenRouter lifetime usage to start live tracking")
    ap.add_argument("--detail", action="store_true", help="show spend per API key")
    args = ap.parse_args()

    if args.snapshot_openrouter:
        return snapshot_openrouter()

    cfg = yaml.safe_load(CONFIG.read_text())
    start = date.fromisoformat(cfg["count_from"])
    end = date.today() + timedelta(days=1)  # exclusive; includes today so far

    per_provider = {}
    notes = []
    for name, fn in [("Claude (Anthropic)", lambda t: anthropic_spend(cfg["anthropic"], start, end, t)),
                     ("OpenAI", lambda t: openai_spend(cfg["openai"], start, end, t)),
                     ("OpenRouter credits", lambda t: openrouter_spend(cfg["openrouter"], t))]:
        tally = Tally()
        note = fn(tally)
        if note:
            notes.append(f"{name}: {note}")
        per_provider[name] = tally
    missing = [n for n in notes if "ADMIN_KEY not in Keychain" in n]
    if missing:
        # A provider without its admin key would count as EUR 0 and overstate what is left.
        raise SystemExit("Cannot compute the budget; missing admin key(s):\n  " + "\n  ".join(missing))
    incomplete = [n for n in notes if n.startswith("OpenRouter")]

    rates, latest = usd_to_eur_rates(start, date.today())

    def eur(month_usd):
        return sum(usd * rates.get(m, latest) for m, usd in month_usd.items())

    grant = sum(g["amount_eur"] for g in cfg["grants"])
    rows = []
    unclassified = []
    excluded_eur = 0.0
    for name, t in per_provider.items():
        project = eur(t.usd["project"]) + eur(t.usd["unclassified"])
        rows.append((name, project))
        excluded_eur += eur(t.usd["other"])
        unclassified += [(name, label, usd) for (b, label), usd in t.labels.items()
                         if b == "unclassified" and usd >= 0.005]
    rows.append((f"Google Gemini (entered by hand, as of {cfg['google']['as_of']})",
                 float(cfg["google"]["amount_eur"])))
    spent = sum(v for _, v in rows)
    left = grant - spent

    print(f"\nCooperative AI grant budget, spend from {start} to {date.today()} (EUR)\n")
    for name, v in rows:
        print(f"  {name:<58} €{v:>9,.2f}")
    print(f"  {'-' * 70}")
    print(f"  {'Spent':<58} €{spent:>9,.2f}")
    print(f"  {'Grant received':<58} €{grant:>9,.2f}")
    flag = "   ** INCOMPLETE: OpenRouter spend after the fixed months is not counted (see NOTE) **" if incomplete else ""
    print(f"  {'Left':<58} €{left:>9,.2f}   ({spent / grant:.0%} used){flag}")
    print(f"\n  Not counted (other projects' keys): €{excluded_eur:,.2f}")

    if args.plan_usd is not None or args.plan_eur is not None:
        plan = args.plan_eur if args.plan_eur is not None else args.plan_usd * latest
        after = left - plan
        print(f"\n  Planned run: €{plan:,.2f}"
              + (f" (${args.plan_usd:,.2f} at {latest:.4f} EUR/USD)" if args.plan_usd is not None else ""))
        print(f"  Left after the run: €{after:,.2f}" + ("   ** OVER BUDGET **" if after < 0 else "") + ("   (at most; see NOTE)" if incomplete else ""))

    if args.detail:
        print("\n  Spend per key (USD, all months):")
        for name, t in per_provider.items():
            for (b, label), usd in sorted(t.labels.items(), key=lambda kv: -kv[1]):
                if usd >= 0.005:
                    print(f"    {name:<20} {b:<13} ${usd:>9,.2f}  {label}")

    if unclassified:
        print("\n  UNCLASSIFIED spend (counted against the grant; add these keys to "
              "config/grant_budget.yaml):")
        for name, label, usd in unclassified:
            print(f"    {name}: ${usd:,.2f}  {label}")
    for note in notes:
        print(f"\n  NOTE {note}")
    print()


if __name__ == "__main__":
    main()

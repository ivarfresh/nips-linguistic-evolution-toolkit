"""Build the S-filler text pool from Simple English Wikipedia.

§17.4: length-matched factual encyclopedia prose — country/animal/element
articles. Pulls first paragraphs from a fixed list of Simple English
Wikipedia articles via the public REST API. Output:
data/share/phase2_filler_paragraphs.json
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

SLUGS = [
    "Belgium", "Norway", "Brazil", "Japan", "Egypt", "Canada", "Vietnam",
    "Argentina", "Finland", "Kenya", "Iceland", "Thailand",
    "Octopus", "Elephant", "Penguin", "Salamander", "Cheetah", "Owl",
    "Dolphin", "Beetle", "Hummingbird", "Wolf", "Falcon", "Tortoise",
    "Carbon", "Helium", "Iron", "Silicon", "Gold", "Hydrogen",
    "Copper", "Nitrogen", "Calcium", "Mercury_(element)", "Lithium",
    "Sodium",
]


def fetch(slug):
    url = f"https://simple.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(slug, safe='_')}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "phase2-filler-fetcher/1.0 (ivarfrisch@gmail.com)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def main():
    out_dir = Path("data/share")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase2_filler_paragraphs.json"
    pool = []
    for slug in SLUGS:
        try:
            data = fetch(slug)
        except Exception as exc:
            print(f"[skip] {slug}: {exc}")
            continue
        text = (data.get("extract") or "").strip()
        if not text:
            continue
        pool.append(
            {
                "source": f"simple.wikipedia.org/wiki/{slug}",
                "slug": slug,
                "text": text,
                "tokens": len(text.split()),
            }
        )
    pool.sort(key=lambda x: x["tokens"])
    with open(out_path, "w") as f:
        json.dump(pool, f, indent=2)
    print(f"Wrote {len(pool)} filler paragraphs to {out_path}")
    if pool:
        ns = [p["tokens"] for p in pool]
        print(
            f"Token-count distribution: min={min(ns)} max={max(ns)} "
            f"median={sorted(ns)[len(ns)//2]}"
        )


if __name__ == "__main__":
    main()

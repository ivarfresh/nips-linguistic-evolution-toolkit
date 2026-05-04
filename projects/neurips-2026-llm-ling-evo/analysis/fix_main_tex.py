#!/usr/bin/env python3
"""Post-process main.tex to convert Quarto's CSL inline citations into
natbib \\citep{...} commands and strip the CSL-rendered bibliography
(replaced by \\bibliography{references}).

Run after rebuild_overleaf.py; updates manuscript/main.tex,
overleaf-main.tex, and overleaf-upload/main.tex in place.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT = REPO_ROOT / "projects" / "neurips-2026-llm-ling-evo"
MANUSCRIPT = PROJECT / "manuscript"
REFS_JSON = PROJECT / "references" / "bib"  # one file per record
MAIN_TEX_PATHS = [
    MANUSCRIPT / "main.tex",
    MANUSCRIPT / "overleaf-main.tex",
    MANUSCRIPT / "overleaf-upload" / "main.tex",
]


def build_author_year_index() -> dict:
    """Build mapping from (last name lowercase, year) -> bib key.

    Pulls from references/bib/*.json (CSL JSON, one record per file).
    """
    index = {}
    for f in REFS_JSON.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            records = data
        else:
            records = [data]
        for rec in records:
            key = rec.get("id")
            if not key:
                continue
            authors = rec.get("author") or []
            if not authors:
                continue
            family = authors[0].get("family", "")
            if not family:
                continue
            issued = rec.get("issued", {}).get("date-parts", [[None]])
            year = issued[0][0] if issued and issued[0] else None
            if year is None:
                continue
            # Index by (lowercase family, year).
            index[(family.lower(), int(year))] = key
            # Also index by simpler form for variants like "et al"
            index.setdefault((family.lower().split("-")[0], int(year)), key)
    return index


def convert_citations(text: str, author_year: dict) -> tuple[str, list]:
    """Convert Pandoc/CSL-rendered inline citations to \\citep{key1, key2}.

    Patterns handled:
      {[}Smith, 2020{]}
      {[}Smith and Jones, 2020{]}
      {[}Smith et al., 2020{]}
      {[}Smith, 2020; Jones, 2021{]}
      {[}e.g.~Smith, 2020{]}
      {[}Smith et al., 2020a{]}  (year suffixes)
    """
    unmatched = []

    def lookup(author: str, year: int) -> str | None:
        author = author.lower().strip()
        # Strip 'et al' / 'and Jones' etc. for first-author lookup
        author = re.split(r" et al| and |, ", author, maxsplit=1)[0].strip()
        return author_year.get((author, year))

    def replace(match: re.Match) -> str:
        body = match.group(1)
        # Collapse whitespace (Pandoc may have wrapped citations across lines)
        body = re.sub(r"\s+", " ", body).strip()

        # Skip purely numeric content like CIs ("+0.22, +0.96") — these
        # have no alphabetic content before the first comma.
        first_chunk = body.split(";")[0]
        # If the first chunk has no alphabetic characters before the year
        # pattern, it's not a citation. Also require a 4-digit year somewhere.
        if not re.search(r"[A-Za-z]{3,}", first_chunk):
            return match.group(0)
        if not re.search(r"\b(19|20)\d{2}\b", body):
            return match.group(0)
        # Skip if this looks like an inline-code placeholder (single-word
        # body without comma — e.g., {[}ag{]} from \texttt{...[ag]...}).
        if "," not in body and len(body) < 8:
            return match.group(0)

        # Strip "e.g.~" / "see ~" / "cf. ~" / similar prefixes
        body = re.sub(r"^(?:e\.g\.~?|see~?|cf\.~?|c\.f\.~?)\s*", "", body)
        # Split on semicolons.
        keys = []
        for chunk in body.split(";"):
            chunk = chunk.strip()
            # Trailing year suffix like 2020a -> grab as 2020 with suffix
            m = re.match(r"(.+?),\s*(\d{4})([a-z])?$", chunk)
            if not m:
                unmatched.append(chunk)
                return match.group(0)  # leave unchanged
            author = m.group(1).strip()
            year = int(m.group(2))
            suffix = m.group(3) or ""
            key = lookup(author, year)
            if key is None and suffix:
                # Try without suffix
                key = lookup(author, year)
            if key is None:
                unmatched.append(f"{author}, {year}{suffix}")
                return match.group(0)
            keys.append(key)
        if not keys:
            return match.group(0)
        return r"\citep{" + ", ".join(keys) + "}"

    # Match {[}...{]} style with author-year content. Allow newlines inside.
    new_text = re.sub(
        r"\{\[\}([^{}\[\]]+?)\{\]\}",
        replace,
        text,
        flags=re.DOTALL,
    )
    return new_text, unmatched


def strip_csl_bibliography(text: str) -> str:
    """Remove the CSL-rendered References section.

    The CSL-rendered references block sits between
    \\subsection*{References} and \\end{CSLReferences}, with optional
    surrounding markup. Stop at \\end{CSLReferences} so the appendix
    (which follows the references in the include order) is preserved.
    """
    pattern = re.compile(
        r"\\subsection\*\{References\}.*?\\end\{CSLReferences\}\s*",
        re.DOTALL,
    )
    return pattern.sub("", text)


def fix_section_numbering(text: str) -> str:
    """The c_final files use Markdown headers like '# 4. Results' which
    Quarto renders as '\\section{4. Results}'. With \\setcounter
    {secnumdepth}{-1}, LaTeX won't auto-number, so the explicit '4.' is
    correct. Leave this alone — the prefixes in the source are intentional.
    """
    return text


def main():
    print(">> Building author-year -> bib key index ...")
    author_year = build_author_year_index()
    print(f"   {len(author_year)} (author, year) entries indexed")

    for path in MAIN_TEX_PATHS:
        if not path.exists():
            continue
        print(f"\n>> Processing {path}")
        text = path.read_text(encoding="utf-8")

        text2, unmatched = convert_citations(text, author_year)
        n_converted = text.count("{[}") - text2.count("{[}")
        print(f"   converted {n_converted} citation groups to \\citep{{}}")
        if unmatched:
            print(f"   {len(unmatched)} unmatched citation chunks (left unchanged):")
            for u in unmatched[:8]:
                print(f"     - {u}")
            if len(unmatched) > 8:
                print(f"     ... and {len(unmatched) - 8} more")

        text3 = strip_csl_bibliography(text2)
        if text3 != text2:
            print("   stripped CSL-rendered bibliography section")

        text3 = fix_section_numbering(text3)

        path.write_text(text3, encoding="utf-8")
        print(f"   wrote {path}")


if __name__ == "__main__":
    main()

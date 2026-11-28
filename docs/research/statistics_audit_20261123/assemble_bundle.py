#!/usr/bin/env python3
"""Package the audited source and results; no scientific reruns or model calls."""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAPER = ROOT.parent / "overleaf-paper" / "overleaf-upload"
ARCHIVE = HERE / "statistics_audit_code_and_results.zip"

CLAIMS = [
    ("4.4 founding amount, 71%, 12/12, original normal CI", "docs/research/myth_predictors_20260930/amount/r4_reconcile.py", 6),
    ("4.4 portable opening replication and cluster-t alternative", "docs/research/frontier_myth_predictors_20260930/plan/analyze.py", 130),
    ("4.4 opening adjustment: $0.42 to $0.17", "analyses/myth_replay_analysis.py", 72),
    ("4.4 lag-adjusted dyad $1.44/$1.59; matched-block bootstrap", "analyses/partner_responsiveness.py", 285),
    ("Supplement partner-following slopes", "analyses/partner_responsiveness.py", 261),
    ("Supplement replay regression / Holm algorithm / 3-test family", "analyses/myth_replay_analysis.py", 45),
    ("4.4 injection final validation, means and SDs", "analyses/plot_slide678_rerun.py", 1),
    ("4.5 new-word definition and eligible unseen controls", "analyses/linguistic_uptake.py", 71),
    ("4.5 pair scores; run averaging; Wilcoxon and intervals", "analyses/linguistic_uptake.py", 129),
    ("4.5 exact Holm step-down helper", "analyses/linguistic_uptake.py", 182),
    ("4.5 five-setting and ten-setting family construction", "analyses/linguistic_uptake.py", 454),
    ("4.5 signature-word rates and permutation null", "analyses/linguistic_uptake.py", 202),
    ("4.5 word-placebo construction", "analyses/word_adoption_placebo.py", 67),
    ("4.5 placebo per-run tests and per-contrast 5-test Holm", "analyses/word_adoption_placebo.py", 167),
    ("4.5 moral-placebo raw Wilcoxon and bootstrap", "analyses/moral_placebo.py", 52),
    ("4.5 main moral sample: full vs complete placebo pairs", "analyses/moral_placebo.py", 145),
    ("4.5 held-out-run classifier and map tests", "analyses/myth_map_significance.py", 273),
    ("4.5 run-bootstrap family-distance curves", "analyses/myth_map_significance.py", 101),
    ("4.5 read-amount-to-send regression", "docs/research/frontier_myth_predictors_20260930/plan/analyze.py", 215),
    ("4.5 named-else-band amount measure and panel", "docs/research/frontier_myth_predictors_20260930/plan/common.py", 40),
    ("4.5 own/shown moral, lag and agent/round fixed effects", "analyses/moral_carryover.py", 251),
    ("Figure 6 plot: descriptive demeaning, run means/CIs", "analyses/plot_moral_send_return_simple.py", 64),
    ("4.5 later 71% exact-match sample", "docs/research/myth_predictors_20260930/search/later_rounds_amount.py", 1),
    ("4.5 +$0.074 written amount regression", "docs/research/myth_predictors_20260930/amount/analyze.py", 185),
    ("4.5 amount Holm: rung / all / selected primary", "docs/research/myth_predictors_20260930/amount/analyze.py", 424),
    ("4.5 label shares by round and task order", "analyses/moral_carryover.py", 87),
    ("Methods judge agreement and Cohen's kappa", "analyses/moral_validation.py", 47),
    ("Historical judge/rule-agreement cross-check", "docs/research/myth_predictors_20260930/judges/agreement.py", 19),
    ("4.4 original saved-message context check", "docs/research/myth_predictors_20260930/verify_msgs.py", 1),
    ("4.4 original prompt-fingerprint check", "docs/research/myth_predictors_20260930/verify_fp.py", 1),
    ("Corpus reconstruction from final JSONs", "analyses/linguistic_corpus.py", 138),
    ("Dataset definitions and path selection", "analyses/linguistic_datasets.py", 1),
    ("Dataset/output provenance manifests", "analyses/linguistic_provenance.py", 52),
    ("Completed-final validation and hash manifests", "src/experiment_condition.py", 219),
]


def local_file(module: str, importer: Path) -> Path | None:
    stem = Path(*module.split("."))
    for base in (importer.parent, ROOT, ROOT / "analyses", ROOT / "scripts"):
        for candidate in (base / stem.with_suffix(".py"), base / stem / "__init__.py"):
            if candidate.is_file() and candidate.resolve().is_relative_to(ROOT):
                return candidate.resolve()
    return None


def dependencies(path: Path) -> list[Path]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        names = []
        if isinstance(node, ast.Import):
            names = [x.name for x in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                prefix = ".".join(base.relative_to(ROOT).parts)
                module = ".".join(x for x in (prefix, node.module) if x)
            else:
                module = node.module or ""
            names = [module] + [f"{module}.{x.name}" if module else x.name for x in node.names if x.name != "*"]
        for name in names:
            dep = local_file(name, path)
            if dep and dep != path:
                found.add(dep)
    return sorted(found)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    purposes = {}
    old = json.loads((HERE / "section44/code_manifest.json").read_text())
    for item in old["files"]:
        purposes[item["path"]] = item["purpose"]
    for row in csv.DictReader(io.StringIO((HERE / "section45/source_code.tsv").read_text()), delimiter="\t"):
        purposes[row["path"]] = row["purpose"]
    purposes["src/experiment_condition.py"] = "Completed-final validation and source/output provenance"
    purposes["analyses/linguistic_provenance.py"] = "Dataset and output manifests"
    purposes["analyses/_llm_judge.py"] = "Historical judge helpers and rubric call plumbing; not invoked"
    for label, path, _ in CLAIMS:
        purposes.setdefault(path, label)
    for path in HERE.rglob("*.py"):
        purposes[str(path.relative_to(ROOT))] = "New offline audit/reproduction or artifact-packaging code"

    queue = [ROOT / p for p in purposes]
    local_deps = {}
    while queue:
        path = queue.pop()
        key = str(path.relative_to(ROOT))
        if key in local_deps:
            continue
        deps = dependencies(path)
        local_deps[key] = [str(p.relative_to(ROOT)) for p in deps]
        for dep in deps:
            rel = str(dep.relative_to(ROOT))
            purposes.setdefault(rel, "Transitive local dependency")
            if rel not in local_deps:
                queue.append(dep)

    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    files = []
    for rel in sorted(purposes):
        data = (ROOT / rel).read_bytes()
        files.append({"path": rel, "purpose": purposes[rel], "sha256": sha(data),
                      "bytes": len(data), "lines": len(data.splitlines()),
                      "git_tracked": rel in tracked, "local_dependencies": local_deps[rel]})
    manifest = {"date": "2026-11-23", "repository_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "manuscript_fetched_revision": "cb6336c39afcad6f3fc193502113f3ff2f827c19",
        "source_count": len(files), "files": files}
    (HERE / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    index = ["All code for Sections 4.4 / 4.5 — audited 2026-11-23", "",
             f"{len(files)} Python source files, including original analyses, local dependencies,",
             "historical label/replay producers, and new offline audit scripts.",
             "Exact file bytes are included under source/ in the ZIP; hashes are in source_manifest.json.",
             "Third-party libraries are identified by version in REPRODUCE.txt, not vendored.", "",
             "CLAIM TO CODE (repository-relative path:line)", ""]
    for label, path, line in CLAIMS:
        index.extend([label, f"  {path}:{line}"])
    index.extend(["", "OFFLINE REPRODUCTION", "",
                  "  section44/audit_section44.py — full Section 4.4 plus direct supplement",
                  "  section45/reproduce.py — word/moral/markers/regression/map/amount/descriptive",
                  "  holm_provenance/audit.py — independent Holm, hashes and raw/cache comparison",
                  "  REPRODUCE.txt — exact commands and required existing data", "",
                  "COMPLETE SOURCE INVENTORY", ""])
    for item in files:
        index.extend([item["path"], "  " + item["purpose"],
                      f"  {item['lines']} lines; sha256 {item['sha256']}"])
    (HERE / "CODE_INDEX.txt").write_text("\n".join(index) + "\n")

    entries = {}
    for item in files:
        entries["source/" + item["path"]] = (ROOT / item["path"]).read_bytes()
    for rel in ("requirements.txt", "analyses/requirements.txt"):
        entries["source/" + rel] = (ROOT / rel).read_bytes()
    for name in ("4_results.tex", "3_methods.tex", "supplementary_results.tex"):
        entries["manuscript/" + name] = (PAPER / name).read_bytes()
    for path in sorted(HERE.rglob("*")):
        if path.is_file() and path.suffix in {".txt", ".csv", ".tsv", ".json", ".py"} and "__pycache__" not in path.parts:
            entries["audit/" + str(path.relative_to(HERE))] = path.read_bytes()
    content_manifest = {name: {"sha256": sha(data), "bytes": len(data)} for name, data in entries.items()}
    entries["ARCHIVE_CONTENTS.json"] = (json.dumps(content_manifest, indent=2) + "\n").encode()
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        for name, expected in content_manifest.items():
            assert sha(archive.read(name)) == expected["sha256"], name
    print(json.dumps({"source_files": len(files), "archive_entries": len(entries),
                      "archive_bytes": ARCHIVE.stat().st_size,
                      "sha256": sha(ARCHIVE.read_bytes()), "verified": True}))


if __name__ == "__main__":
    main()

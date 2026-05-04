#!/usr/bin/env python3
"""Rebuild the Overleaf upload folder from the latest rendered manuscript.

Pipeline:
  1. Render the manuscript via the ms-render skill (Quarto → index.tex).
     Assumes _quarto.yml has `cite-method: natbib` so that Quarto emits
     `\\citep{...}` commands directly.
  2. Splice the freshly rendered body (between `\\begin{document}` and
     `\\end{document}`) into the existing main.tex preamble template
     (preserved across runs because the preamble is hand-tuned for
     Overleaf).
  3. Update the abstract in main.tex from index.qmd frontmatter.
  4. Copy new figures from `manuscript/figures/` into
     `manuscript/overleaf-upload/figures/`.
  5. Re-zip overleaf-upload/ → overleaf-upload.zip.

Run from the repo root or via:
    python3 projects/neurips-2026-llm-ling-evo/analysis/rebuild_overleaf.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT = REPO_ROOT / "projects" / "neurips-2026-llm-ling-evo"
MANUSCRIPT = PROJECT / "manuscript"
UPLOAD_DIR = MANUSCRIPT / "overleaf-upload"
FIGURES_SRC = MANUSCRIPT / "figures"
RENDER_SKILL = REPO_ROOT / ".claude" / "skills" / "ms-render" / "render.py"


def render_quarto():
    """Run Quarto via the ms-render skill to refresh index.tex."""
    print(">> Rendering manuscript via ms-render skill ...")
    result = subprocess.run(
        ["python3", str(RENDER_SKILL), "--pdf"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Render stderr:")
        print(result.stderr)
        print("Render stdout:")
        print(result.stdout)
        # Don't bail — index.tex might still have been written before any
        # PDF-only failure (e.g. LaTeX warnings).
    else:
        print(">> Render OK")


def extract_body(tex: str) -> str:
    """Pull the body between \\begin{document} and \\end{document}.

    Excludes the abstract block (handled separately via index.qmd
    frontmatter) and the bibliography command (we add our own).
    """
    m_begin = re.search(r"\\begin\{document\}", tex)
    m_end = re.search(r"\\end\{document\}", tex)
    if not m_begin or not m_end:
        raise RuntimeError("Could not locate \\begin{document} / \\end{document}")
    body = tex[m_begin.end() : m_end.start()]

    # Strip the \maketitle and abstract block — main.tex handles its own.
    body = re.sub(
        r"\\maketitle\s*\n*\\begin\{abstract\}.*?\\end\{abstract\}",
        "",
        body,
        count=1,
        flags=re.DOTALL,
    )
    body = body.lstrip()
    return body


def extract_abstract_from_qmd() -> str:
    """Return the abstract string from index.qmd YAML frontmatter."""
    qmd = (MANUSCRIPT / "index.qmd").read_text(encoding="utf-8")
    m = re.search(r'(?ms)^abstract:\s*"(.*?)"\s*\n', qmd)
    if not m:
        raise RuntimeError("Could not parse abstract from index.qmd")
    abstract = m.group(1)
    # Replace double-quote escapes if any.
    abstract = abstract.replace('\\"', '"')
    return abstract.strip()


# Existing preamble (hand-tuned for Overleaf). Pulled from main.tex.
PREAMBLE_TEMPLATE = r"""% !TeX program = pdflatex
% Upload this whole folder to Overleaf and compile main.tex with pdfLaTeX.
\documentclass[10pt,letterpaper]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{textcomp}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{natbib}
\bibliographystyle{plainnat}
\usepackage{hyperref}
\IfFileExists{xurl.sty}{\usepackage{xurl}}{}
\IfFileExists{upquote.sty}{\usepackage{upquote}}{}
\IfFileExists{microtype.sty}{\usepackage[]{microtype}}{}
% --- Tables (longtable + booktabs) for the Pandoc-rendered tables ---
\usepackage{longtable,booktabs,array}
\newcounter{none} % for unnumbered tables emitted by Pandoc
\usepackage{calc}
\usepackage{etoolbox}
\makeatletter
\patchcmd\longtable{\par}{\if@noskipsec\mbox{}\fi\par}{}{}
\makeatother
\IfFileExists{footnotehyper.sty}{\usepackage{footnotehyper}}{\usepackage{footnote}}
\makesavenoteenv{longtable}
% --- Pandoc Highlighting / Shaded for fenced code blocks ---
\usepackage{fancyvrb}
\newcommand{\VerbBar}{|}
\newcommand{\VERB}{\Verb[commandchars=\\\{\}]}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\}}
\usepackage{framed}
\definecolor{shadecolor}{RGB}{241,243,245}
\newenvironment{Shaded}{\begin{snugshade}}{\end{snugshade}}
\newcommand{\AlertTok}[1]{\textcolor[rgb]{0.68,0.00,0.00}{#1}}
\newcommand{\AnnotationTok}[1]{\textcolor[rgb]{0.37,0.37,0.37}{#1}}
\newcommand{\AttributeTok}[1]{\textcolor[rgb]{0.40,0.45,0.13}{#1}}
\newcommand{\BaseNTok}[1]{\textcolor[rgb]{0.68,0.00,0.00}{#1}}
\newcommand{\BuiltInTok}[1]{\textcolor[rgb]{0.00,0.23,0.31}{#1}}
\newcommand{\CharTok}[1]{\textcolor[rgb]{0.13,0.47,0.30}{#1}}
\newcommand{\CommentTok}[1]{\textcolor[rgb]{0.37,0.37,0.37}{#1}}
\newcommand{\CommentVarTok}[1]{\textcolor[rgb]{0.37,0.37,0.37}{\textit{#1}}}
\newcommand{\ConstantTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{#1}}
\newcommand{\ControlFlowTok}[1]{\textcolor[rgb]{0.00,0.23,0.31}{\textbf{#1}}}
\newcommand{\DataTypeTok}[1]{\textcolor[rgb]{0.68,0.00,0.00}{#1}}
\newcommand{\DecValTok}[1]{\textcolor[rgb]{0.68,0.00,0.00}{#1}}
\newcommand{\DocumentationTok}[1]{\textcolor[rgb]{0.37,0.37,0.37}{\textit{#1}}}
\newcommand{\ErrorTok}[1]{\textcolor[rgb]{0.68,0.00,0.00}{#1}}
\newcommand{\ExtensionTok}[1]{\textcolor[rgb]{0.00,0.23,0.31}{#1}}
\newcommand{\FloatTok}[1]{\textcolor[rgb]{0.68,0.00,0.00}{#1}}
\newcommand{\FunctionTok}[1]{\textcolor[rgb]{0.28,0.35,0.67}{#1}}
\newcommand{\ImportTok}[1]{\textcolor[rgb]{0.00,0.46,0.62}{#1}}
\newcommand{\InformationTok}[1]{\textcolor[rgb]{0.37,0.37,0.37}{#1}}
\newcommand{\KeywordTok}[1]{\textcolor[rgb]{0.00,0.23,0.31}{\textbf{#1}}}
\newcommand{\NormalTok}[1]{\textcolor[rgb]{0.00,0.23,0.31}{#1}}
\newcommand{\OperatorTok}[1]{\textcolor[rgb]{0.37,0.37,0.37}{#1}}
\newcommand{\OtherTok}[1]{\textcolor[rgb]{0.00,0.23,0.31}{#1}}
\newcommand{\PreprocessorTok}[1]{\textcolor[rgb]{0.68,0.00,0.00}{#1}}
\newcommand{\RegionMarkerTok}[1]{\textcolor[rgb]{0.00,0.23,0.31}{#1}}
\newcommand{\SpecialCharTok}[1]{\textcolor[rgb]{0.37,0.37,0.37}{#1}}
\newcommand{\SpecialStringTok}[1]{\textcolor[rgb]{0.13,0.47,0.30}{#1}}
\newcommand{\StringTok}[1]{\textcolor[rgb]{0.13,0.47,0.30}{#1}}
\newcommand{\VariableTok}[1]{\textcolor[rgb]{0.07,0.07,0.07}{#1}}
\newcommand{\VerbatimStringTok}[1]{\textcolor[rgb]{0.13,0.47,0.30}{#1}}
\newcommand{\WarningTok}[1]{\textcolor[rgb]{0.37,0.37,0.37}{\textit{#1}}}
\setcounter{secnumdepth}{-1}
\setlength{\emergencystretch}{3em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\urlstyle{same}
\hypersetup{
  pdftitle={LLM Linguistic Evolution: Language and Cooperation in LLM Dyads},
  pdfauthor={Anonymous Author(s)},
  hidelinks
}
\title{LLM Linguistic Evolution: Language and Cooperation in LLM Dyads}
\author{Anonymous Author(s)}
\date{April 2026}
\begin{document}
\maketitle
"""

POSTAMBLE_TEMPLATE = r"""

\bibliography{references}

\end{document}
"""


def write_main_tex():
    """Compose main.tex from the freshly rendered index.tex body."""
    print(">> Composing main.tex from rendered index.tex ...")
    # Quarto manuscript projects emit to _manuscript/_tex/.
    candidates = [
        MANUSCRIPT / "_manuscript" / "_tex" / "index.tex",
        MANUSCRIPT / "index.tex",
    ]
    index_tex_path = next((p for p in candidates if p.exists()), None)
    if index_tex_path is None:
        raise RuntimeError(
            f"Could not find rendered index.tex in any of {candidates}"
        )
    index_tex = index_tex_path.read_text(encoding="utf-8")

    body = extract_body(index_tex)
    abstract = extract_abstract_from_qmd()

    main_tex = (
        PREAMBLE_TEMPLATE
        + r"\begin{abstract}" + "\n"
        + abstract + "\n"
        + r"\end{abstract}" + "\n\n"
        + body.strip() + "\n"
        + POSTAMBLE_TEMPLATE
    )
    (MANUSCRIPT / "main.tex").write_text(main_tex, encoding="utf-8")
    (MANUSCRIPT / "overleaf-main.tex").write_text(main_tex, encoding="utf-8")
    (UPLOAD_DIR / "main.tex").write_text(main_tex, encoding="utf-8")
    print(f"   wrote {MANUSCRIPT / 'main.tex'}")
    print(f"   wrote {UPLOAD_DIR / 'main.tex'}")


def copy_figures():
    """Copy all figures from manuscript/figures/ to overleaf-upload/figures/."""
    print(">> Copying figures ...")
    dest_dir = UPLOAD_DIR / "figures"
    dest_dir.mkdir(exist_ok=True)
    n = 0
    for src in FIGURES_SRC.glob("*"):
        if src.is_file() and src.suffix.lower() in {".png", ".pdf"}:
            shutil.copy2(src, dest_dir / src.name)
            n += 1
    print(f"   copied {n} figures to {dest_dir}")


def update_readme():
    """Refresh README.txt with the rebuild date."""
    from datetime import datetime
    readme = (
        "Overleaf upload instructions\n"
        "============================\n\n"
        "Upload this whole folder, including the figures/ subfolder, to a new Overleaf project.\n"
        "Set main.tex as the main document and compile with pdfLaTeX.\n\n"
        "Files:\n"
        "- main.tex: conventional LaTeX manuscript using natbib/BibTeX.\n"
        "- references.bib: BibTeX bibliography used by main.tex.\n"
        "- figures/: figure assets used by main.tex.\n"
        "- main-embedded-bibliography.tex: fallback version with references embedded as ordinary text. "
        "Use this only if BibTeX causes trouble; it still needs figures/.\n\n"
        f"This bundle was regenerated on {datetime.now().strftime('%Y-%m-%d %H:%M')} from the current index.qmd "
        "state, including the .c_final Results and Discussion files and the current figure set.\n"
    )
    (UPLOAD_DIR / "README.txt").write_text(readme, encoding="utf-8")
    print(f">> Updated {UPLOAD_DIR / 'README.txt'}")


def make_zip():
    """Zip the overleaf-upload directory."""
    out_zip = MANUSCRIPT / "overleaf-upload.zip"
    if out_zip.exists():
        out_zip.unlink()
    print(">> Building zip ...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in UPLOAD_DIR.rglob("*"):
            if path.is_file() and path.name != ".DS_Store":
                arc = path.relative_to(UPLOAD_DIR.parent)
                zf.write(path, arcname=arc)
    print(f"   wrote {out_zip} ({out_zip.stat().st_size // 1024} KB)")


def fix_citations():
    """Run fix_main_tex.py to convert CSL inline citations to natbib."""
    print(">> Running fix_main_tex.py to convert citations ...")
    fixer = Path(__file__).parent / "fix_main_tex.py"
    result = subprocess.run(
        ["python3", str(fixer)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("fix_main_tex stderr:")
        print(result.stderr)
    print(result.stdout)


def main():
    if not RENDER_SKILL.exists():
        print(f"ERROR: ms-render skill missing at {RENDER_SKILL}", file=sys.stderr)
        sys.exit(1)
    render_quarto()
    write_main_tex()
    copy_figures()
    update_readme()
    fix_citations()
    make_zip()  # re-zip after fix_citations to capture the natbib version
    print("\nDone. Latest overleaf bundle at:")
    print(f"  {MANUSCRIPT / 'overleaf-upload'}/")
    print(f"  {MANUSCRIPT / 'overleaf-upload.zip'}")


if __name__ == "__main__":
    main()

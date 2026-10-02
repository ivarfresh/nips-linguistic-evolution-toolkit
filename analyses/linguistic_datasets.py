"""Which myth corpus the linguistic analyses read and write.

The September informed negative-only runs are the default. The frontier set
(Opus 5, Gemini 3.1 Pro, GPT-5.6 Sol; the 2026-09-18 homogeneous runs plus the
2026-09-28 mixed runs) is selected with LINGUISTIC_DATASET=frontier, or with
--dataset frontier on the scripts that take it. Frontier family names are
distinct from the September ones (Opus / GeminiPro / Sol, never Sonnet / Gemini
/ GPT), so the two corpora cannot be pooled by accident.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = "LINGUISTIC_DATASET"


@dataclass(frozen=True)
class Dataset:
    name: str
    data: Path          # per-myth tables, judge outputs, embeddings (gitignored)
    figs: Path          # summary CSVs and figures
    family_of: dict     # model key -> family
    exact_model: bool   # match the provider model id exactly (else: substring, as in September)
    families: tuple     # display order
    colors: dict
    run_tables: dict | None = None  # per-run tables listing the runs (None: linguistic_corpus default)

    def family(self, model: str) -> str:
        if self.exact_model:
            fam = self.family_of.get(model.split("/", 1)[-1])
            if fam:
                return fam
        else:
            for key, fam in self.family_of.items():
                if key in model:
                    return fam
        raise ValueError(f"unknown model {model!r} for dataset {self.name}")


DATASETS = {
    "september": Dataset(
        "september", ROOT / "data/analysis/linguistic_20260923", ROOT / "docs/figures/linguistic_analysis_20260923",
        {"claude-sonnet-4.5": "Sonnet", "gpt-5-nano": "GPT", "gemini-3.7-flash": "Gemini"}, False,
        ("Sonnet", "Gemini", "GPT"), {"Sonnet": "#7570b3", "GPT": "#d95f02", "Gemini": "#1b9e77"}),
    # 2026-10-01: the same September design at n = 10 runs per cell (original runs plus the
    # Table 1 extension), from the tables written by analyses/table1_n10.py.
    "september_n10": Dataset(
        "september_n10", ROOT / "data/analysis/linguistic_n10_20261001", ROOT / "docs/figures/linguistic_analysis_n10_20261001",
        {"claude-sonnet-4.5": "Sonnet", "gpt-5-nano": "GPT", "gemini-3.7-flash": "Gemini"}, False,
        ("Sonnet", "Gemini", "GPT"), {"Sonnet": "#7570b3", "GPT": "#d95f02", "Gemini": "#1b9e77"},
        {2: ROOT / "docs/figures/mixed_vs_average_n10_20261001/dyad_decisions.csv",
         8: ROOT / "docs/figures/mixed_vs_average_n10_20261001/population_games.csv"}),
    "frontier": Dataset(
        "frontier", ROOT / "data/analysis/linguistic_frontier_20260930",
        ROOT / "data/analysis/linguistic_frontier_20260930/figures",
        {"claude-opus-5": "Opus", "gemini-3.1-pro-preview": "GeminiPro", "gpt-5.6-sol": "Sol"}, True,
        ("Opus", "GeminiPro", "Sol"), {"Opus": "#7570b3", "Sol": "#d95f02", "GeminiPro": "#1b9e77"}),
}


def get(name: str | None = None) -> Dataset:
    return DATASETS[name or os.environ.get(ENV, "september")]

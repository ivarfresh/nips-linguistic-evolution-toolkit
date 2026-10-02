#!/usr/bin/env python3
"""Build the myth and decision panels for both datasets (gitignored scratch outputs).

Writes data/analysis/frontier_myth_predictors_20260930/plan/{myths,panel}_{september,frontier}.pkl
"""
from common import WT, load_panel

CACHE = WT / "data/analysis/frontier_myth_predictors_20260930/plan"

if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    for ds in ("september", "frontier"):
        m, p = load_panel(ds)
        m.to_pickle(CACHE / f"myths_{ds}.pkl")
        p.to_pickle(CACHE / f"panel_{ds}.pkl")
        print(ds, len(m), "myths,", len(p), "decisions")

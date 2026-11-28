#!/usr/bin/env python3
"""Offline independent multiplicity, saved-run, and cache audit. No model/API calls.

Run from any directory with Python 3.11 and numpy/pandas/scipy/statsmodels.
Writes only alongside this file, never to original analysis outputs.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
from statsmodels.stats.multitest import multipletests

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
DETAILS: dict = {}


def read(path):
    return pd.read_csv(ROOT / path)


def independent_holm(p):
    """Sorted Bonferroni factors, cumulative maximum, invert permutation."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid], kind="stable")]
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(order), 0, -1)))
    return out


def checked_holm(p):
    p = np.asarray(p, dtype=float)
    h = independent_holm(p)
    ok = np.isfinite(p)
    if ok.any():
        np.testing.assert_allclose(h[ok], multipletests(p[ok], method="holm")[1], atol=0, rtol=1e-14)
    return h


def load_function(path, name):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def custom_implementations():
    rng = np.random.default_rng(20261123)
    rows = []
    for file in ["analyses/linguistic_uptake.py", "analyses/myth_replay_analysis.py"]:
        fn = load_function(file, "holm")
        count = 0
        for n in (1, 3, 5, 10, 25, 62, 178):
            for _ in range(50):
                p = rng.random(n)
                if n > 2:
                    p[:3] = [0, 1, p[-1]]
                if file.endswith("linguistic_uptake.py") and n > 3:
                    p[3] = np.nan
                np.testing.assert_allclose(fn(p), checked_holm(p), rtol=1e-14, atol=0, equal_nan=True)
                count += 1
        rows.append({"source": file, "randomized_edge_cases": count, "matches_independent_and_statsmodels": True})
    pd.DataFrame(rows).to_csv(OUT / "holm_implementations.csv", index=False)


@lru_cache(maxsize=None)
def exact_rank_cumulative(n):
    """Integer subset-sum distribution avoids floating tail cancellation."""
    counts = [0] * (n * (n+1)//2 + 1)
    counts[0] = 1
    total = 0
    for rank in range(1, n+1):
        for s in range(total, -1, -1):
            counts[s+rank] += counts[s]
        total += rank
    cumulative, running = [], 0
    for count in counts:
        running += count
        cumulative.append(running)
    return cumulative


def wilcoxon_record(values, **label):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    zeros = int((v == 0).sum())
    ties = len(np.unique(np.abs(v[v != 0]))) != int((v != 0).sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto = stats.wilcoxon(v, method="auto")
        approx = stats.wilcoxon(v, method="asymptotic")
        exact = stats.wilcoxon(v, method="exact") if not zeros and not ties else None
    return {**label, "n": len(v), "zero_differences": zeros, "absolute_rank_ties": ties,
            "mean_difference": float(np.mean(v)), "sd_difference": float(np.std(v, ddof=1)),
            "statistic": auto.statistic, "p_auto": auto.pvalue, "p_asymptotic": approx.pvalue,
            "p_exact_no_ties": min(1, 2*exact_rank_cumulative(len(v))[int(auto.statistic)] / (2**len(v)))
                               if exact is not None else np.nan,
            "p_exact_scipy_sensitivity": exact.pvalue if exact is not None else np.nan,
            "positive": int((v > 0).sum()), "negative": int((v < 0).sum()),
            "p_sign_two_sided": stats.binomtest(int((v > 0).sum()), int((v != 0).sum()), .5).pvalue}


def word_tables():
    children = read("data/analysis/linguistic_n10_20261001/uptake_children.csv")
    results, checks = [], []
    for by, filename in [(["size", "exposure"], "reuse_summary.csv"),
                         (["size", "exposure", "task_order"], "reuse_by_task_order.csv")]:
        run = children.groupby(by + ["run_id"])[["adopt_excess", "cos_excess"]].mean().reset_index()
        original = read("docs/figures/linguistic_analysis_n10_20261001/" + filename)
        for metric in ("adopt_excess", "cos_excess"):
            rows = [wilcoxon_record(g[metric], table=filename, metric=metric, **dict(zip(by, key)))
                    for key, g in run.groupby(by)]
            table = pd.DataFrame(rows)
            table["p_holm"] = checked_holm(table.p_auto)
            comparison = table.merge(original, on=by, suffixes=("", "_saved"))
            rawdiff = np.max(np.abs(comparison.p_auto - comparison[metric + "_p"]))
            hd = np.max(np.abs(comparison.p_holm - comparison[metric + "_p_holm"]))
            np.testing.assert_allclose(comparison.p_auto, comparison[metric + "_p"], rtol=1e-10, atol=0)
            np.testing.assert_allclose(comparison.p_holm, comparison[metric + "_p_holm"], rtol=1e-10, atol=0)
            checks.append({"table": filename, "metric": metric, "n_tests": len(table),
                           "max_abs_raw_difference": rawdiff, "max_abs_holm_difference": hd})
            results.extend(table.to_dict("records"))
    pd.DataFrame(results).to_csv(OUT / "word_wilcoxon_and_holm.csv", index=False)
    per = read("docs/figures/word_adoption_placebo_20261008/placebo_per_run.csv")
    original = read("docs/figures/word_adoption_placebo_20261008/placebo_summary.csv")
    contrasts = {"shown_vs_unseen": ("adopt_parent", "adopt_null"),
                 "shown_vs_placebo": ("adopt_parent", "adopt_placebo"),
                 "placebo_vs_unseen": ("adopt_placebo", "adopt_null"),
                 "placebo_vs_unseen_same_round": ("adopt_placebo", "adopt_null_same_round"),
                 "shownonly_vs_futureunseen": ("adopt_shown_only", "adopt_future_unseen"),
                 "sym_shown_vs_future": ("adopt_sym_shown_only", "adopt_sym_future_only")}
    results = []
    for name, (a, b) in contrasts.items():
        rows = [wilcoxon_record((g[a] - g[b]).dropna().round(10), contrast=name, size=sz, exposure=exp)
                for (sz, exp), g in per.groupby(["size", "exposure"])]
        table = pd.DataFrame(rows)
        table["p_holm"] = checked_holm(table.p_auto)
        cmp = table.merge(original, on=["size", "exposure"])
        np.testing.assert_allclose(cmp.p_auto, cmp[name + "_p"], rtol=1e-10, atol=0)
        np.testing.assert_allclose(cmp.p_holm, cmp[name + "_p_holm"], rtol=1e-10, atol=0)
        checks.append({"table": "placebo_summary.csv", "metric": name, "n_tests": len(table),
                       "max_abs_raw_difference": np.max(abs(cmp.p_auto-cmp[name+"_p"])),
                       "max_abs_holm_difference": np.max(abs(cmp.p_holm-cmp[name+"_p_holm"]))})
        results.extend(table.to_dict("records"))
    table = pd.DataFrame(results)
    table["p_holm_all_30_sensitivity"] = checked_holm(table.p_auto)
    table.to_csv(OUT / "word_placebo_wilcoxon_and_holm.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "holm_table_checks.csv", index=False)


def moral_tables():
    children = read("data/analysis/moral_placebo_20261008/moral_placebo_children.csv")
    main = children[(children.setting == "8-agent mixed") & (children.task_order == "myth_game")]
    restricted = main.dropna(subset=["same_label_shown", "same_label_unseen", "same_label_placebo", "same_label_unseen_next"])
    records = []
    for name, df, b in [("shown_minus_unseen", main, "same_label_unseen"),
                        ("shown_minus_placebo", restricted, "same_label_placebo")]:
        for exp, group in df.groupby("exposure"):
            per = group.groupby("run_id")[["same_label_shown", b]].mean()
            records.append(wilcoxon_record((per.same_label_shown - per[b]).round(10), contrast=name, exposure=exp))
    out = pd.DataFrame(records)
    out["p_holm_4_headline_sensitivity"] = checked_holm(out.p_auto)
    out["p_holm_2_per_contrast_sensitivity"] = out.groupby("contrast").p_auto.transform(lambda x: checked_holm(x))
    out.to_csv(OUT / "moral_headline_multiplicity_sensitivity.csv", index=False)
    # Full existing search: ten setting/exposure/order cells x four contrast types.
    existing = read("docs/figures/moral_placebo_20261008/placebo_by_setting_task_order.csv")
    long = existing.melt(id_vars=["setting", "exposure", "task_order"],
                         value_vars=[c for c in existing if c.endswith("_p")], var_name="contrast", value_name="p")
    long["p_holm_per_contrast_sensitivity"] = long.groupby("contrast").p.transform(lambda x: checked_holm(x))
    long["p_holm_all_sensitivity"] = checked_holm(long.p)
    long.to_csv(OUT / "moral_full_table_multiplicity_sensitivity.csv", index=False)


def amount_and_replay():
    tests = read("docs/research/myth_predictors_20260930/amount/tests.csv")
    for old, new in [("p_holm_all", checked_holm(tests.p)),
                     ("p_holm_within_rung", tests.groupby("rung").p.transform(lambda x: checked_holm(x)))]:
        np.testing.assert_allclose(tests[old], new, rtol=1e-10, atol=0, equal_nan=True)
    primary = tests["primary"]
    np.testing.assert_allclose(tests.loc[primary, "p_holm_primary"], checked_holm(tests.loc[primary, "p"]),
                               rtol=1e-10, atol=0, equal_nan=True)
    tests.to_csv(OUT / "amount_tests_verified.csv", index=False)
    DETAILS["amount_families"] = {"all_rows": len(tests), "estimable": int(tests.p.notna().sum()),
                                  "primary_estimable": int(tests.loc[primary, "p"].notna().sum()),
                                  "by_rung": tests.groupby("rung").p.count().to_dict()}
    replay = read("docs/research/myth_replay_probe_20261001/slopes.csv")
    replay = replay[replay.family == "Sonnet+GPT (primary)"].copy()
    replay["p_holm_recalculated"] = checked_holm(replay.p)
    np.testing.assert_allclose(replay.p_holm, replay.p_holm_recalculated, rtol=1e-10, atol=0)
    replay.to_csv(OUT / "replay_primary_holm.csv", index=False)


def correction_ledger():
    records = []
    def add(table, family, p_col, adjusted, labels, subset=None):
        sub = table if subset is None else table.loc[subset]
        corrected = checked_holm(sub[p_col])
        m = int(sub[p_col].notna().sum())
        for (_, row), calculated in zip(sub.iterrows(), corrected):
            if pd.isna(row[p_col]):
                continue
            records.append({"family": family, "family_size": m,
                            "hypothesis": " | ".join(str(row[k]) for k in labels),
                            "p_raw": row[p_col], "p_holm_stored": row[adjusted],
                            "p_holm_independent": calculated,
                            "absolute_difference": abs(row[adjusted]-calculated)})
    for filename in ["reuse_summary.csv", "reuse_by_task_order.csv"]:
        t = read("docs/figures/linguistic_analysis_n10_20261001/"+filename)
        keys = [c for c in ["size", "exposure", "task_order"] if c in t]
        for metric in ["adopt_excess", "cos_excess"]:
            add(t, filename+":"+metric, metric+"_p", metric+"_p_holm", keys)
    t = read("docs/figures/word_adoption_placebo_20261008/placebo_summary.csv")
    for c in [c for c in t if c.endswith("_p_holm")]:
        add(t, "word placebo:"+c[:-7], c[:-5], c, ["size", "exposure"])
    t = read("docs/research/myth_predictors_20260930/amount/tests.csv")
    keys = ["rung", "test", "family", "measure", "term"]
    add(t, "amount:all", "p", "p_holm_all", keys)
    add(t, "amount:primary", "p", "p_holm_primary", keys, t.primary)
    for rung in t.rung.unique():
        add(t, "amount:rung:"+rung, "p", "p_holm_within_rung", keys, t.rung==rung)
    t = read("docs/research/myth_replay_probe_20261001/slopes.csv")
    add(t, "replay:three primary positions", "p", "p_holm", ["arm", "family", "mode"], t.family=="Sonnet+GPT (primary)")
    pd.DataFrame(records).to_csv(OUT / "all_corrections.csv", index=False)


MANIFESTS = [
    "docs/figures/linguistic_analysis_n10_20261001/provenance.json",
    "docs/figures/word_adoption_placebo_20261008/provenance.json",
    "docs/figures/moral_placebo_20261008/provenance.json",
    "docs/figures/myth_convergence_map_n10_20261006/provenance.json",
    "docs/figures/linguistic_analysis_20260923/provenance.json",
    "docs/figures/moral_send_return_simple_20261007/provenance.json",
    "docs/figures/moral_send_return_simple_frontier_20261007/provenance.json",
    "docs/figures/partner_responsiveness_20260930/provenance.json",
    "docs/figures/slide678_dyad_rerun_20260917/provenance.json",
]


def rel_json(p):
    s = str(p)
    return "data/json/" + s.split("data/json/", 1)[1]


def check_provenance():
    manifests, source_refs, output_checks = {}, {}, []
    for name in MANIFESTS:
        doc = json.loads((ROOT / name).read_text())
        manifests[name] = doc
        for run in doc["runs"]:
            rel = rel_json(run["path"])
            source_refs.setdefault(rel, []).append((name, run["sha256"], Path(run["path"]).exists()))
        for path, expected in doc["outputs"].items():
            p = ROOT / name
            p = p.parent / path
            exists = p.exists()
            match = exists and hashlib.sha256(p.read_bytes()).hexdigest() == expected
            output_checks.append({"manifest": name, "file": path, "exists": exists, "hash_match": match})
    pd.DataFrame(output_checks).to_csv(OUT / "manifest_output_hashes.csv", index=False)
    cache_frames = {}
    for corpus in ["linguistic_20260923", "linguistic_n10_20261001", "linguistic_frontier_20260930"]:
        for kind in ["myths", "decisions"]:
            df = read(f"data/analysis/{corpus}/{kind}.csv")
            cache_frames[(corpus, kind)] = dict(tuple(df.groupby("path", sort=False)))
    source_checks, cache_checks = [], []
    for index, (relative, refs) in enumerate(source_refs.items()):
        p = ROOT / relative
        raw = p.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        md = data.get("run_metadata", {})
        history = data.get("conversation_history", [])
        rounds = [int(r["round"]) for r in history]
        nround = int(md["num_turns"])
        agents = set(data.get("agents", {}))
        n = int(md["num_agents"])
        is_full = all(k in data for k in ("conversation_history", "game_data", "run_metadata", "agents", "task_order"))
        suffix_ok = p.suffix == ".json" and not any(x in p.name for x in (".checkpoint.", ".error.", ".results."))
        complete = rounds == list(range(1, nround+1)) and len(agents) == n
        games_complete = all(len(e.get("dyads", [])) == n//2 for e in history)
        family_lookup = {"claude-sonnet-4.5": "Sonnet", "gpt-5-nano": "GPT", "gemini-3.7-flash": "Gemini",
                         "claude-opus-5": "Opus", "gemini-3.1-pro-preview": "GeminiPro", "gpt-5.6-sol": "Sol"}
        requests = (md.get("llm_request") or {}).get("agents")
        models = {a: requests[a]["model"] if requests else md["model"] for a in agents}
        families = {a: family_lookup.get(model.split("/")[-1]) for a, model in models.items()}
        for manifest, expected, original_exists in refs:
            source_checks.append({"manifest": manifest, "path": relative, "saved_absolute_exists": original_exists,
                                  "hash_match": digest == expected, "full_state": is_full, "final_suffix": suffix_ok,
                                  "rounds_complete": complete, "all_dyads_complete": games_complete,
                                  "rounds": len(rounds), "agents": n})
        myth_rows, dec_rows = [], []
        for entry in history:
            rnd = int(entry["round"])
            partner = {}
            for pair in entry.get("pairings", []):
                partner[pair["investor"]] = pair["trustee"]
                partner[pair["trustee"]] = pair["investor"]
            exposures = entry.get("myth_exposures") or {}
            for agent, txt in (entry.get("myths") or {}).items():
                exp = exposures.get(agent) or {}
                myth_rows.append({"round": rnd, "agent": agent, "text": txt, "n_words": len(str(txt).split()),
                                  "partner_this_round": partner.get(agent), "exposed_author": exp.get("original_author_id"),
                                  "exposed_round": exp.get("source_round")})
            for d in entry.get("dyads") or []:
                s, r, b = float(d["sent"]), float(d["received"]), float(d["returned"])
                frac = b/r if r>0 else np.nan
                for a, role, other in ((d["investor"], "investor", d["trustee"]), (d["trustee"], "trustee", d["investor"])):
                    dec_rows.append({"round": rnd, "agent": a, "role": role, "partner": other, "sent": s,
                                     "received": r, "returned": b, "return_proportion": frac,
                                     "coop": s/5 if role == "investor" else frac})
        for (corpus, kind), frame_by_path in cache_frames.items():
            cached = frame_by_path.get(relative)
            if cached is None:
                continue
            actual = pd.DataFrame(myth_rows if kind == "myths" else dec_rows)
            cols = actual.columns.tolist()
            expected = cached[cols]
            actual = actual.sort_values(["round", "agent"]).reset_index(drop=True)
            expected = expected.sort_values(["round", "agent"]).reset_index(drop=True)
            # CSV readers use NaN for absent optional fields/empty texts.
            actual = actual.replace({None: np.nan, "": np.nan})
            try:
                pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
                match, error = True, ""
            except AssertionError as exc:
                match, error = False, str(exc).splitlines()[0]
            base_match = bool((cached["size"] == n).all() and
                              (cached.task_order == "_".join(data["task_order"])).all() and
                              (cached.run_id == p.stem).all() and
                              (cached.replicate_id == md["replicate_id"]).all() and
                              (cached.mixed == (len(set(families.values())) > 1)).all() and
                              all(row.family == families[row.agent] for row in cached.itertuples()))
            other = "partner" if kind == "decisions" else "exposed_author"
            other_family = "partner_family" if kind == "decisions" else "exposed_family"
            for row in cached.itertuples():
                agent = getattr(row, other)
                expected_family = getattr(row, other_family)
                if pd.notna(agent) and expected_family != families[agent]:
                    base_match = False
            # Verify the displayed composition names encode exactly the raw per-agent family counts.
            counts = pd.Series(list(families.values())).value_counts().to_dict()
            for composition in cached.composition.unique():
                got = {}
                for token in composition.split("+"):
                    token = token.strip()
                    count, fam = token.split(" ", 1) if " " in token else ("1", token)
                    got[fam] = got.get(fam, 0) + int(count)
                if counts != got:
                    base_match = False
            cache_checks.append({"corpus": corpus, "table": kind, "path": relative, "rows": len(expected),
                                 "all_checked_fields_match": match, "metadata_fields_match": base_match, "error": error})
        if index % 100 == 0:
            print(f"verified {index+1}/{len(source_refs)} distinct final JSON files", flush=True)
    pd.DataFrame(source_checks).to_csv(OUT / "manifest_source_hashes_and_completion.csv", index=False)
    pd.DataFrame(cache_checks).to_csv(OUT / "raw_to_cache_checks.csv", index=False)
    DETAILS["provenance"] = {"unique_finals": len(source_refs), "manifest_entries": len(source_checks),
                              "source_hash_mismatches": sum(not x["hash_match"] for x in source_checks),
                              "stale_absolute_paths": sum(not x["saved_absolute_exists"] for x in source_checks),
                              "output_hash_mismatches": sum(not x["hash_match"] for x in output_checks),
                              "cache_mismatches": sum(not x["all_checked_fields_match"] for x in cache_checks),
                              "metadata_cache_mismatches": sum(not x["metadata_fields_match"] for x in cache_checks)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    DETAILS["versions"] = {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                            "statsmodels": statsmodels.__version__}
    custom_implementations()
    word_tables()
    moral_tables()
    amount_and_replay()
    correction_ledger()
    check_provenance()
    (OUT / "summary.json").write_text(json.dumps(DETAILS, indent=2) + "\n")
    print(json.dumps(DETAILS, indent=2))

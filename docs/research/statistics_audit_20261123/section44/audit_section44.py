#!/usr/bin/env python3
"""Read-only numerical audit of paper §4.4; outputs beside this script.

Uses saved data only. No simulations, judges, network calls, or old-output writes.
Independent fits use statsmodels/pandas; bootstraps use sufficient statistics.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
KEY = ["run_id", "round", "agent"]
DATA = ROOT / "data/analysis/linguistic_20260923"


def write(name, value):
    if isinstance(value, pd.DataFrame):
        value.to_csv(OUT / f"{name}.csv", index=False)
    else:
        (OUT / f"{name}.json").write_text(json.dumps(value, indent=2, default=str) + "\n")


def estimates(fit, term, **extra):
    ci = fit.conf_int().loc[term]
    return {**extra, "coef": fit.params[term], "se": fit.bse[term], "ci_lo": ci[0],
            "ci_hi": ci[1], "p": fit.pvalues[term], "n": int(fit.nobs)}


def check_raw(d, m, tag):
    """Cross-check every cached decision and myth against a 10-round full final."""
    decisions, myths, final_rows, fps = [], [], [], []
    fam_lookup = d.set_index(["run_id", "agent"]).family.to_dict()
    for path, g in d.groupby("path"):
        p = ROOT / path
        raw = p.read_bytes()
        j = json.loads(raw)
        rid = g.run_id.iloc[0]
        h = j["conversation_history"]
        assert [v["round"] for v in h] == list(range(1, 11)), path
        assert j.get("agents") and all(a.get("interaction_history") for a in j["agents"].values()), path
        assert not any(s in p.name for s in (".checkpoint.", ".results.", ".error.")), path
        final_rows.append({"dataset": tag, "run_id": rid, "path": path,
                           "sha256": hashlib.sha256(raw).hexdigest(), "rounds": len(h),
                           "n_agents": len(j["agents"]), "noise_semantics": j["run_metadata"].get("noise_semantics")})
        for turn in h:
            for dyad in turn["dyads"]:
                for agent, role, partner in [(dyad["investor"], "investor", dyad["trustee"]),
                                             (dyad["trustee"], "trustee", dyad["investor"])]:
                    decisions.append({"run_id": rid, "round": turn["round"], "agent": agent, "role": role,
                                      "partner": partner, "sent": float(dyad["sent"]),
                                      "received": float(dyad["received"]), "returned": float(dyad["returned"])})
                assert dyad["sent"] == dyad.get("sent_decision", dyad["sent"]), path
            for agent, myth in turn.get("myths", {}).items():
                myths.append({"run_id": rid, "round": turn["round"], "agent": agent, "text": myth})
        if tag == "september" and j["task_order"] == ["myth", "game"]:
            for agent, a in j["agents"].items():
                calls = [c for c in a["interaction_history"] if c["metadata"].get("round") == 1]
                games = [c for c in calls if c["metadata"].get("task") == "game" and c["metadata"].get("role") == "investor"]
                for call in games:
                    messages = call["messages_sent"]
                    assert [v["role"] for v in messages] == ["system", "user", "assistant", "user"], (path, agent)
                    masked = [{"role": v["role"], "content": "<OWN MYTH>" if v["role"] == "assistant" else v["content"]} for v in messages]
                    fps.append({"run_id": rid, "agent": agent, "family": fam_lookup[(rid, agent)],
                                "size": len(j["agents"]), "prompt_hash": hashlib.sha256(json.dumps(masked, sort_keys=True).encode()).hexdigest()})
    rebuilt = pd.DataFrame(decisions).set_index(KEY).sort_index()
    cached = d.set_index(KEY).sort_index()
    assert cached.index.equals(rebuilt.index)
    for c in ("sent", "received", "returned"):
        np.testing.assert_allclose(cached[c], rebuilt[c], rtol=0, atol=1e-12)
    for c in ("role", "partner"):
        assert cached[c].equals(rebuilt[c])
    rbm = pd.DataFrame(myths).set_index(KEY).sort_index()
    cm = m.set_index(KEY).sort_index()
    assert cm.index.equals(rbm.index)
    assert cm.text.fillna("").equals(rbm.text.fillna(""))
    write(f"{tag}_finals_verified", pd.DataFrame(final_rows))
    if fps:
        fp = pd.DataFrame(fps)
        write("round1_prompt_fingerprints", fp)
        write("round1_prompt_fingerprint_summary", fp.groupby(["family", "size"]).agg(senders=("agent", "size"), runs=("run_id", "nunique"), distinct_nonmyth_prompts=("prompt_hash", "nunique")).reset_index())
    return {"runs": len(final_rows), "decisions": len(rebuilt), "myths": len(rbm), "cache_matches_finals": True}


def round1(d, rules, tag):
    r = d[d.role.eq("investor") & d.task_order.eq("myth_game") & d["round"].eq(1)].merge(
        rules.loc[rules.status.eq("ok"), KEY + ["send_amount", "send_rule"]], on=KEY, how="left", validate="one_to_one")
    fams = ["Sonnet", "GPT"] if tag == "september" else ["Sol", "Opus"]
    rows, samples = [], []
    for label, group in [("+".join(fams), fams)] + [(f, [f]) for f in sorted(r.family.unique())]:
        a = r[r.family.isin(group)].dropna(subset=["send_amount"])
        exact = np.isclose(a.sent, a.send_amount)
        base = {"dataset": tag, "family": label, "n_senders_all": int(r.family.isin(group).sum()), "n_named": len(a),
                "n_runs": a.run_id.nunique(), "n_exact": int(exact.sum()), "exact_share": exact.mean(),
                "mean_sent": a.sent.mean(), "sd_sent": a.sent.std(), "mean_named": a.send_amount.mean(), "sd_named": a.send_amount.std()}
        samples.append(base)
        if a.sent.nunique() > 1 and a.send_amount.nunique() > 1:
            for use_t in (False, True):
                fit = smf.ols("sent ~ send_amount + C(composition)", a).fit(
                    cov_type="cluster", cov_kwds={"groups": pd.factorize(a.run_id)[0]}, use_t=use_t)
                rows.append(estimates(fit, "send_amount", **base, inference="t(G-1)" if use_t else "normal"))
    if tag == "september":
        s = r[r.family.eq("Sonnet") & r.send_amount.eq(3)].copy()
        write("sonnet_3_named", s[KEY + ["path", "composition", "sent", "send_amount"]])
        a = r[r.family.isin(fams)].dropna(subset=["send_amount"])
        # Clearly labelled sensitivity, not a substitute for the original analysis.
        f = smf.ols("sent ~ send_amount + C(composition) + C(family)", a).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(a.run_id)[0]}, use_t=True)
        rows.append(estimates(f, "send_amount", dataset=tag, family="Sonnet+GPT", inference="t(G-1), additionally family FE", n_runs=a.run_id.nunique()))
    write(f"{tag}_round1_named_rows", r.dropna(subset=["send_amount"])[KEY + ["path", "family", "composition", "size", "sent", "send_amount"]])
    return rows, samples


def mediation(d):
    inv = d[d.role.eq("investor")]
    x = d.drop_duplicates("run_id").set_index("run_id")[["composition", "size", "mixed", "task_order", "replicate_id"]].join([
        inv[inv["round"].eq(1)].groupby("run_id").sent.mean().rename("r1_send"),
        inv[inv["round"].between(2, 10)].groupby("run_id").sent.mean().rename("later_send")]).reset_index()
    x["myth_first"] = x.task_order.eq("myth_game").astype(float)
    x["cell"] = x.composition + "|" + x["size"].astype(str)
    write("opening_adjustment_runs", x)
    rows = []
    for name, g in [("single-model", x[~x.mixed]), ("mixed-model", x[x.mixed]), ("all", x)]:
        for spec, formula in [("task order only", "later_send ~ myth_first + C(cell)"),
                              ("+ round-1 send", "later_send ~ myth_first + r1_send + C(cell)")]:
            f = smf.ols(formula, g).fit(cov_type="HC1")
            rows.append(estimates(f, "myth_first", runs=name, spec=spec, formula=formula, n_runs=len(g), n_cells=g.cell.nunique(),
                                  r1_coef=f.params.get("r1_send"), r1_p=f.pvalues.get("r1_send")))
    out = pd.DataFrame(rows)
    old = pd.read_csv(ROOT / "docs/research/myth_replay_probe_20261001/opening_mediation.csv")
    np.testing.assert_allclose(out.coef, old.gap, atol=1e-12)
    np.testing.assert_allclose(out[["ci_lo", "ci_hi"]], old[["ci_low", "ci_high"]], atol=1e-12)
    write("opening_adjustment_models", out)
    write("opening_adjustment_descriptives", x.groupby(["mixed", "task_order"]).agg(n_runs=("run_id", "size"), opening_mean=("r1_send", "mean"), opening_sd=("r1_send", "std"), later_mean=("later_send", "mean"), later_sd=("later_send", "std")).reset_index())


def replay():
    from analyses import myth_replay_probe as producer
    # These two producer functions only load existing runs and build a job list.
    # Neither replay(), edited_version(), make_main(), nor producer.main() is called.
    planned = len(producer.plan_main(producer.load_all_runs(ROOT)))
    assert planned == 4720
    raw = ROOT / "data/analysis/myth_replay_probe_20261001/main.jsonl"
    d = pd.DataFrame([json.loads(s) for s in raw.read_text().splitlines() if s.strip()]).drop_duplicates("key", keep="last")
    rep = d[d.made.eq(True) & d.error.isna()]
    ed = rep[rep["mode"].isin(["rule", "natural"])]
    rows = []
    groups = [(arm, fam, mode, g) for (arm, fam, mode), g in ed.groupby(["arm", "family", "mode"]) if fam != "Gemini"]
    groups += [(arm, "Sonnet+GPT (primary)", "rule", g) for arm, g in ed[ed["mode"].eq("rule") & ed.family.isin(["Sonnet", "GPT"])].groupby("arm")]
    for arm, fam, mode, g in groups:
        f = smf.ols("send ~ edited_amount + C(cid)", g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.run)[0]}, use_t=True)
        rows.append(estimates(f, "edited_amount", arm=arm, family=fam, mode=mode, n_contexts=g.cid.nunique(), n_runs=g.run.nunique(), threshold_030_standard_errors=(f.params.edited_amount - .30) / f.bse.edited_amount))
    out = pd.DataFrame(rows)
    idx = out.family.eq("Sonnet+GPT (primary)")
    out.loc[idx, "p_holm"] = multipletests(out.loc[idx, "p"], method="holm")[1]
    from analyses.myth_replay_analysis import holm
    np.testing.assert_allclose(holm(out.loc[idx, "p"].tolist()), out.loc[idx, "p_holm"], rtol=1e-12, atol=0)
    old = pd.read_csv(ROOT / "docs/research/myth_replay_probe_20261001/slopes.csv")
    j = out.merge(old, on=["arm", "family", "mode"], suffixes=("_new", "_old"))
    for a, b in [("coef", "slope"), ("ci_lo", "ci_low"), ("ci_hi", "ci_high")]:
        np.testing.assert_allclose(j[a], j[b], atol=1e-11)
    primary = j.family.eq("Sonnet+GPT (primary)")
    np.testing.assert_allclose(j.loc[primary, "p_holm_new"], j.loc[primary, "p_holm_old"], rtol=1e-10, atol=0)
    write("replay_models", out)
    means = rep.groupby(["arm", "family", "mode", "edited_amount"], dropna=False).agg(n=("send", "size"), mean=("send", "mean"), sd=("send", "std"), original_mean=("original_send", "mean"), original_sd=("original_send", "std")).reset_index()
    write("replay_descriptives", means)
    write("replay_integrity", {"raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "n_rows": len(d), "n_replayed": len(rep), "errors": int(d.error.notna().sum()), "parse_failures": int(rep.send.isna().sum()), "planned": planned})


def push():
    from analyses import partner_responsiveness as pr
    from analyses import partner_responsiveness_extract as pe
    dec = pr.load()
    # Rebuild precisely the dyad inputs used in this model directly from full finals.
    specs = [s for s in pe.run_specs() if s["run_set"] != "random"]
    paths = set(dec.loc[dec.num_agents.eq(2) & ~dec.run_set.eq("random"), "path"])
    rebuilt = pd.DataFrame([row for s in specs if s["path"] in paths for row in pe.decision_rows(s)])
    cols = ["decision_frac", "sig_shown", "own_prev2_sent"]
    k = ["path", "round", "agent", "role"]
    a = dec[dec.path.isin(paths)].set_index(k).sort_index()
    b = rebuilt.set_index(k).sort_index()
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a[cols], b[cols], equal_nan=True, atol=1e-12)
    write("push_input_verification", {"n_runs": len(paths), "n_decisions": len(a), "columns_verified": cols, "cache_matches_finals": True})
    d = dec[dec.role.eq("investor") & dec.num_agents.eq(2) & ~dec.run_set.eq("random")].copy()
    d["own_prev_usd"] = d.own_prev2_sent * 5
    d = d.dropna(subset=["send_usd", "sig_shown_usd", "own_prev_usd"])
    rows = []
    rng = np.random.default_rng(20261123)
    for model, g in d.groupby("model"):
        def xy(z):
            return np.column_stack([np.ones(len(z)), z.task_order.eq("game_myth"), z.task_order.eq("myth_game"), z.sig_shown_usd, z.own_prev_usd]).astype(float), z.send_usd.to_numpy(float)
        X, y = xy(g)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        parts = [xy(s) for _, s in g.groupby(["replicate", "noise"])]
        xx = np.array([x.T @ x for x, y in parts])
        xyv = np.array([x.T @ y for x, y in parts])
        draw = rng.integers(0, len(parts), (20000, len(parts)))
        A, B = xx[draw].sum(axis=1), xyv[draw].sum(axis=1)
        bs = np.einsum("nij,nj->ni", np.linalg.pinv(A), B)
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        for j, term in enumerate(["intercept", "game_myth", "myth_game", "partner_last", "own_previous"]):
            rows.append({"model": model, "term": term, "coef": coef[j], "ci_lo_independent_bootstrap": lo[j], "ci_hi_independent_bootstrap": hi[j], "n_decisions": len(g), "n_runs": g.path.nunique(), "n_matched_blocks": len(parts), "n_boot": len(draw)})
    out = pd.DataFrame(rows)
    old = pd.read_csv(ROOT / "docs/figures/partner_responsiveness_20260930/myth_push.csv")
    for r in old.itertuples():
        for order in ["game_myth", "myth_game"]:
            got = out.loc[out.model.eq(r.model) & out.term.eq(order), "coef"].item()
            assert abs(got - getattr(r, order + "_push")) <= .000501
    write("push_models", out)
    # Exact following point-estimate method; original table CIs are not printed.
    following = []
    d = dec[dec.role.eq("investor") & dec.num_agents.eq(2) & ~dec.run_set.eq("random") & dec.sig_shown_usd.notna()]
    for (model, order), g in d.groupby(["model", "task_order"]):
        y = g.send_usd - g.groupby("path").send_usd.transform("mean")
        x = g.sig_shown_usd - g.groupby("path").sig_shown_usd.transform("mean")
        y, x = y - y.groupby(g["round"]).transform("mean"), x - x.groupby(g["round"]).transform("mean")
        slope = (x*y).sum()/(x*x).sum() if (x*x).sum() > 1e-9 else np.nan
        runmeans = g.groupby("path").send_usd.mean()
        following.append({"model": model, "task_order": order, "slope": slope, "n_runs": len(runmeans), "mean_sent": runmeans.mean(), "sd_sent": runmeans.std()})
    write("following_models", pd.DataFrame(following))


def original_push_bootstrap():
    """Reproduce the original 2,000-resample stream without unnecessary regressions.

    The original module uses one generator for earlier analyses before myth_push.
    Reproduce exactly those draws; skip calculations whose result does not affect it.
    """
    from analyses import partner_responsiveness as pr
    dec = pr.load()
    pr.RNG = np.random.default_rng(20260930)
    original_iv, original_boot = pr.iv_slope, pr._boot_by_replicate
    pr.iv_slope = lambda arrays, keys, nz: (np.nan, np.nan)
    def consume_boot(g, fn):
        reps = sorted(g.replicate.unique())
        for _ in range(pr.N_BOOT):
            pr.RNG.choice(reps, size=len(reps), replace=True)
        return np.nan, np.nan
    pr._boot_by_replicate = consume_boot
    try:
        pr.responsiveness(dec)
        pr.defection_events(dec)
        pr.partner_following(dec)
    finally:
        pr.iv_slope, pr._boot_by_replicate = original_iv, original_boot
    d = dec[dec.role.eq("investor") & dec.num_agents.eq(2) & ~dec.run_set.eq("random")].copy()
    d["own_prev_usd"] = d.own_prev2_sent * 5
    d = d.dropna(subset=["send_usd", "sig_shown_usd", "own_prev_usd"])
    rows = []
    for model, g in d.groupby("model"):
        def xy(z):
            return np.column_stack([np.ones(len(z)), z.task_order.eq("game_myth"), z.task_order.eq("myth_game"), z.sig_shown_usd, z.own_prev_usd]).astype(float), z.send_usd.to_numpy(float)
        X, y = xy(g)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        parts = [xy(s) for _, s in g.groupby(["replicate", "noise"])]
        xx = np.array([x.T @ x for x, y in parts])
        xyv = np.array([x.T @ y for x, y in parts])
        draw = np.array([pr.RNG.integers(0, len(parts), len(parts)) for _ in range(pr.N_BOOT)])
        A, B = xx[draw].sum(axis=1), xyv[draw].sum(axis=1)
        bs = np.einsum("nij,nj->ni", np.linalg.pinv(A), B)
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        for j, term in [(1, "game_myth"), (2, "myth_game")]:
            rows.append({"model": model, "term": term, "coef": coef[j], "ci_lo": lo[j], "ci_hi": hi[j]})
    out = pd.DataFrame(rows)
    old = pd.read_csv(ROOT / "docs/figures/partner_responsiveness_20260930/myth_push.csv")
    for r in old.itertuples():
        for order in ["game_myth", "myth_game"]:
            row = out[out.model.eq(r.model) & out.term.eq(order)].iloc[0]
            for new, suffix in [("coef", ""), ("ci_lo", "_lo"), ("ci_hi", "_hi")]:
                assert abs(row[new] - getattr(r, order + "_push" + suffix)) <= .000501, (r.model, order, new, row[new], getattr(r, order + "_push" + suffix))
    write("push_original_bootstrap", out)


def examples():
    sources = []
    for dirname in ["linguistic_n10_20261001", "linguistic_frontier_20260930"]:
        d = pd.read_csv(ROOT / "data/analysis" / dirname / "myths.csv")
        d["corpus"] = dirname
        sources.append(d)
    myths = pd.concat(sources, ignore_index=True)
    fragments = ["Lir placed five suns", "three sacks of grain", "Sender, offered up five coins", "seeds locked in a fist", "weavers return half the harvest", "If Solara offered with restraint"]
    expected = [("GPT", 1, 5, 71.0), ("Sonnet", 1, 3, 73.4), ("GPT", 1, 5, 74.3), ("Sol", 1, 5, 75.0), ("Gemini", 5, 5, 72.0), ("GPT", 5, 0, 29.8)]
    rows = []
    for i, (fragment, (family, rnd, sent, resources)) in enumerate(zip(fragments, expected), 1):
        matches = myths[myths.text.fillna("").str.contains(fragment, regex=False) & myths.family.eq(family) & myths["round"].eq(rnd)]
        for row in matches.itertuples():
            run = json.loads((ROOT / row.path).read_text())
            history = run["conversation_history"]
            # Same round is available for myth-first; the next round for game-first.
            earliest = row.round if row.task_order == "myth_game" else row.round + 1
            candidates = [(h["round"], float(d["sent"])) for h in history if h["round"] >= earliest for d in h["dyads"] if d["investor"] == row.agent]
            assert candidates
            play_round, amount = candidates[0]
            resource = sum(history[-1]["balances"].values()) / len(run["agents"])
            rows.append({"table_row": i, "fragment": fragment, "run_id": row.run_id, "path": row.path, "agent": row.agent,
                         "myth_round": row.round, "next_send_round": play_round, "next_send": amount, "final_resources_per_agent": resource,
                         "matches_printed_amount_and_resource": bool(amount == sent and abs(resource-resources) < .051)})
    out = pd.DataFrame(rows)
    assert set(out.table_row) == set(range(1, 7)), out
    assert out.groupby("table_row").matches_printed_amount_and_resource.any().all(), out
    write("table1_examples", out)


def injection():
    from analyses.plot_slide678_rerun import load_verified_values
    root = ROOT / "data/json/noise_experiments/slide678_dyad_rerun_20260917"
    vals, receipt, sources = load_verified_values(root, root / "completion_receipt.json")
    rows = []
    for cell, v in vals.items():
        y = np.asarray(v)/2
        rows.append({"cell": cell, "n_runs": len(y), "mean_per_agent": y.mean(), "sd_per_agent": y.std(ddof=1), "values_per_agent": list(y)})
    write("injection_resources", pd.DataFrame(rows))
    write("injection_integrity", {"n_finals": len(sources), "receipt_hashes_match": True, "completion_receipt_sha256": hashlib.sha256((root / "completion_receipt.json").read_bytes()).hexdigest()})


def main():
    info = {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__}
    summaries, model_rows, sample_rows = {}, [], []
    for tag, dirname, rule in [("september", "linguistic_20260923", "myth_rules_september_z-ai__glm-5.2.csv"),
                                ("frontier", "linguistic_frontier_20260930", "myth_rules_frontier_z-ai__glm-5.2.csv")]:
        p = ROOT / "data/analysis" / dirname
        d, m, r = pd.read_csv(p / "decisions.csv"), pd.read_csv(p / "myths.csv"), pd.read_csv(p / rule)
        summaries[tag] = check_raw(d, m, tag)
        a, b = round1(d, r, tag)
        model_rows += a; sample_rows += b
        if tag == "september":
            mediation(d)
        print(tag, summaries[tag], flush=True)
    write("round1_named_models", pd.DataFrame(model_rows))
    write("round1_named_descriptives", pd.DataFrame(sample_rows))
    replay(); print("replay reproduced", flush=True)
    push(); print("push inputs and coefficients reproduced", flush=True)
    original_push_bootstrap(); print("original 2,000-draw push intervals reproduced", flush=True)
    injection(); print("injection reproduced", flush=True)
    examples(); print("Table 1 examples reproduced", flush=True)
    write("audit_run", {"versions": info, "datasets": summaries, "all_assertions_passed": True})


if __name__ == "__main__":
    main()

"""Does myth convergence relate to trust-game cooperation?

Coauthor question on the "dyads converge, populations diverge" result:
does stronger convergence predict higher cooperation in gameplay?

Three levels of analysis:
  A. Run level: per-run similarity slope (from the existing
     between_agent_convergence_raw.csv, mpnet mean pairwise cosine) vs
     per-run mean cooperation (sent fraction, return fraction).
  B. Round level within runs: round-t mean similarity vs round-t mean
     cooperation, demeaned within run (and lagged both directions).
  C. Dyad level within 8-agent runs: for each game pairing (i, j) in
     round t, cosine similarity of i's and j's round-t myths (written
     just before that game) vs sent/return fraction in that game.
     Run fixed effects via within-run demeaning.

Outputs to data/plots/linguistic_analysis/_convergence_vs_cooperation/.
Run from repo root: python3 analyses/convergence_vs_cooperation.py
"""

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

RAW_SIM_CSV = "data/plots/linguistic_analysis/_condition_comparison/between_agent_convergence_raw.csv"
OUT_DIR = "data/plots/linguistic_analysis/_convergence_vs_cooperation"
ENDOWMENT = 5.0

# (exp label in CSV, cond) -> glob for the simulation JSONs
RUN_GROUPS = {
    ("sonnet45_2agent_dir_vs_norm", "directive"): (
        "dyad_directive",
        "data/json/sonnet45_directive_normative_r10_n5/claude-sonnet-4.5/myth_game/*myth_game_directive_anything.json",
    ),
    ("sonnet45_2agent_dir_vs_norm", "normative"): (
        "dyad_normative",
        "data/json/sonnet45_directive_normative_r10_n5/claude-sonnet-4.5/myth_game/*myth_game_normative_anything.json",
    ),
    ("sonnet45_8agent_named", "directive"): (
        "pop8_named",
        "data/json/sonnet45_8agent_myth_directive_history3_r10_n5/claude-sonnet-4.5/myth_game/*.json",
    ),
    ("sonnet45_8agent_anon", "directive"): (
        "pop8_anon",
        "data/json/sonnet45_8agent_myth_directive_history3_anon_r10_n5/claude-sonnet-4.5/myth_game/*.json",
    ),
    ("sonnet45_8agent_game_directive", "directive"): (
        "pop8_game_directive",
        "data/json/sonnet45_8agent_game_directive_r10_n5/claude-sonnet-4.5/myth_game/*.json",
    ),
}


def load_similarity_by_run():
    """{file_stem: {round: mean_pairwise_sim}} restricted to RUN_GROUPS."""
    wanted = {(exp, cond) for (exp, cond) in RUN_GROUPS}
    sims = defaultdict(dict)
    labels = {}
    with open(RAW_SIM_CSV) as fh:
        for row in csv.DictReader(fh):
            key = (row["exp"], row["cond"])
            if key not in wanted:
                continue
            stem = row["file"]
            sims[stem][int(row["round"])] = float(row["mean_pairwise_sim"])
            labels[stem] = RUN_GROUPS[key][0]
    return sims, labels


def iter_dyads(entry):
    dyads = entry.get("dyads") or []
    if dyads:
        for d in dyads:
            if d.get("sent") is not None and d.get("returned") is not None:
                yield d
        return
    roles = entry.get("roles") or {}
    inv = next((a for a, r in roles.items() if r == "investor"), None)
    tru = next((a for a, r in roles.items() if r == "trustee"), None)
    if inv and tru and entry.get("sent") is not None and entry.get("returned") is not None:
        yield {
            "agents": [inv, tru], "investor": inv, "trustee": tru,
            "sent": entry["sent"], "received": entry["received"],
            "returned": entry["returned"],
        }


def load_game_rounds(path):
    """[(round, [(investor, trustee, sent_frac, ret_frac), ...])]"""
    data = json.load(open(path))
    out = []
    for entry in data["conversation_history"]:
        rnd = int(entry["round"])
        dyads = []
        for d in iter_dyads(entry):
            sent_frac = d["sent"] / ENDOWMENT
            received = d.get("received")
            ret_frac = (d["returned"] / received) if received else None
            dyads.append((d["investor"], d["trustee"], sent_frac, ret_frac))
        out.append((rnd, dyads))
    return out, data


def slope(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return np.polyfit(xs, ys, 1)[0]


def corr_report(name, x, y, fh):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        line = f"{name}: n={len(x)} (insufficient/degenerate)"
    else:
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        line = (f"{name}: n={len(x)}  pearson r={pr:+.3f} (p={pp:.3f})  "
                f"spearman rho={sr:+.3f} (p={sp:.3f})")
    print(line)
    fh.write(line + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sims, labels = load_similarity_by_run()

    # ---- assemble run-level table -------------------------------------
    runs = []  # one dict per run
    myth_texts = {}  # stem -> {round: {agent: myth}}
    seen = set()
    for (exp, cond), (group, pattern) in RUN_GROUPS.items():
        for path in sorted(glob.glob(pattern)):
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem.endswith(".results") or stem in seen:
                continue
            if stem not in sims:
                print(f"WARNING: no similarity rows for {stem}, skipping")
                continue
            seen.add(stem)
            rounds, data = load_game_rounds(path)
            sent, ret, per_round = [], [], {}
            for rnd, dyads in rounds:
                s = [d[2] for d in dyads]
                r = [d[3] for d in dyads if d[3] is not None]
                sent += s
                ret += r
                per_round[rnd] = (np.mean(s) if s else np.nan,
                                  np.mean(r) if r else np.nan)
            sim_rounds = sorted(sims[stem])
            runs.append({
                "group": group, "stem": stem, "path": path,
                "sim_slope": slope(sim_rounds, [sims[stem][r] for r in sim_rounds]),
                "sim_mean": float(np.mean([sims[stem][r] for r in sim_rounds])),
                "sim_r1": sims[stem][min(sim_rounds)],
                "sim_r10": sims[stem][max(sim_rounds)],
                "sent_frac": float(np.mean(sent)),
                "ret_frac": float(np.mean(ret)),
                "sent_slope": slope([r for r, _ in rounds],
                                    [per_round[r][0] for r, _ in rounds]),
                "ret_slope": slope([r for r, _ in rounds],
                                   [per_round[r][1] for r, _ in rounds]),
                "per_round": per_round,
                "game_rounds": rounds,
            })
            myth_texts[stem] = {
                int(e["round"]): dict(e.get("myths") or {})
                for e in data["conversation_history"]
            }

    with open(os.path.join(OUT_DIR, "run_level.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "run", "sim_slope", "sim_mean", "sim_r1", "sim_r10",
                    "sent_frac", "ret_frac", "sent_slope", "ret_slope"])
        for r in runs:
            w.writerow([r["group"], r["stem"]] +
                       [f"{r[k]:.4f}" for k in
                        ("sim_slope", "sim_mean", "sim_r1", "sim_r10",
                         "sent_frac", "ret_frac", "sent_slope", "ret_slope")])

    rep = open(os.path.join(OUT_DIR, "report.txt"), "w")

    def section(title):
        bar = "=" * 70
        print(f"\n{bar}\n{title}\n{bar}")
        rep.write(f"\n{bar}\n{title}\n{bar}\n")

    section("Run-level summary by group")
    for group in ["dyad_directive", "dyad_normative", "pop8_named",
                  "pop8_anon", "pop8_game_directive"]:
        sub = [r for r in runs if r["group"] == group]
        if not sub:
            continue
        line = (f"{group:22s} n={len(sub)}  "
                f"sim_slope={np.mean([r['sim_slope'] for r in sub]):+.4f}  "
                f"sent_frac={np.mean([r['sent_frac'] for r in sub]):.3f}  "
                f"ret_frac={np.mean([r['ret_frac'] for r in sub]):.3f}  "
                f"sent_slope={np.mean([r['sent_slope'] for r in sub]):+.4f}")
        print(line)
        rep.write(line + "\n")

    section("A. Run level: similarity slope vs cooperation")
    groups = {
        "dyads (directive, n=5)": ["dyad_directive"],
        "dyads (directive+normative, n=10)": ["dyad_directive", "dyad_normative"],
        "8-agent pooled (n=15)": ["pop8_named", "pop8_anon", "pop8_game_directive"],
        "all sonnet runs (n=25)": ["dyad_directive", "dyad_normative", "pop8_named",
                                   "pop8_anon", "pop8_game_directive"],
    }
    for gname, gs in groups.items():
        sub = [r for r in runs if r["group"] in gs]
        rep.write(f"\n[{gname}]\n")
        print(f"\n[{gname}]")
        corr_report("  sim_slope vs sent_frac", [r["sim_slope"] for r in sub],
                    [r["sent_frac"] for r in sub], rep)
        corr_report("  sim_slope vs ret_frac", [r["sim_slope"] for r in sub],
                    [r["ret_frac"] for r in sub], rep)
        corr_report("  sim_slope vs sent_slope", [r["sim_slope"] for r in sub],
                    [r["sent_slope"] for r in sub], rep)
        corr_report("  sim_mean  vs sent_frac", [r["sim_mean"] for r in sub],
                    [r["sent_frac"] for r in sub], rep)
        corr_report("  sim_mean  vs ret_frac", [r["sim_mean"] for r in sub],
                    [r["ret_frac"] for r in sub], rep)

    section("B. Round level within runs (demeaned within run)")
    for gname, gs in [("dyads directive", ["dyad_directive"]),
                      ("8-agent pooled", ["pop8_named", "pop8_anon",
                                          "pop8_game_directive"])]:
        sub = [r for r in runs if r["group"] in gs]
        same_s, same_c = [], []          # sim_t vs coop_t
        lag_s, lag_c = [], []            # sim_{t-1} -> coop_t
        rev_s, rev_c = [], []            # coop_{t-1} -> sim_t (behaviour -> myth)
        det_s, det_c = [], []            # both linearly detrended within run

        def detrend(ts, vs):
            ts, vs = np.asarray(ts, float), np.asarray(vs, float)
            return vs - np.polyval(np.polyfit(ts, vs, 1), ts)

        for r in sub:
            stem_sims = sims[r["stem"]]
            rounds_avail = sorted(set(stem_sims) & set(r["per_round"]))
            sim_v = np.array([stem_sims[t] for t in rounds_avail])
            coop_v = np.array([r["per_round"][t][0] for t in rounds_avail])
            sim_d, coop_d = sim_v - sim_v.mean(), coop_v - coop_v.mean()
            same_s += list(sim_d)
            same_c += list(coop_d)
            det_s += list(detrend(rounds_avail, sim_v))
            det_c += list(detrend(rounds_avail, coop_v))
            for i, t in enumerate(rounds_avail):
                if t - 1 in rounds_avail:
                    j = rounds_avail.index(t - 1)
                    lag_s.append(sim_d[j]); lag_c.append(coop_d[i])      # sim_{t-1} -> coop_t
                    rev_s.append(coop_d[j]); rev_c.append(sim_d[i])      # coop_{t-1} -> sim_t
        rep.write(f"\n[{gname}]\n")
        print(f"\n[{gname}]")
        corr_report("  sim_t     vs sent_frac_t  ", same_s, same_c, rep)
        corr_report("  sim_(t-1) vs sent_frac_t  ", lag_s, lag_c, rep)
        corr_report("  sent_frac_(t-1) vs sim_t  ", rev_s, rev_c, rep)
        corr_report("  detrended sim_t vs sent_frac_t", det_s, det_c, rep)

    # ---- C. dyad level: needs embeddings ------------------------------
    section("C. Dyad level in 8-agent runs: pair myth similarity vs pair game behaviour")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")

    pair_rows = []
    pop_runs = [r for r in runs if r["group"].startswith("pop8")]
    for r in pop_runs:
        texts, keys = [], []
        for rnd, agents in myth_texts[r["stem"]].items():
            for aid, myth in agents.items():
                texts.append(myth)
                keys.append((rnd, aid))
        emb = model.encode(texts, normalize_embeddings=True,
                           show_progress_bar=False)
        emb_by_key = {k: e for k, e in zip(keys, emb)}
        for rnd, dyads in r["game_rounds"]:
            for inv, tru, sent_frac, ret_frac in dyads:
                e1, e2 = emb_by_key.get((rnd, inv)), emb_by_key.get((rnd, tru))
                ep1, ep2 = emb_by_key.get((rnd - 1, inv)), emb_by_key.get((rnd - 1, tru))
                sim_t = float(np.dot(e1, e2)) if e1 is not None and e2 is not None else np.nan
                sim_tm1 = float(np.dot(ep1, ep2)) if ep1 is not None and ep2 is not None else np.nan
                pair_rows.append({
                    "group": r["group"], "stem": r["stem"], "round": rnd,
                    "investor": inv, "trustee": tru,
                    "pair_sim_t": sim_t, "pair_sim_tm1": sim_tm1,
                    "sent_frac": sent_frac,
                    "ret_frac": ret_frac if ret_frac is not None else np.nan,
                })

    with open(os.path.join(OUT_DIR, "pair_level.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

    def demean(rows, xkey, ykey, cell):
        by_cell = defaultdict(list)
        for row in rows:
            if not (np.isnan(row[xkey]) or np.isnan(row[ykey])):
                by_cell[cell(row)].append(row)
        xs, ys = [], []
        for rr in by_cell.values():
            if len(rr) < 2:
                continue
            xv = np.array([q[xkey] for q in rr])
            yv = np.array([q[ykey] for q in rr])
            xs += list(xv - xv.mean())
            ys += list(yv - yv.mean())
        return xs, ys

    for what, xkey in [("same-round myth sim", "pair_sim_t"),
                       ("prev-round myth sim", "pair_sim_tm1")]:
        for fe_name, cell in [("run FE", lambda q: q["stem"]),
                              ("run x round FE", lambda q: (q["stem"], q["round"]))]:
            rep.write(f"\n[{what}, {fe_name} (demeaned)]\n")
            print(f"\n[{what}, {fe_name} (demeaned)]")
            for ykey in ("sent_frac", "ret_frac"):
                xs, ys = demean(pair_rows, xkey, ykey, cell)
                corr_report(f"  {xkey} vs {ykey}", xs, ys, rep)

    # raw (no FE) for reference
    rep.write("\n[raw pooled, no fixed effects]\n")
    print("\n[raw pooled, no fixed effects]")
    for ykey in ("sent_frac", "ret_frac"):
        ok = [q for q in pair_rows
              if not (np.isnan(q["pair_sim_t"]) or np.isnan(q[ykey]))]
        corr_report(f"  pair_sim_t vs {ykey}",
                    [q["pair_sim_t"] for q in ok], [q[ykey] for q in ok], rep)

    rep.close()
    print(f"\nWrote {OUT_DIR}/run_level.csv, pair_level.csv, report.txt")


if __name__ == "__main__":
    main()

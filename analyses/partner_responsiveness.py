"""Do myths make agents react more or less to their partner's game moves?

Reads the decision table written by partner_responsiveness_extract.py.

1. Reaction to the partner (causal). The amount an agent is shown of its
   partner's move is shifted by chance: communication noise (every noisy run)
   and random forced defection (2026-09-09 dyads). Using only that chance
   variation as the instrument, we estimate how much the agent's own move
   changes per unit of the partner's shown move:

   investor   (dyads)  own send ($) per $1 of the partner's shown send last round
   trustee_now (all)   share returned, in points, per $1 of the send just shown
   trustee_prev (dyads) share returned, in points, per 10 points of the share the
                       partner was shown to return last round

   Estimates use run fixed effects and are pooled, within one model and
   population size, over the noise conditions and random-defection arms that
   carry the instrument. Uncertainty: bootstrap over replicates, resampled
   jointly across task orders because noise draws and forced defections are
   shared by replicate. The draw that would have been applied in no-noise runs
   gives a placebo, which must show no effect.

2. Opening versus later rounds. Round 1 sends happen before any partner
   information. The myth effect (task order minus game only) is split into
   the round-1 part and the rounds 2-10 part.

   Also: defection_events (next send after a forced $0, plain means) and
   partner_following (correlational slope of send on partner's last send),
   defection_placebo (within-run shuffle of the forced-$0 labels), myth_push
   (extra send per round at the same partner and own history), gap_by_round.

3. What Sonnet 4.5 writes. Share of game rationales that mention the
   partner's moves and share that mention a myth (only Sonnet writes prose).

Usage (from repo root):
  python analyses/partner_responsiveness_extract.py
  python analyses/partner_responsiveness.py

The second step also writes provenance.json for the folder: the 450 run
finals listed by the extract script's manifests, with the allowed differences
of the frontier rerun plus the noise-regime factor of Figure 2. decisions.csv
is a local intermediate (gitignored) and is not listed as an output.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "figures" / "partner_responsiveness_20260930"
ORDERS = ["game", "game_myth", "myth_game"]
N_BOOT = 2000
RNG = np.random.default_rng(20260930)
INTERMEDIATES = {"decisions.csv"}
SOURCE_MANIFESTS = [
    "docs/figures/figure2_noise_comparison_20260916/provenance.json",
    "docs/figures/negative_only_crossmodel_reasoning_rerun_20260909/provenance.json",
    "docs/figures/frontier_rerun_20260918/provenance.json",
]


def load():
    dec = pd.read_csv(OUT_DIR / "decisions.csv", low_memory=False)
    dec = dec[dec.own_source == "llm"].copy()
    for col in ["sig_partner_forced", "own_prev_forced", "prev_ret_partner_forced"]:
        dec[col] = dec[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    dec["send_usd"] = np.where(dec.role == "investor", dec.decision_frac * 5, np.nan)
    dec["sig_shown_usd"] = dec.sig_shown * 5
    dec["sig_actual_usd"] = dec.sig_actual * 5
    return dec


# --- instrumental-variable slope -------------------------------------------

def run_arrays(df, y, x, zs, ws):
    """Per-run arrays after removing run means (run fixed effects)."""
    cols = [y, x] + zs + ws
    d = df[cols + ["path"]].copy()
    d[zs] = d[zs].fillna(0.0)  # an instrument absent from a run set is constant there
    d = d.dropna()
    out = {}
    for path, sub in d.groupby("path"):
        m = sub[cols].to_numpy(float)
        out[path] = m - m.mean(axis=0)
    return out


def iv_slope(arrays, keys, nz):
    """2SLS slope of y on x using nz instruments; columns: y, x, z..., w..."""
    if not keys:
        return np.nan, np.nan
    m = np.vstack([arrays[k] for k in keys])
    y, x, Z, W = m[:, 0], m[:, 1], m[:, 2:2 + nz], m[:, 2 + nz:]
    if W.shape[1]:
        proj = np.linalg.lstsq(W, np.column_stack([y, x, Z]), rcond=None)[0]
        r = np.column_stack([y, x, Z]) - W @ proj
        y, x, Z = r[:, 0], r[:, 1], r[:, 2:]
    keep = Z.std(axis=0) > 1e-9
    Z = Z[:, keep]
    if Z.shape[1] == 0 or x.std() < 1e-9:
        return np.nan, np.nan
    gamma = np.linalg.lstsq(Z, x, rcond=None)[0]
    xhat = Z @ gamma
    if xhat @ x < 1e-9:
        return np.nan, np.nan
    first_stage_r2 = (xhat @ xhat) / (x @ x)
    return float((xhat @ y) / (xhat @ x)), float(first_stage_r2)


def iv_table(frame, name, y, x, zs, ws, scale):
    rows = []
    for (model, n_agents), g in frame.groupby(["model", "num_agents"]):
        arrays = run_arrays(g, y, x, zs, ws)
        runs = {}
        for (o, r), sub in g.groupby(["task_order", "replicate"]):
            runs[(o, r)] = [p for p in sub.path.unique() if p in arrays]
        reps = sorted(g.replicate.unique())
        point = {o: iv_slope(arrays, [p for r in reps for p in runs.get((o, r), [])], len(zs)) for o in ORDERS}
        boots = {o: np.full(N_BOOT, np.nan) for o in ORDERS}
        for b in range(N_BOOT):
            draw = RNG.choice(reps, size=len(reps), replace=True)
            for o in ORDERS:
                boots[o][b] = iv_slope(arrays, [p for r in draw for p in runs.get((o, r), [])], len(zs))[0]
        row = {"measure": name, "model": model, "num_agents": n_agents,
               "run_sets": "+".join(sorted(g.run_set.unique()))}
        for o in ORDERS:
            sub = g[g.task_order == o]
            ok = np.isfinite(boots[o])
            row[f"{o}_slope"] = point[o][0] * scale
            row[f"{o}_lo"], row[f"{o}_hi"] = (np.percentile(boots[o][ok], [2.5, 97.5]) * scale) if ok.mean() > 0.8 else (np.nan, np.nan)
            row[f"{o}_first_stage_r2"] = point[o][1]
            row[f"{o}_n"] = len(sub)
            row[f"{o}_mean_y"] = sub[y].mean()
        for o in ["game_myth", "myth_game"]:
            diff = (boots[o] - boots["game"]) * scale
            ok = np.isfinite(diff)
            row[f"{o}_minus_game"] = (point[o][0] - point["game"][0]) * scale
            if ok.mean() > 0.8:
                row[f"{o}_minus_game_lo"], row[f"{o}_minus_game_hi"] = np.percentile(diff[ok], [2.5, 97.5])
            else:
                row[f"{o}_minus_game_lo"] = row[f"{o}_minus_game_hi"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def responsiveness(dec):
    instrumented = dec[(dec.noise != "no_noise")]
    inv = instrumented[(instrumented.role == "investor") & (instrumented.num_agents == 2)
                       & instrumented.sig_partner_forced.notna() & (instrumented.own_prev_forced == 0)].copy()
    inv["z_forced"] = inv.sig_partner_forced
    inv["z_noise"] = inv.sig_u
    tru_now = instrumented[(instrumented.role == "trustee") & (instrumented.sig_partner_forced == 0)
                           & instrumented.decision_frac.notna()].copy()
    tru_now["z_noise"] = tru_now.sig_u
    tru_prev = instrumented[(instrumented.role == "trustee") & (instrumented.num_agents == 2)
                            & (instrumented.sig_partner_forced == 0) & (instrumented.own_prev_forced == 0)
                            & instrumented.prev_ret_shown.notna() & instrumented.decision_frac.notna()].copy()
    tru_prev["z_forced"] = tru_prev.prev_ret_partner_forced
    tru_prev["z_noise"] = tru_prev.prev_ret_u
    placebo = dec[(dec.noise == "no_noise") & (dec.role == "trustee") & dec.decision_frac.notna()].copy()
    placebo["x_placebo"] = placebo.placebo_u  # reduced form: the draw itself
    placebo["z_noise"] = placebo.placebo_u
    tables = [
        iv_table(inv, "investor_send_per_usd", "send_usd", "sig_shown_usd", ["z_forced", "z_noise"], [], 1.0),
        iv_table(tru_now, "trustee_share_pts_per_usd_sent_now", "decision_frac", "sig_shown_usd", ["z_noise"], ["sig_actual_usd"], 100.0),
        iv_table(tru_prev, "trustee_share_pts_per_10pts_prev_return", "decision_frac", "prev_ret_shown", ["z_forced", "z_noise"], [], 10.0),
        iv_table(placebo, "placebo_trustee_share_pts_per_usd_unapplied_draw", "decision_frac", "x_placebo", ["z_noise"], ["sig_actual_usd"], 100.0),
    ]
    return pd.concat(tables, ignore_index=True)


# --- opening versus later rounds -------------------------------------------

def opening_split(dec):
    main = dec[(dec.role == "investor") & (
        ((dec.run_set == "figure2") & (dec.noise == "noise_informed")) | (dec.run_set == "frontier"))]
    per_run = (main.assign(phase=np.where(main["round"] == 1, "round1", "rounds2_10"))
               .groupby(["model", "num_agents", "task_order", "replicate", "phase"]).send_usd.mean()
               .unstack("phase").reset_index())
    rows = []
    for (model, n), g in per_run.groupby(["model", "num_agents"]):
        row = {"model": model, "num_agents": n}
        for o in ORDERS:
            s = g[g.task_order == o]
            for ph in ["round1", "rounds2_10"]:
                row[f"{o}_{ph}_mean"] = s[ph].mean()
                row[f"{o}_{ph}_std"] = s[ph].std()
        for o in ["game_myth", "myth_game"]:
            for ph in ["round1", "rounds2_10"]:
                row[f"{o}_minus_game_{ph}"] = row[f"{o}_{ph}_mean"] - row[f"game_{ph}_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


# --- what Sonnet writes ------------------------------------------------------

PARTNER_RE = re.compile(
    r"\b(?:they|their|them|co-?player|partner|the sender|the receiver|other (?:player|agent)|this player)\b"
    r"[^.\n]{0,100}\b(?:sent|send|sending|returned|return|returning|gave|kept|trust|trusted|reciprocat\w*|generos\w*|defect\w*)"
    r"|\b(?:sent|returned|gave) (?:me|us)\b|\byou (?:sent|returned)\b",
    re.I,
)
MYTH_RE = re.compile(r"\b(?:myths?|story|stories|tale|legend|parable|fable|mythic)\b", re.I)


def rationale_mentions(dec):
    s = dec[(dec.model == "Sonnet 4.5")].copy()
    text = s.text.fillna("")
    s["words"] = text.str.split().str.len()
    s = s[s.words > 8]  # bare JSON replies carry no rationale
    s["mentions_partner_moves"] = text[s.index].str.contains(PARTNER_RE)
    s["mentions_myth"] = text[s.index].str.contains(MYTH_RE)
    per_run = s.groupby(["run_set", "num_agents", "task_order", "path"]).agg(
        partner=("mentions_partner_moves", "mean"), myth=("mentions_myth", "mean"),
        words=("words", "median"), n=("words", "size")).reset_index()
    return per_run.groupby(["run_set", "num_agents", "task_order"]).agg(
        runs=("path", "size"), decisions=("n", "sum"),
        partner_mean=("partner", "mean"), partner_std=("partner", "std"),
        myth_mean=("myth", "mean"), myth_std=("myth", "std"),
        median_words=("words", "median")).reset_index()


def _boot_by_replicate(g, fn):
    parts = {r: sub for r, sub in g.groupby("replicate")}
    reps = sorted(parts)
    out = []
    for _ in range(N_BOOT):
        draw = RNG.choice(reps, size=len(reps), replace=True)
        out.append(fn(pd.concat([parts[r].assign(path=parts[r].path + f"#{i}") for i, r in enumerate(draw)])))
    out = np.array(out, float)
    return np.nanpercentile(out, [2.5, 97.5]) if np.isfinite(out).mean() > 0.8 else (np.nan, np.nan)


def defection_events(dec):
    """Next send after the partner was forced to send $0, vs after a chosen send."""
    d = _defection_frame(dec)

    def drop(g):
        a, b = g[g.sig_partner_forced == 1], g[g.sig_partner_forced == 0]
        return a.send_usd.mean() - b.send_usd.mean()

    rows = []
    for (model, o), g in d.groupby(["model", "task_order"]):
        after, other = g[g.sig_partner_forced == 1], g[g.sig_partner_forced == 0]
        partner_gap = other.sig_shown_usd.mean() - after.sig_shown_usd.mean()
        lo, hi = _boot_by_replicate(g, drop)
        rows.append({"model": model, "task_order": o, "events": len(after), "other_decisions": len(other),
                     "send_after_chosen_partner_send": other.send_usd.mean(),
                     "send_after_forced_zero": after.send_usd.mean(),
                     "change": drop(g), "change_lo": lo, "change_hi": hi,
                     "partner_shown_gap": partner_gap,
                     "change_per_usd_partner_gap": -drop(g) / partner_gap if partner_gap > 0 else np.nan})
    return pd.DataFrame(rows)


def partner_following(dec):
    """Dyad investors: own send ($) per $1 of the partner's last shown send.

    Correlational (all chosen variation), run and round means removed. No-defector
    runs only (figure2 all noise conditions, frontier).
    """
    d = dec[(dec.role == "investor") & (dec.num_agents == 2) & (dec.run_set != "random")
            & dec.sig_shown_usd.notna()]

    def slope(g):
        y = g.send_usd - g.groupby("path").send_usd.transform("mean")
        x = g.sig_shown_usd - g.groupby("path").sig_shown_usd.transform("mean")
        y, x = y - y.groupby(g["round"]).transform("mean"), x - x.groupby(g["round"]).transform("mean")
        return (x * y).sum() / (x * x).sum() if (x * x).sum() > 1e-9 else np.nan

    rows = []
    for (model, o), g in d.groupby(["model", "task_order"]):
        lo, hi = _boot_by_replicate(g, slope)
        rows.append({"model": model, "task_order": o, "runs": g.path.nunique(),
                     "mean_send": g.send_usd.mean(), "within_run_send_sd": g.groupby("path").send_usd.std().mean(),
                     "follow_slope": slope(g), "follow_lo": lo, "follow_hi": hi})
    return pd.DataFrame(rows)


def myth_push(dec):
    """Dyad sends given the same history: send ~ task order + partner's last shown
    send + own previous send. The task-order terms are the extra send per round
    that history does not explain. Bootstrap over runs' (replicate, noise) cells.
    """
    d = dec[(dec.role == "investor") & (dec.num_agents == 2) & (dec.run_set != "random")].copy()
    d["own_prev_usd"] = d.own_prev2_sent * 5
    d = d.dropna(subset=["send_usd", "sig_shown_usd", "own_prev_usd"])

    def fit(g):
        X = np.column_stack([np.ones(len(g)), g.task_order.eq("game_myth"), g.task_order.eq("myth_game"),
                             g.sig_shown_usd, g.own_prev_usd]).astype(float)
        return np.linalg.lstsq(X, g.send_usd.to_numpy(float), rcond=None)[0]

    rows = []
    for model, g in d.groupby("model"):
        b = fit(g)
        parts = [sub for _, sub in g.groupby(["replicate", "noise"])]
        bs = np.array([fit(pd.concat([parts[i] for i in RNG.integers(0, len(parts), len(parts))]))
                       for _ in range(N_BOOT)])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        rows.append({"model": model, "decisions": len(g),
                     "game_myth_push": b[1], "game_myth_push_lo": lo[1], "game_myth_push_hi": hi[1],
                     "myth_game_push": b[2], "myth_game_push_lo": lo[2], "myth_game_push_hi": hi[2],
                     "per_usd_partner_last_send": b[3], "per_usd_own_previous_send": b[4]})
    return pd.DataFrame(rows)


def gap_by_round(dec):
    main = dec[(dec.role == "investor") & (
        ((dec.run_set == "figure2") & (dec.noise == "noise_informed")) | (dec.run_set == "frontier"))]
    t = main.groupby(["model", "num_agents", "task_order", "round"]).send_usd.mean().unstack("task_order")
    t["myth_game_minus_game"] = t.myth_game - t.game
    t["game_myth_minus_game"] = t.game_myth - t.game
    return t.reset_index()


def _defection_frame(dec):
    """Dyad investor decisions in the random-defection runs. Rows whose partner's
    return two rounds earlier was forced are dropped: that is also a betrayal and
    would contaminate the comparison group."""
    d = dec[(dec.run_set == "random") & (dec.role == "investor") & (dec.own_prev_forced == 0)
            & dec.sig_partner_forced.notna()].copy()
    ret2 = d.ret2_partner_forced.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    return d[ret2.fillna(0) == 0]


def defection_placebo(dec, n_perm=5000):
    """Chance range for defection_events, inside the same runs: shuffle which
    decisions count as 'after a forced $0' within each run, keep the rest.
    Ignores that the 25% and 50% arms share events, so p-values are optimistic.
    """
    d = _defection_frame(dec)
    rows = []
    for (model, o), g in d.groupby(["model", "task_order"]):
        y = g.send_usd.to_numpy(float)
        ev = g.sig_partner_forced.to_numpy(float)
        groups = [np.flatnonzero(g.path.to_numpy() == p) for p in g.path.unique()]
        observed = y[ev == 1].mean() - y[ev == 0].mean()
        sims = np.empty(n_perm)
        for k in range(n_perm):
            e = ev.copy()
            for idx in groups:
                e[idx] = RNG.permutation(ev[idx])
            sims[k] = y[e == 1].mean() - y[e == 0].mean() if 0 < e.sum() < len(e) else np.nan
        sims = sims[np.isfinite(sims)]
        rows.append({"model": model, "task_order": o, "observed_change": observed,
                     "chance_lo": np.percentile(sims, 2.5), "chance_hi": np.percentile(sims, 97.5),
                     "p_two_sided": float((np.abs(sims) >= abs(observed) - 1e-12).mean())})
    return pd.DataFrame(rows)


def judge_summary():
    """Per-run shares from partner_responsiveness_judge.py, summarised per cell."""
    path = OUT_DIR / "sonnet_rationale_judge.csv"
    if not path.exists():
        return None
    j = pd.read_csv(path, low_memory=False)
    for col in ["partner_basis", "myth_basis"]:
        j[col] = j[col].astype(str).str.lower().eq("true")
    for v in ["partner", "myth", "both", "other"]:
        j[f"driver_{v}"] = j.primary_driver.eq(v)
    for v in ["conditional", "unconditional"]:
        j[f"rule_{v}"] = j.myth_rule.eq(v).where(j.myth_basis)
    cols = ["partner_basis", "myth_basis", "driver_partner", "driver_myth", "driver_both",
            "driver_other", "rule_conditional", "rule_unconditional"]
    j[cols] = j[cols].astype(float)
    per_run = j.groupby(["num_agents", "task_order", "path"])[cols].mean().reset_index()
    out = per_run.groupby(["num_agents", "task_order"])[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out.insert(0, "runs", per_run.groupby(["num_agents", "task_order"]).size())
    out.insert(1, "decisions", j.groupby(["num_agents", "task_order"]).size())
    return out.reset_index()


def main():
    dec = load()
    resp = responsiveness(dec)
    resp.to_csv(OUT_DIR / "responsiveness.csv", index=False, float_format="%.4f")
    opening = opening_split(dec)
    opening.to_csv(OUT_DIR / "opening_vs_later.csv", index=False, float_format="%.3f")
    text = rationale_mentions(dec)
    text.to_csv(OUT_DIR / "sonnet_rationale_mentions.csv", index=False, float_format="%.3f")
    names = ["responsiveness.csv", "opening_vs_later.csv", "sonnet_rationale_mentions.csv",
             "defection_events.csv", "partner_following.csv"]
    defection_events(dec).to_csv(OUT_DIR / "defection_events.csv", index=False, float_format="%.3f")
    partner_following(dec).to_csv(OUT_DIR / "partner_following.csv", index=False, float_format="%.3f")
    myth_push(dec).to_csv(OUT_DIR / "myth_push.csv", index=False, float_format="%.3f")
    gap_by_round(dec).to_csv(OUT_DIR / "gap_by_round.csv", index=False, float_format="%.3f")
    defection_placebo(dec).to_csv(OUT_DIR / "defection_placebo.csv", index=False, float_format="%.3f")
    names += ["myth_push.csv", "gap_by_round.csv", "defection_placebo.csv"]
    judged = judge_summary()
    if judged is not None:
        judged.to_csv(OUT_DIR / "sonnet_rationale_judge_summary.csv", index=False, float_format="%.3f")
        names.append("sonnet_rationale_judge_summary.csv")
    for name in names:
        print(f"-> {(OUT_DIR / name).relative_to(ROOT)}")
    write_provenance()


def write_provenance():
    """Record the run finals behind decisions.csv and hash every tracked output."""
    sys.path.insert(0, str(ROOT))
    from analyses.partner_responsiveness_extract import run_specs
    from scripts.analyze_frontier_rerun import ALLOWED as FRONTIER_ALLOWED
    from src.experiment_condition import output_provenance

    paths = [spec["path"] for spec in run_specs()]
    if len(paths) != len(set(paths)):
        raise SystemExit("duplicate run in the partner-responsiveness run list")
    used = set(pd.read_csv(OUT_DIR / "decisions.csv", usecols=["path"]).path)
    if used != set(paths):
        raise SystemExit(f"decisions.csv and the run list disagree: {len(used ^ set(paths))} paths differ")
    # Each final must be the file its source manifest hashed.
    recorded = {}
    for manifest in SOURCE_MANIFESTS:
        for run in json.loads((ROOT / manifest).read_text())["runs"]:
            recorded["data/json/" + run["path"].split("data/json/", 1)[1]] = run["sha256"]
    for path in paths:
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != recorded.get(path):
            raise SystemExit(f"{path}: not the final its source manifest recorded")
    allowed = {
        **FRONTIER_ALLOWED,
        "protocol.game.noise_config": "Noise regime (none, uninformed, informed) is a design factor of the "
                                      "Figure 2 runs; the analysis pools over it within model and population size.",
    }
    outputs = sorted(p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "provenance.json"
                     and p.name not in INTERMEDIATES and not p.name.startswith("."))
    document = output_provenance([(ROOT / p).resolve() for p in paths], outputs, allowed, output_root=OUT_DIR)
    (OUT_DIR / "provenance.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"provenance: {len(paths)} runs, {len(outputs)} outputs -> {(OUT_DIR / 'provenance.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

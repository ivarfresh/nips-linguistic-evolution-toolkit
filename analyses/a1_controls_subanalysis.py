"""Diagnostic sub-analysis: real-A1 partner-myth vs three controls + no-A1 baseline.

Computes five sub-metrics on the bootstrap-cooperation cells (informed +
uninformed, game_myth + myth_game) across five conditions:

    no_a1_baseline / real_a1 / c1_shuffled / c2_filler / c3_own

Sub-metrics:
    1. Trajectory shape: round-by-round mean cumulative dyad balance and
       early-vs-late slope (rounds 1-5 vs 6-10).
    2. Recovery from defection events: avg rounds-to-recovery after a per-round
       payoff dropping >1 SD below the run mean.
    3. Cooperation stability: std of per-round trust_ratio (sent / endowment)
       averaged across seeds (lower = more stable).
    4. Linguistic convergence: cosine similarity between Agent_1 and Agent_2
       myths per round (sentence-transformers all-MiniLM-L6-v2). Reports mean
       per-run similarity and round-1 -> round-N slope.
    5. Trust-ratio drift: per-agent (investor vs trustee role) trust_ratio
       averaged across rounds, plus drift = late - early.

Outputs a markdown report at:
    data/plots/v4_direct_provider_controls/diagnostic_subanalysis.md

Embeddings are cached in data/plots/v4_direct_provider_controls/.embed_cache.npz
to keep reruns fast.

Run from repo root:
    python3 analyses/a1_controls_subanalysis.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = "/Users/aron/nips-linguistic-evolution-toolkit"

CONDITIONS: Dict[str, str] = {
    "no_a1_baseline": f"{REPO_ROOT}/data/json/noise_experiments/v4_direct_provider/noise_bootstrap_mem3/gpt-5-nano",
    "real_a1": f"{REPO_ROOT}/data/json/noise_experiments/v4_direct_provider_A1_partner_myth/gpt5nano_partner_myth_injection/gpt-5-nano",
    "c1_shuffled": f"{REPO_ROOT}/data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_shuffled_bootstrap/gpt-5-nano",
    "c2_filler": f"{REPO_ROOT}/data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_filler_bootstrap/gpt-5-nano",
    "c3_own": f"{REPO_ROOT}/data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_own_bootstrap/gpt-5-nano",
}

CONDITION_ORDER = ["no_a1_baseline", "real_a1", "c1_shuffled", "c2_filler", "c3_own"]
CONDITION_LABEL = {
    "no_a1_baseline": "no-A1 baseline",
    "real_a1": "real A1 (partner myth)",
    "c1_shuffled": "C1 shuffled",
    "c2_filler": "C2 filler",
    "c3_own": "C3 own",
}

TASK_ORDERS = ["game_myth", "myth_game"]
CELLS = ["noisy_bootstrap_cooperation", "noisy_bootstrap_cooperation_informed"]

ENDOWMENT = 10.0  # gpt-5-nano bootstrap experiments use endowment=10

OUTPUT_DIR = f"{REPO_ROOT}/data/plots/v4_direct_provider_controls"
MD_PATH = f"{OUTPUT_DIR}/diagnostic_subanalysis.md"
EMBED_CACHE_PATH = f"{OUTPUT_DIR}/.embed_cache.npz"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def list_files(condition: str, task: str, cell: str) -> List[str]:
    root = CONDITIONS[condition]
    pat = f"{root}/{task}/{cell}/*.json"
    files = sorted(glob.glob(pat))
    return [
        f for f in files
        if not any(s in f for s in [".checkpoint.", ".results.", ".error."])
    ]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_myth_wrapping(myth: Optional[str]) -> str:
    """Strip outer JSON wrapping like {"text": "..."} or {"myth": "..."}.

    Returns empty string for None/empty input.
    """
    if not myth:
        return ""
    m = myth.strip()
    if not m:
        return ""
    # Try parsing as JSON if it looks like a dict
    if m.startswith("{") and m.endswith("}"):
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                for key in ("text", "myth", "content", "story"):
                    if key in obj and isinstance(obj[key], str):
                        return obj[key].strip()
        except Exception:
            pass
    return m


# ---------------------------------------------------------------------------
# Per-run metric computation
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    file: str
    n_rounds: int
    cum_dyad_balance: List[float]  # cumulative (Agent_1 + Agent_2) per round
    per_round_total_payoff: List[float]  # investor_payoff + trustee_payoff
    trust_ratio_per_round: List[float]  # sent / ENDOWMENT
    investor_trust_ratio_by_round: Dict[str, List[Optional[float]]]
    # Linguistic convergence (cosine sim per round)
    sim_per_round: List[Optional[float]]
    # Recovery
    recovery_lengths: List[int]  # rounds-to-recovery for each defection event


def compute_run_metrics(data: dict, embed_fn) -> Optional[RunMetrics]:
    history = data.get("conversation_history", [])
    rounds = [r for r in history if r.get("sent") is not None and r.get("returned") is not None]
    if not rounds:
        return None

    sent = np.array([r["sent"] for r in rounds], dtype=float)
    investor_payoff = np.array([r["investor_payoff"] for r in rounds], dtype=float)
    trustee_payoff = np.array([r["trustee_payoff"] for r in rounds], dtype=float)
    per_round_total = (investor_payoff + trustee_payoff).tolist()

    cum_dyad = []
    for r in rounds:
        bal = r.get("balances", {})
        cum_dyad.append(float(bal.get("Agent_1", 0)) + float(bal.get("Agent_2", 0)))

    trust_ratio = (sent / ENDOWMENT).tolist()

    # Per-role investor trust ratio. roles: who is investor that round.
    inv_trust = {"Agent_1": [], "Agent_2": []}
    for r in rounds:
        roles = r.get("roles", {})
        # The round has a single sent value (from whoever the investor was)
        s = float(r["sent"])
        for aid in ("Agent_1", "Agent_2"):
            if roles.get(aid) == "investor":
                inv_trust[aid].append(s / ENDOWMENT)
            else:
                inv_trust[aid].append(None)

    # Linguistic convergence: cosine similarity between agent_1 and agent_2 myths
    sim_per_round: List[Optional[float]] = []
    pairs_to_embed: List[Tuple[int, str, str]] = []
    for i, r in enumerate(rounds):
        myths = r.get("myths") or {}
        m1 = strip_myth_wrapping(myths.get("Agent_1"))
        m2 = strip_myth_wrapping(myths.get("Agent_2"))
        if m1 and m2:
            pairs_to_embed.append((i, m1, m2))
            sim_per_round.append(np.nan)  # placeholder
        else:
            sim_per_round.append(None)

    if pairs_to_embed:
        all_texts = []
        for _, m1, m2 in pairs_to_embed:
            all_texts.append(m1)
            all_texts.append(m2)
        embeddings = embed_fn(all_texts)
        # cosine sim per pair
        for k, (idx, _m1, _m2) in enumerate(pairs_to_embed):
            e1 = embeddings[2 * k]
            e2 = embeddings[2 * k + 1]
            denom = (np.linalg.norm(e1) * np.linalg.norm(e2)) or 1.0
            cos = float(np.dot(e1, e2) / denom)
            sim_per_round[idx] = cos

    # Recovery from defection events
    arr = np.array(per_round_total)
    if len(arr) >= 3:
        run_mean = float(np.mean(arr))
        run_std = float(np.std(arr))
        threshold_low = run_mean - run_std
    else:
        run_mean = float(np.mean(arr)) if len(arr) else 0.0
        threshold_low = -np.inf

    recovery_lengths: List[int] = []
    if len(arr) >= 2:
        for i in range(len(arr) - 1):
            if arr[i] < threshold_low:
                # search next rounds until we hit at-or-above mean
                rec_len = None
                for j in range(i + 1, len(arr)):
                    if arr[j] >= run_mean:
                        rec_len = j - i
                        break
                if rec_len is not None:
                    recovery_lengths.append(rec_len)

    return RunMetrics(
        file=os.path.basename("unused"),
        n_rounds=len(rounds),
        cum_dyad_balance=cum_dyad,
        per_round_total_payoff=per_round_total,
        trust_ratio_per_round=trust_ratio,
        investor_trust_ratio_by_round=inv_trust,
        sim_per_round=sim_per_round,
        recovery_lengths=recovery_lengths,
    )


# ---------------------------------------------------------------------------
# Embedding caching
# ---------------------------------------------------------------------------


_EMBED_MODEL = None
_EMBED_CACHE: Dict[str, np.ndarray] = {}


def _hash_text(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def load_embed_cache():
    global _EMBED_CACHE
    if os.path.exists(EMBED_CACHE_PATH):
        try:
            data = np.load(EMBED_CACHE_PATH, allow_pickle=True)
            keys = list(data["keys"])
            vecs = data["vecs"]
            _EMBED_CACHE = {k: vecs[i] for i, k in enumerate(keys)}
            print(f"  Loaded embedding cache with {len(_EMBED_CACHE)} entries")
        except Exception as e:
            print(f"  Could not load embedding cache: {e}")
            _EMBED_CACHE = {}


def save_embed_cache():
    if not _EMBED_CACHE:
        return
    keys = list(_EMBED_CACHE.keys())
    vecs = np.stack([_EMBED_CACHE[k] for k in keys])
    os.makedirs(os.path.dirname(EMBED_CACHE_PATH), exist_ok=True)
    np.savez(EMBED_CACHE_PATH, keys=np.array(keys), vecs=vecs)
    print(f"  Saved embedding cache with {len(_EMBED_CACHE)} entries")


def _get_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print("  Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return stacked embeddings for `texts`, using cache where possible."""
    if not texts:
        return np.zeros((0, 384))
    missing_idx = [i for i, t in enumerate(texts) if _hash_text(t) not in _EMBED_CACHE]
    if missing_idx:
        model = _get_model()
        missing_texts = [texts[i] for i in missing_idx]
        new_embs = model.encode(missing_texts, batch_size=32, show_progress_bar=False)
        for i, idx in enumerate(missing_idx):
            _EMBED_CACHE[_hash_text(texts[idx])] = np.asarray(new_embs[i], dtype=np.float32)
    return np.stack([_EMBED_CACHE[_hash_text(t)] for t in texts])


# ---------------------------------------------------------------------------
# Aggregation across runs in a cell
# ---------------------------------------------------------------------------


@dataclass
class CellAgg:
    n: int
    # 1. Trajectory
    cum_dyad_round_means: List[float]  # mean across runs, per round (truncated to common length)
    cum_dyad_final: Tuple[float, float]  # (mean, std) of final-round dyad balance
    early_slope: Tuple[float, float]  # (mean, std) of (round5 - round1) cumulative balance per run
    late_slope: Tuple[float, float]  # (mean, std) of (round10 - round6) cumulative balance per run
    # 2. Recovery
    recovery_avg: Tuple[float, float, int]  # (mean, std, n_runs_with_events)
    recovery_event_count_mean: float  # avg events per run
    # 3. Stability
    trust_ratio_std_mean: Tuple[float, float]  # (mean, std) of per-run std(trust_ratio)
    # 4. Linguistic convergence
    sim_mean_per_run: Tuple[float, float, int]  # (mean, std, n_runs)
    sim_round_means: List[float]  # mean similarity per round across runs
    sim_slope: Tuple[float, float]  # late_round_avg - early_round_avg per run
    # 5. Trust ratio drift
    trust_drift_per_run: Tuple[float, float]  # (mean, std) of late_avg - early_avg


def aggregate_cell(metrics_list: List[RunMetrics]) -> Optional[CellAgg]:
    if not metrics_list:
        return None
    # Standardize rounds to the minimum across runs (typically 10)
    n_rounds = min(m.n_rounds for m in metrics_list)

    # ---- Trajectory ----
    cum_matrix = np.array([m.cum_dyad_balance[:n_rounds] for m in metrics_list], dtype=float)
    cum_round_means = cum_matrix.mean(axis=0).tolist()
    finals = cum_matrix[:, -1]
    cum_final = (float(finals.mean()), float(finals.std()))

    # Use rounds 1-5 (idx 0-4) and rounds 6-10 (idx 5-9) when available
    if n_rounds >= 5:
        early_change = cum_matrix[:, 4] - cum_matrix[:, 0]
        early_slope = (float(early_change.mean()), float(early_change.std()))
    else:
        early_slope = (float("nan"), float("nan"))
    if n_rounds >= 10:
        late_change = cum_matrix[:, 9] - cum_matrix[:, 5]
        late_slope = (float(late_change.mean()), float(late_change.std()))
    else:
        # fallback: second half
        mid = n_rounds // 2
        if n_rounds - mid >= 2:
            late_change = cum_matrix[:, -1] - cum_matrix[:, mid]
            late_slope = (float(late_change.mean()), float(late_change.std()))
        else:
            late_slope = (float("nan"), float("nan"))

    # ---- Recovery ----
    all_lens: List[int] = []
    runs_with_events = 0
    event_counts = []
    for m in metrics_list:
        if m.recovery_lengths:
            all_lens.extend(m.recovery_lengths)
            runs_with_events += 1
        event_counts.append(len(m.recovery_lengths))
    if all_lens:
        recovery_avg = (float(np.mean(all_lens)), float(np.std(all_lens)), runs_with_events)
    else:
        recovery_avg = (float("nan"), float("nan"), 0)
    recovery_event_count_mean = float(np.mean(event_counts)) if event_counts else 0.0

    # ---- Stability ----
    per_run_std = [float(np.std(m.trust_ratio_per_round[:n_rounds])) for m in metrics_list]
    trust_ratio_std_mean = (float(np.mean(per_run_std)), float(np.std(per_run_std)))

    # ---- Linguistic convergence ----
    sim_run_means: List[float] = []
    # round-by-round means: average across runs that have a value at that round
    sim_per_round_lists: List[List[float]] = [[] for _ in range(n_rounds)]
    sim_slopes: List[float] = []
    for m in metrics_list:
        sims_with_idx = [
            (i, v) for i, v in enumerate(m.sim_per_round[:n_rounds])
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        if sims_with_idx:
            sim_run_means.append(float(np.mean([v for _, v in sims_with_idx])))
            for i, v in sims_with_idx:
                sim_per_round_lists[i].append(v)
            # slope: late half mean - early half mean
            half = n_rounds // 2
            early = [v for i, v in sims_with_idx if i < half]
            late = [v for i, v in sims_with_idx if i >= half]
            if early and late:
                sim_slopes.append(float(np.mean(late) - np.mean(early)))
    if sim_run_means:
        sim_mean_per_run = (float(np.mean(sim_run_means)), float(np.std(sim_run_means)), len(sim_run_means))
    else:
        sim_mean_per_run = (float("nan"), float("nan"), 0)
    sim_round_means = [
        float(np.mean(vals)) if vals else float("nan")
        for vals in sim_per_round_lists
    ]
    if sim_slopes:
        sim_slope = (float(np.mean(sim_slopes)), float(np.std(sim_slopes)))
    else:
        sim_slope = (float("nan"), float("nan"))

    # ---- Trust ratio drift ----
    drifts: List[float] = []
    for m in metrics_list:
        tr = m.trust_ratio_per_round[:n_rounds]
        half = n_rounds // 2
        if half >= 1 and len(tr) - half >= 1:
            early = float(np.mean(tr[:half]))
            late = float(np.mean(tr[half:]))
            drifts.append(late - early)
    if drifts:
        trust_drift_per_run = (float(np.mean(drifts)), float(np.std(drifts)))
    else:
        trust_drift_per_run = (float("nan"), float("nan"))

    return CellAgg(
        n=len(metrics_list),
        cum_dyad_round_means=cum_round_means,
        cum_dyad_final=cum_final,
        early_slope=early_slope,
        late_slope=late_slope,
        recovery_avg=recovery_avg,
        recovery_event_count_mean=recovery_event_count_mean,
        trust_ratio_std_mean=trust_ratio_std_mean,
        sim_mean_per_run=sim_mean_per_run,
        sim_round_means=sim_round_means,
        sim_slope=sim_slope,
        trust_drift_per_run=trust_drift_per_run,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def collect_all() -> Dict[Tuple[str, str, str], Optional[CellAgg]]:
    """Returns mapping (condition, task, cell) -> CellAgg (or None if no files)."""
    load_embed_cache()
    result: Dict[Tuple[str, str, str], Optional[CellAgg]] = {}
    for cond in CONDITION_ORDER:
        for task in TASK_ORDERS:
            for cell in CELLS:
                files = list_files(cond, task, cell)
                if not files:
                    result[(cond, task, cell)] = None
                    continue
                metrics_list: List[RunMetrics] = []
                for fp in files:
                    try:
                        data = load_json(fp)
                    except Exception as e:
                        print(f"  Failed to load {fp}: {e}")
                        continue
                    rm = compute_run_metrics(data, embed_texts)
                    if rm is not None:
                        metrics_list.append(rm)
                agg = aggregate_cell(metrics_list)
                result[(cond, task, cell)] = agg
                print(
                    f"  {cond:18s} {task:9s} {cell:42s} "
                    f"N={agg.n if agg else 0}"
                )
    save_embed_cache()
    return result


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------


COLUMN_LABELS = [
    ("game_myth", "noisy_bootstrap_cooperation", "game_myth uninformed"),
    ("game_myth", "noisy_bootstrap_cooperation_informed", "game_myth informed"),
    ("myth_game", "noisy_bootstrap_cooperation", "myth_game uninformed"),
    ("myth_game", "noisy_bootstrap_cooperation_informed", "myth_game informed"),
]


def fmt_mean_sd_n(m: float, sd: float, n: int) -> str:
    if n == 0 or (isinstance(m, float) and np.isnan(m)):
        return "—"
    return f"{m:.2f} ± {sd:.2f} (N={n})"


def render_cell_table(agg_lookup, value_fn) -> str:
    """value_fn(agg) -> str (already formatted)."""
    header = ["Condition"] + [lbl for _, _, lbl in COLUMN_LABELS]
    sep = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for cond in CONDITION_ORDER:
        row = [CONDITION_LABEL[cond]]
        for task, cell, _ in COLUMN_LABELS:
            agg = agg_lookup.get((cond, task, cell))
            if agg is None:
                row.append("—")
            else:
                row.append(value_fn(agg))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_round_means_table(agg_lookup, task: str, cell: str) -> str:
    """Per-round mean cumulative dyad balance, one column per condition."""
    # collect per-round means
    rows = {}
    max_rounds = 0
    for cond in CONDITION_ORDER:
        agg = agg_lookup.get((cond, task, cell))
        if agg is None:
            continue
        rows[cond] = agg.cum_dyad_round_means
        max_rounds = max(max_rounds, len(agg.cum_dyad_round_means))
    if not rows:
        return "_(no data)_"
    header = ["Round"] + [CONDITION_LABEL[c] for c in CONDITION_ORDER if c in rows]
    sep = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for r_idx in range(max_rounds):
        row = [f"R{r_idx + 1}"]
        for c in CONDITION_ORDER:
            if c not in rows:
                continue
            vals = rows[c]
            row.append(f"{vals[r_idx]:.1f}" if r_idx < len(vals) else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def gap_vs(strong: Optional[CellAgg], focal: Optional[CellAgg], get_value) -> Optional[Tuple[float, float, float]]:
    """Returns (focal_value, strong_value, gap = focal - strong) or None."""
    if focal is None or strong is None:
        return None
    fv = get_value(focal)
    sv = get_value(strong)
    if fv is None or sv is None:
        return None
    if isinstance(fv, float) and (np.isnan(fv) or np.isnan(sv)):
        return None
    return float(fv), float(sv), float(fv - sv)


def write_markdown(agg_lookup):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parts: List[str] = []
    parts.append("# A1 partner-myth vs controls: diagnostic sub-analysis\n")
    parts.append(
        "Comparison of the real-A1 partner-myth injection against three defensive "
        "controls (C1 shuffled, C2 filler, C3 own-myth) and the no-A1 baseline. "
        "All cells are bootstrap-cooperation; informed and uninformed framing both "
        "shown when data exist. Real-A1 has no informed cells (—) and only N=13 in "
        "myth_game uninformed.\n"
    )
    parts.append(
        "Endowment is fixed at 10 for these gpt-5-nano runs. Embeddings: "
        "`sentence-transformers/all-MiniLM-L6-v2`. Defection events for the "
        "recovery metric are per-round total payoff (investor + trustee) below "
        "(run mean − 1 run SD); recovery is the next round meeting or exceeding the "
        "run mean. All slopes/drifts use rounds 1–5 vs 6–10 when 10 rounds are present.\n"
    )

    # --- Sub-metric 1: trajectory shape ---
    parts.append("## 1. Trajectory shape\n")
    parts.append("### 1a. Final cumulative dyad balance (mean ± SD)\n")
    parts.append(
        render_cell_table(
            agg_lookup,
            lambda a: fmt_mean_sd_n(a.cum_dyad_final[0], a.cum_dyad_final[1], a.n),
        )
    )
    parts.append("\n\n### 1b. Early slope (round 5 − round 1 cumulative balance)\n")
    parts.append(
        render_cell_table(
            agg_lookup,
            lambda a: fmt_mean_sd_n(a.early_slope[0], a.early_slope[1], a.n),
        )
    )
    parts.append("\n\n### 1c. Late slope (round 10 − round 6 cumulative balance)\n")
    parts.append(
        render_cell_table(
            agg_lookup,
            lambda a: fmt_mean_sd_n(a.late_slope[0], a.late_slope[1], a.n),
        )
    )
    parts.append("\n\n### 1d. Per-round mean cumulative dyad balance, game_myth uninformed\n")
    parts.append(render_round_means_table(agg_lookup, "game_myth", "noisy_bootstrap_cooperation"))
    parts.append("\n\n### 1e. Per-round mean cumulative dyad balance, myth_game uninformed\n")
    parts.append(render_round_means_table(agg_lookup, "myth_game", "noisy_bootstrap_cooperation"))

    # --- Sub-metric 2: recovery ---
    parts.append("\n\n## 2. Recovery from defection events\n")
    parts.append(
        "Average rounds-to-recovery after a per-round payoff drop > 1 SD below "
        "the run mean (cell value = mean ± SD over events; n_runs_with_events shown). "
        "Lower is faster recovery.\n"
    )
    parts.append(
        render_cell_table(
            agg_lookup,
            lambda a: (
                f"{a.recovery_avg[0]:.2f} ± {a.recovery_avg[1]:.2f} "
                f"(events={int(a.recovery_event_count_mean * a.n)}, "
                f"runs_w_events={a.recovery_avg[2]}/{a.n})"
                if a.recovery_avg[2] > 0 else "— (no defection events)"
            ),
        )
    )

    # --- Sub-metric 3: stability ---
    parts.append("\n\n## 3. Cooperation stability (std of trust_ratio across rounds)\n")
    parts.append("Per-run std of `sent / endowment` across the 10 rounds, averaged across seeds. Lower = more stable.\n")
    parts.append(
        render_cell_table(
            agg_lookup,
            lambda a: fmt_mean_sd_n(a.trust_ratio_std_mean[0], a.trust_ratio_std_mean[1], a.n),
        )
    )

    # --- Sub-metric 4: linguistic convergence ---
    parts.append("\n\n## 4. Linguistic convergence (cosine sim between agent_1 and agent_2 myth per round)\n")
    parts.append("### 4a. Mean per-round cosine similarity (averaged over rounds within each run, then across runs)\n")
    parts.append(
        render_cell_table(
            agg_lookup,
            lambda a: fmt_mean_sd_n(a.sim_mean_per_run[0], a.sim_mean_per_run[1], a.sim_mean_per_run[2]),
        )
    )
    parts.append("\n\n### 4b. Convergence slope (late-half − early-half mean cosine sim, per run)\n")
    parts.append("Positive = myths get more similar over time.\n")
    parts.append(
        render_cell_table(
            agg_lookup,
            lambda a: fmt_mean_sd_n(a.sim_slope[0], a.sim_slope[1], a.n),
        )
    )

    # --- Sub-metric 5: trust ratio drift ---
    parts.append("\n\n## 5. Trust-ratio drift (late-half − early-half mean trust_ratio per run)\n")
    parts.append("Positive = trust grew over the game; negative = trust eroded.\n")
    parts.append(
        render_cell_table(
            agg_lookup,
            lambda a: fmt_mean_sd_n(a.trust_drift_per_run[0], a.trust_drift_per_run[1], a.n),
        )
    )

    # --- Verdict ---
    parts.append("\n\n## Verdict\n")
    parts.append(
        "For each metric we compare real-A1 against the strongest control (C2 filler, "
        "which beat real-A1 on the headline dyad-balance metric). Cell used: "
        "**game_myth uninformed** (the only cell where real-A1 has the full N=15 sample). "
        "myth_game uninformed gap is reported in parentheses where available.\n"
    )

    real_gm_un = agg_lookup.get(("real_a1", "game_myth", "noisy_bootstrap_cooperation"))
    filler_gm_un = agg_lookup.get(("c2_filler", "game_myth", "noisy_bootstrap_cooperation"))
    real_mg_un = agg_lookup.get(("real_a1", "myth_game", "noisy_bootstrap_cooperation"))
    filler_mg_un = agg_lookup.get(("c2_filler", "myth_game", "noisy_bootstrap_cooperation"))

    def gap_line(name: str, getter, lower_better: bool, units: str = "") -> str:
        gm = gap_vs(filler_gm_un, real_gm_un, getter)
        mg = gap_vs(filler_mg_un, real_mg_un, getter)
        chunks = []
        if gm is not None:
            fv, sv, g = gm
            direction = "lower" if g < 0 else "higher"
            sign = "-" if g < 0 else "+"
            chunks.append(
                f"game_myth: real-A1 {fv:.3f}{units}, C2 filler {sv:.3f}{units}, "
                f"Δ = {sign}{abs(g):.3f}{units} ({'better' if (g < 0) == lower_better else 'worse'} for real-A1 if {'lower' if lower_better else 'higher'} is desired)"
            )
        if mg is not None:
            fv, sv, g = mg
            sign = "-" if g < 0 else "+"
            chunks.append(
                f"myth_game: Δ = {sign}{abs(g):.3f}{units}"
            )
        return f"- **{name}**: " + "; ".join(chunks) if chunks else f"- **{name}**: insufficient data"

    parts.append(gap_line("Final dyad balance", lambda a: a.cum_dyad_final[0], lower_better=False))
    parts.append(gap_line("Early slope (R1→R5 cum)", lambda a: a.early_slope[0], lower_better=False))
    parts.append(gap_line("Late slope (R6→R10 cum)", lambda a: a.late_slope[0], lower_better=False))
    parts.append(
        gap_line(
            "Recovery rounds-to-recovery",
            lambda a: a.recovery_avg[0],
            lower_better=True,
        )
    )
    parts.append(
        gap_line(
            "Trust-ratio std (lower = stabler)",
            lambda a: a.trust_ratio_std_mean[0],
            lower_better=True,
        )
    )
    parts.append(
        gap_line(
            "Linguistic mean cosine sim (higher = more convergent)",
            lambda a: a.sim_mean_per_run[0],
            lower_better=False,
        )
    )
    parts.append(
        gap_line(
            "Linguistic convergence slope (higher = late > early)",
            lambda a: a.sim_slope[0],
            lower_better=False,
        )
    )
    parts.append(
        gap_line(
            "Trust drift (positive = trust grew)",
            lambda a: a.trust_drift_per_run[0],
            lower_better=False,
        )
    )

    parts.append(
        "\n### Concluding paragraph\n"
        "Read the per-row Δ values above. The intended interpretation: if real-A1 "
        "is within ~1 SD of C2 filler on every economic metric (final balance, "
        "slopes, recovery, stability, trust drift) but clearly higher on the "
        "linguistic-convergence rows (mean cosine sim and convergence slope), "
        "then the partner-myth condition's only measurable signature is "
        "linguistic, not economic. That is the narrow claim §4.6 can still "
        "defend. If the linguistic rows are also within ~1 SD of C2 filler, "
        "§4.6 must be rewritten as 'any prompt augmentation lifts cooperation'. "
        "Hand-edit this paragraph after a rerun to reflect the actual numerical "
        "verdict for the new data."
    )

    md = "\n".join(parts) + "\n"
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nWrote markdown to {MD_PATH}")


def main():
    print("Collecting per-cell metrics...")
    agg_lookup = collect_all()
    print("Writing markdown...")
    write_markdown(agg_lookup)
    print("Done.")


if __name__ == "__main__":
    main()

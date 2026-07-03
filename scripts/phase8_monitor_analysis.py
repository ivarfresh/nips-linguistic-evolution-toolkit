"""Phase 8 — silent-monitor analysis.

Compares the monitored 8-agent Sonnet runs (each round's strategy-laden myths
have that round's game earnings zeroed) against the unmonitored control
(sonnet45_8agent_myth_directive_history3_anon_r10_n5).

Produces:
  1. Per-round monitor detection rate (fraction of myths flagged), monitored runs.
  2. Per-round joint balance, monitored vs control (mean ±sd), with $600 ceiling.
  3. Round-10 legibility: the frozen phase-7 extraction probe over each pool's
     round-10 myths (monitored, control) and the s_end_plus seed originals —
     has_strategy %. Answers "did strategy go covert".
  4. Retrospective control detection: the frozen monitor run over the control's
     round-10 myths (no selection pressure) vs monitored round-10 detection —
     separates outcome (b) from an intrinsic-drift confound.

Detection for monitored runs is read from the saved monitor records (free).
The extraction probe and the retrospective control monitor make paid Sonnet
calls; a preflight line is printed before they run.

Usage (from repo root):
    python scripts/phase8_monitor_analysis.py [--workers 6] [--no-probe]

Outputs: data/phase8/monitor_analysis_results.json
         data/phase8/plots/{01_detection_and_balance,02_round10_legibility}.png
"""

import argparse
import glob
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analyses._shared import configure_matplotlib  # noqa: E402
from src.monitor import StrategyMonitor  # noqa: E402
from src.utils import call_llm, create_llm_client  # noqa: E402
from scripts.phase7_decoder_asymmetry import (  # noqa: E402
    EXTRACTION_PROMPT,
    load_system_prompt,
    parse_response,
)

MONITORED_GLOB = (
    "data/json/phase8_monitored/claude-sonnet-4.5/myth_game/*.json"
)
CONTROL_GLOB = (
    "data/json/sonnet45_8agent_myth_directive_history3_anon_r10_n5/"
    "claude-sonnet-4.5/myth_game/*.json"
)
MANIFEST = REPO_ROOT / "data/phase3/seed_manifest.json"
OUT_JSON = REPO_ROOT / "data/phase8/monitor_analysis_results.json"
PLOT_DIR = REPO_ROOT / "data/phase8/plots"
CEILING = 600.0
NUM_ROUNDS = 10
DECODER = "anthropic/claude-sonnet-4.5"
MONITOR_MODEL = "anthropic/claude-sonnet-4.5"


def _real_runs(pattern):
    files = sorted(glob.glob(str(REPO_ROOT / pattern)))
    return [
        f for f in files
        if ".results." not in f and ".checkpoint." not in f and ".error." not in f
    ]


def load_runs(pattern):
    runs = []
    for f in _real_runs(pattern):
        with open(f) as fh:
            runs.append((f, json.load(fh)))
    return runs


def joint_balance(entry):
    bal = entry.get("balances") or {}
    return sum(bal.values()) if bal else None


def detection_fraction(entry):
    mon = entry.get("monitor") or {}
    if not mon:
        return None
    return sum(1 for r in mon.values() if r.get("flagged")) / len(mon)


def round10_myths(runs):
    """Return list of (run_label, agent_id, myth_text) for the final round."""
    out = []
    for path, d in runs:
        ch = d.get("conversation_history", [])
        if not ch:
            continue
        last = ch[-1]
        for agent_id, myth in (last.get("myths") or {}).items():
            out.append((Path(path).name, agent_id, myth))
    return out


def cumulative_prepenalty(runs):
    """Per-round mean/sd of the cumulative joint balance the agents WOULD have if
    nothing were confiscated — i.e. the running sum of raw round payoffs. This
    isolates whether in-game cooperation itself persisted vs collapsed, separate
    from the monitor's confiscation of it."""
    means, sds, ns = [], [], []
    for rnd in range(1, NUM_ROUNDS + 1):
        vals = []
        for _, d in runs:
            ch = d.get("conversation_history", [])
            total = 0.0
            seen = False
            for e in ch:
                if e.get("round", 0) <= rnd and e.get("payoffs"):
                    total += sum(e["payoffs"].values())
                    if e.get("round") == rnd:
                        seen = True
            if seen:
                vals.append(total)
        means.append(statistics.mean(vals) if vals else None)
        sds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        ns.append(len(vals))
    return means, sds, ns


def per_round_stats(runs, fn):
    """Mean/sd across runs per round of fn(entry)."""
    means, sds, ns = [], [], []
    for rnd in range(1, NUM_ROUNDS + 1):
        vals = []
        for _, d in runs:
            ch = d.get("conversation_history", [])
            entry = next((e for e in ch if e.get("round") == rnd), None)
            if entry is None:
                continue
            v = fn(entry)
            if v is not None:
                vals.append(v)
        means.append(statistics.mean(vals) if vals else None)
        sds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        ns.append(len(vals))
    return means, sds, ns


# ---- paid probes -----------------------------------------------------------


def extract_one(job, system_prompt):
    label, agent_id, myth = job
    client = create_llm_client(DECODER)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": EXTRACTION_PROMPT.format(myth=myth, endowment=5)},
    ]
    resp = call_llm(client, DECODER, 0.0, messages)
    parsed = parse_response((resp or {}).get("content") or "")
    return {"label": label, "agent": agent_id, "parsed": parsed}


def run_extraction(pool_jobs, workers):
    """pool_jobs: {pool_name: [(label, agent, myth), ...]}. Returns per-pool stats."""
    system_prompt = load_system_prompt()
    jobs = [(pool, job) for pool, jobs in pool_jobs.items() for job in jobs]
    results = {pool: [] for pool in pool_jobs}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(extract_one, job, system_prompt): pool for pool, job in jobs}
        for fut in as_completed(futures):
            pool = futures[fut]
            try:
                results[pool].append(fut.result())
            except Exception as exc:
                print(f"  extract failed ({pool}): {exc}")
    stats = {}
    for pool, rows in results.items():
        parsed = [r["parsed"] for r in rows if r["parsed"] is not None]
        n = len(parsed)
        strat = sum(1 for p in parsed if p["has_strategy"]) / n if n else None
        sends = [p["send"] for p in parsed if p["has_strategy"] and isinstance(p["send"], (int, float))]
        rfs = [p["return_fraction"] for p in parsed
               if p["has_strategy"] and isinstance(p["return_fraction"], (int, float))]
        stats[pool] = {
            "n": n,
            "has_strategy_pct": strat,
            "mean_send": statistics.mean(sends) if sends else None,
            "sd_send": statistics.stdev(sends) if len(sends) > 1 else 0.0,
            "mean_return_fraction": statistics.mean(rfs) if rfs else None,
        }
    return stats


def run_control_monitor(control_r10, workers):
    """Frozen monitor over the control's round-10 myths (no selection pressure)."""
    rules = load_system_prompt()
    monitor = StrategyMonitor(rules, model=MONITOR_MODEL, temperature=0.0)
    flagged = 0
    total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(monitor.classify, myth) for _, _, myth in control_r10]
        for fut in as_completed(futures):
            v = fut.result()
            total += 1
            if v.get("flagged"):
                flagged += 1
    return {"n": total, "flagged": flagged, "detection_rate": flagged / total if total else None}


# ---- plotting --------------------------------------------------------------


def plot_detection_and_balance(det_m, bal_m, bal_c, bal_m_pre, control_r10_det):
    configure_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(1, NUM_ROUNDS + 1)

    dm, dsd, _ = det_m
    ax1.plot(x, dm, "-o", color="#d62728", label="Monitored runs (during selection)")
    ax1.fill_between(x, [m - s for m, s in zip(dm, dsd)], [m + s for m, s in zip(dm, dsd)],
                     color="#d62728", alpha=0.15)
    if control_r10_det and control_r10_det.get("detection_rate") is not None:
        ax1.scatter([NUM_ROUNDS], [control_r10_det["detection_rate"]], s=90, marker="D",
                    color="#7f7f7f", zorder=5,
                    label=f"Control round 10 (no pressure, n={control_r10_det['n']})")
    for i, (m, s) in enumerate(zip(dm, dsd)):
        if m is not None:
            ax1.text(x[i], m + 0.03, f"{m:.2f}", ha="center", fontsize=8)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Fraction of myths flagged as game strategy")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_title("Monitor detection rate per round")

    bm, bmsd, _ = bal_m
    bc, bcsd, _ = bal_c
    bmp, bmpsd, _ = bal_m_pre
    ax2.errorbar(x, bc, yerr=bcsd, fmt="-s", color="#2ca02c", capsize=3,
                 label="Control / unmonitored (n=5)")
    ax2.errorbar(x, bmp, yerr=bmpsd, fmt="-^", color="#1f77b4", capsize=3,
                 label="Monitored, PRE-penalty (in-game earnings)")
    ax2.errorbar(x, bm, yerr=bmsd, fmt="-o", color="#d62728", capsize=3,
                 label="Monitored, POST-penalty (what agents see)")
    ax2.axhline(CEILING, color="red", linestyle="--", alpha=0.5,
                label=f"Cooperation ceiling (${int(CEILING)})")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Joint balance (sum across 8 agents)")
    ax2.set_xticks(x)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_title("Joint balance per round")

    fig.suptitle(
        "Phase 8 — silent monitor: detection rate and joint balance across rounds\n"
        "Sonnet 4.5 · 8 agents · myth→game · flagged myths lose that round's earnings · agents told nothing",
        fontsize=12,
    )
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = PLOT_DIR / "01_detection_and_balance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_round10_legibility(extraction_stats):
    configure_matplotlib()
    order = [
        ("s_end_plus\noriginals", "#2ca02c", "s_end_plus"),
        ("Control R10\n(unmonitored)", "#7f7f7f", "control_r10"),
        ("Monitored R10\n(under pressure)", "#d62728", "monitored_r10"),
    ]
    order = [(lbl, c, k) for lbl, c, k in order if k in extraction_stats]
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(order))
    for i, (lbl, color, key) in enumerate(order):
        s = extraction_stats[key]
        pct = (s["has_strategy_pct"] or 0) * 100
        ax.bar(x[i], pct, width=0.6, color=color, edgecolor="black", linewidth=0.7, alpha=0.88)
        send_txt = f"\nsend≈{s['mean_send']:.1f}" if s.get("mean_send") is not None else ""
        ax.text(x[i], pct + 1.5, f"{pct:.0f}%\nn={s['n']}{send_txt}", ha="center", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([o[0] for o in order])
    ax.set_ylabel("Round-10 myths decoded as containing game strategy (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        "Phase 8 — round-10 myth legibility (frozen phase-7 extraction probe, Sonnet decoder)\n"
        "Does monitoring pressure make the end-state strategy covert vs the s_end_plus originals?",
        fontsize=11,
    )
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out = PLOT_DIR / "02_round10_legibility.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-probe", action="store_true",
                        help="skip the paid extraction + control-monitor probes")
    args = parser.parse_args()

    monitored = load_runs(MONITORED_GLOB)
    control = load_runs(CONTROL_GLOB)
    print(f"Loaded {len(monitored)} monitored runs, {len(control)} control runs")
    if not monitored:
        print("No monitored runs found — has the batch finished?")
        return

    det_m = per_round_stats(monitored, detection_fraction)
    bal_m = per_round_stats(monitored, joint_balance)
    bal_c = per_round_stats(control, joint_balance)
    bal_m_pre = cumulative_prepenalty(monitored)

    print("\n=== per-round detection rate (monitored) ===")
    for r, (m, s) in enumerate(zip(det_m[0], det_m[1]), 1):
        print(f"  round {r:2d}: {m:.2f} (±{s:.2f})" if m is not None else f"  round {r:2d}: --")
    print("\n=== per-round joint balance ===")
    for r in range(NUM_ROUNDS):
        mm, ms = bal_m[0][r], bal_m[1][r]
        cm, cs = bal_c[0][r], bal_c[1][r]
        pm, ps = bal_m_pre[0][r], bal_m_pre[1][r]
        print(f"  round {r+1:2d}: control ${cm:.0f} (±{cs:.0f})   "
              f"monitored-PRE ${pm:.0f} (±{ps:.0f})   monitored-POST ${mm:.0f} (±{ms:.0f})")

    extraction_stats, control_r10_det = {}, None
    monitored_r10 = round10_myths(monitored)
    control_r10 = round10_myths(control)
    if not args.no_probe:
        manifest = json.loads(MANIFEST.read_text())
        seed_jobs = [("seed", f"s{i}", s["text"])
                     for i, s in enumerate(manifest["seeds"]["s_end_plus"])]
        pool_jobs = {
            "s_end_plus": seed_jobs,
            "control_r10": control_r10,
            "monitored_r10": monitored_r10,
        }
        n_extract = sum(len(v) for v in pool_jobs.values())
        n_monitor = len(control_r10)
        print(f"\nPREFLIGHT: MODEL={DECODER} EXTRACT_N={n_extract} MONITOR_N={n_monitor} "
              f"WORKERS={args.workers} EST_COST=${(n_extract + n_monitor) * 0.01:.2f}")
        extraction_stats = run_extraction(pool_jobs, args.workers)
        control_r10_det = run_control_monitor(control_r10, args.workers)
        print("\n=== round-10 legibility (has_strategy %) ===")
        for pool, s in extraction_stats.items():
            pct = (s["has_strategy_pct"] or 0) * 100
            print(f"  {pool:16s}: {pct:.0f}%  n={s['n']}  send={s['mean_send']}")
        print(f"\n=== retrospective control R10 detection: "
              f"{control_r10_det['detection_rate']:.2f} "
              f"({control_r10_det['flagged']}/{control_r10_det['n']}) ===")

    monitored_r10_det = det_m[0][NUM_ROUNDS - 1]
    results = {
        "n_monitored": len(monitored),
        "n_control": len(control),
        "detection_rate_per_round_monitored": {"mean": det_m[0], "sd": det_m[1], "n": det_m[2]},
        "joint_balance_per_round": {
            "monitored_post_penalty": {"mean": bal_m[0], "sd": bal_m[1]},
            "monitored_pre_penalty": {"mean": bal_m_pre[0], "sd": bal_m_pre[1]},
            "control": {"mean": bal_c[0], "sd": bal_c[1]},
        },
        "round1_detection_monitored": det_m[0][0],
        "round10_detection_monitored": monitored_r10_det,
        "extraction_stats": extraction_stats,
        "control_round10_retrospective_detection": control_r10_det,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_JSON}")

    plot_detection_and_balance(det_m, bal_m, bal_c, bal_m_pre, control_r10_det)
    if extraction_stats:
        plot_round10_legibility(extraction_stats)


if __name__ == "__main__":
    main()

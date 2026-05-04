"""Summary table for A1 partner-myth defensive controls (C1/C2/C3).

Loads result files from the no-A1 baseline, the real-A1 condition, and the
three new controls, then prints a wide table comparing per-condition
cumulative dyad balance (mean, std, N) and Δmean against the no-A1 baseline.

Run from the repo root:
    python3 analyses/a1_controls_summary.py
"""

from __future__ import annotations

import glob
import os
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from analyses._shared import extract_game_metrics, load_simulation_data


# ----------------------------------------------------------------------------
# Condition map: (label) -> (root directory of result files)
# ----------------------------------------------------------------------------

CONDITION_DIRS: Dict[str, str] = {
    "no_A1_baseline": (
        "data/json/noise_experiments/v4_direct_provider/noise_bootstrap_mem3"
    ),
    "real_A1": (
        "data/json/noise_experiments/v4_direct_provider_A1_partner_myth"
        "/gpt5nano_partner_myth_injection"
    ),
    "C1_shuffled": (
        "data/json/noise_experiments/v4_direct_provider_controls"
        "/gpt5nano_partner_myth_shuffled_bootstrap"
    ),
    "C2_filler": (
        "data/json/noise_experiments/v4_direct_provider_controls"
        "/gpt5nano_partner_myth_filler_bootstrap"
    ),
    "C3_own": (
        "data/json/noise_experiments/v4_direct_provider_controls"
        "/gpt5nano_partner_myth_own_bootstrap"
    ),
}

TASK_ORDERS = ("game_myth", "myth_game")
NOISE_VARIANTS = (
    ("noisy_bootstrap_cooperation", "uninformed"),
    ("noisy_bootstrap_cooperation_informed", "informed"),
)
GAME_ONLY_TASK = "game"  # used to compute the per-cell baseline for Δmean


@dataclass
class CellStats:
    n: int
    mean: float
    std: float
    median: float

    def fmt(self) -> str:
        if self.n == 0:
            return "    n/a"
        return f"{self.mean:6.1f} ± {self.std:4.1f} (n={self.n:>2})"


def find_result_files(root: str, task_order: str, noise_variant: str) -> List[str]:
    """Return all main JSON result files matching the task_order and noise."""
    if not os.path.isdir(root):
        return []
    paths: List[str] = []
    pattern = os.path.join(
        root, "**", "gpt-5-nano", task_order, noise_variant, "*.json"
    )
    for path in glob.glob(pattern, recursive=True):
        name = os.path.basename(path)
        if any(skip in name for skip in (".checkpoint.", ".results.", ".error.")):
            continue
        paths.append(path)
    return sorted(paths)


def cumulative_dyad_balance(filepath: str) -> Optional[float]:
    """Final-round Agent_1 + Agent_2 cumulative balance."""
    try:
        data = load_simulation_data(filepath)
    except Exception:
        return None
    metrics = extract_game_metrics(data)
    if metrics is None:
        return None
    a1 = metrics["agent_1_balances"]
    a2 = metrics["agent_2_balances"]
    if not a1 or not a2:
        return None
    return float(a1[-1] + a2[-1])


def cell_stats(filepaths: List[str]) -> CellStats:
    values = [v for v in (cumulative_dyad_balance(p) for p in filepaths) if v is not None]
    if not values:
        return CellStats(n=0, mean=float("nan"), std=float("nan"), median=float("nan"))
    return CellStats(
        n=len(values),
        mean=statistics.mean(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        median=statistics.median(values),
    )


def collect_all() -> Dict[Tuple[str, str, str], CellStats]:
    """Returns: (condition, task_order, noise_label) -> CellStats."""
    out: Dict[Tuple[str, str, str], CellStats] = {}
    for cond_label, root in CONDITION_DIRS.items():
        for task_order in TASK_ORDERS:
            for noise_dir, noise_label in NOISE_VARIANTS:
                files = find_result_files(root, task_order, noise_dir)
                out[(cond_label, task_order, noise_label)] = cell_stats(files)
    # Game-only baseline cells (used for Δmean reference, no_A1 only).
    for noise_dir, noise_label in NOISE_VARIANTS:
        files = find_result_files(
            CONDITION_DIRS["no_A1_baseline"], GAME_ONLY_TASK, noise_dir
        )
        out[("no_A1_game_only", GAME_ONLY_TASK, noise_label)] = cell_stats(files)
    return out


def render_table(stats: Dict[Tuple[str, str, str], CellStats]) -> str:
    lines: List[str] = []
    lines.append("\nCumulative dyad balance (Agent_1 + Agent_2 final), GPT-5-Nano, bootstrap-cooperation noise.")
    lines.append("Δmean is computed against the matched no-A1 baseline cell (same task_order, same noise variant).")
    lines.append("")
    header_widths = (16, 11, 11, 22, 9)
    columns = ("condition", "task_order", "informed", "dyad balance", "Δ vs no-A1")
    sep = "  "
    lines.append(sep.join(c.ljust(w) for c, w in zip(columns, header_widths)))
    lines.append(sep.join("-" * w for w in header_widths))

    # Row order: baseline first, then real_A1, then controls.
    cond_order = ("no_A1_baseline", "real_A1", "C3_own", "C1_shuffled", "C2_filler")
    for noise_label in ("uninformed", "informed"):
        for task_order in TASK_ORDERS:
            baseline_cell = stats[("no_A1_baseline", task_order, noise_label)]
            for cond in cond_order:
                cell = stats[(cond, task_order, noise_label)]
                if cond == "no_A1_baseline":
                    delta_str = "      —"
                elif cell.n == 0 or baseline_cell.n == 0:
                    delta_str = "      —"
                else:
                    delta = cell.mean - baseline_cell.mean
                    delta_str = f"{delta:+6.1f}"
                row = (
                    cond.ljust(header_widths[0]),
                    task_order.ljust(header_widths[1]),
                    noise_label.ljust(header_widths[2]),
                    cell.fmt().ljust(header_widths[3]),
                    delta_str.ljust(header_widths[4]),
                )
                lines.append(sep.join(row))
            lines.append("")

    # Game-only baseline reference (one number per noise variant).
    lines.append("Reference: no-A1, game-only task (no myth task at all):")
    for noise_label in ("uninformed", "informed"):
        cell = stats[("no_A1_game_only", GAME_ONLY_TASK, noise_label)]
        lines.append(f"  bootstrap_{noise_label}: {cell.fmt()}")
    return "\n".join(lines)


def main() -> int:
    stats = collect_all()
    print(render_table(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

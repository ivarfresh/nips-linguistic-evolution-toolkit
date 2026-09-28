#!/usr/bin/env python3
"""Write provenance.json for the linguistic figure folders.

Default folder: docs/figures/linguistic_analysis_20260923/. The myth-rules
folder (docs/figures/myth_rules_20260928/) also reads the game-only runs and
the slide-678 transplant reruns:
  python3 analyses/linguistic_provenance.py --output docs/figures/myth_rules_20260928 --with-game-only --with-transplant


scripts/check_safeguards.py requires every docs/figures/ folder to carry a
provenance.json whose outputs map equals the folder's tracked files (including
the validation/ subfolder). The linguistic analysis reads only the
myth-bearing runs of the validated mixed-model tables (game-only runs have no
myths), so this records exactly those 156 run finals, with the allowed
differences and pools that analyses/_mixed_model_provenance.py uses for the
full tables.

Run last, after every analysis that writes into the folder:
  python3 analyses/linguistic_provenance.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment_condition import output_provenance  # noqa: E402
from scripts import analyze_mixed_model_dyads as dyads  # noqa: E402
from scripts import analyze_mixed_model_populations as populations  # noqa: E402
from analyses._mixed_model_provenance import POOL_REASON  # noqa: E402
from analyses.linguistic_corpus import run_list  # noqa: E402

OUTPUT = ROOT / "docs/figures/linguistic_analysis_20260923"
TRANSPLANT_ROOTS = ["data/json/noise_experiments/slide678_rerun_20260916",
                    "data/json/noise_experiments/slide678_dyad_rerun_20260917"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--with-game-only", action="store_true", help="also record the game-only runs")
    ap.add_argument("--with-transplant", action="store_true", help="also record the slide-678 rerun finals")
    args = ap.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    used = {spec["path"] for size in (2, 8) for spec in run_list(size, ROOT)}
    dyad_mixed, dyad_september = dyads.final_paths()
    population_mixed, population_september = populations.final_paths()

    def keep(paths):
        if args.with_game_only:
            return list(paths)
        return [p for p in paths if str(p.relative_to(ROOT)) in used]

    mixed = keep(dyad_mixed + population_mixed)
    september = keep(dyad_september + population_september)
    listed = {str(p.relative_to(ROOT)) for p in mixed + september}
    if not args.with_game_only and listed != used:
        raise SystemExit(f"Myth runs and run finals disagree: {len(listed ^ used)} paths differ")
    # record where the finals really live (a worktree may reach them through a symlink)
    mixed, september = [p.resolve() for p in mixed], [p.resolve() for p in september]
    pools = {"mixed": mixed, "september": september}
    reason = POOL_REASON
    if args.with_transplant:
        pools["transplant"] = sorted(p.resolve() for r in TRANSPLANT_ROOTS for p in (ROOT / r).glob("*/rep??.json"))
        reason += (" The transplant pool is the slide-678 rerun (Sonnet hosts, injected donor texts, "
                   "historical game-only apparatus), read only to compare donor rules with host sends.")
    allowed = {**dyads.ALLOWED, **populations.ALLOWED}
    if args.with_transplant:
        why = "Transplant rerun: the injected donor text is the manipulation, one donor per replicate"
        allowed.update({k: why for k in ("protocol.simulation.seed_myth", "protocol.simulation.seed_reinject",
                                         "protocol.simulation.seed_user_prompt", "replicate.identity")})
        apparatus = ("Transplant pool uses the historical slide-678 apparatus (uninformed U(-5,0) noise, "
                     "myth-only memory); it is never pooled with September runs, only compared within itself. "
                     "Declared for the folder as a whole; the September pools do not differ in these fields")
        allowed.update({k: apparatus for k in ("protocol.game.defector_prompt_template",
                                               "protocol.game.noise_config.inform_agents",
                                               "protocol.game.noise_config.range",
                                               "protocol.simulation.chat_memory_mode")})
    outputs = sorted(p for p in output.rglob("*")
                     if p.is_file() and p.name != "provenance.json" and not p.name.startswith("."))
    runs = [p for paths in pools.values() for p in paths]
    document = output_provenance(runs, outputs, allowed, output_root=output, pools=pools, pool_reason=reason)
    (output / "provenance.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"provenance: {len(runs)} runs, {len(outputs)} outputs -> {output / 'provenance.json'}")


if __name__ == "__main__":
    main()

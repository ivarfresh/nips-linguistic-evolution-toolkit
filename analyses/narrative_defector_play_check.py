"""What do cooperators actually send to scripted defectors? (no API calls)

Companion to docs/research/narrative_evolution_20261003/broad_audit_300/READING_FINDINGS_20261004.md.
Frontier myths keep prescribing a small token for a proven taker ("send less,
never nothing"). This checks that against play in the frontier defector pilot:
for every standard investor facing a scripted defector from round 3 on (after
the defector's zeros are visible in its three-game history), it reports the
decided send, split by investor model and task order, per decision and per run.

    python analyses/narrative_defector_play_check.py
"""
import collections
import glob
import json
import statistics

ROOT = "data/json/noise_experiments/frontier_defector_pilot_20261001"


def task_order(path):
    for order in ("game_myth", "myth_game"):
        if f"/{order}/" in path:
            return order
    return "game"


def final_runs():
    paths = glob.glob(f"{ROOT}/frontier*/**/*.json", recursive=True)
    return sorted(p for p in paths
                  if not p.endswith(".results.json") and "checkpoint" not in p)


def main():
    decisions = collections.defaultdict(list)
    per_run = collections.defaultdict(list)
    runs = final_runs()
    for path in runs:
        state = json.load(open(path))
        model = {a: v["model"].split("/")[-1] for a, v in state["agents"].items()}
        order = task_order(path)
        run_sends = collections.defaultdict(list)
        for turn in state["conversation_history"]:
            if turn["round"] < 3:
                continue
            types = turn["agent_types"]
            for dyad in turn["dyads"]:
                inv, tru = dyad["investor"], dyad["trustee"]
                if types[inv] != "standard":
                    continue
                partner = "defector" if types[tru] == "defector" else "cooperator"
                decisions[(model[inv], order, partner)].append(dyad["sent_decision"])
                if partner == "defector":
                    run_sends[model[inv]].append(dyad["sent_decision"])
        for m, sends in run_sends.items():
            per_run[(m, order)].append(sum(s == 0 for s in sends) / len(sends))

    print(f"{len(runs)} final runs under {ROOT}\n")
    print(f"{'model':16} {'order':10} {'partner':10} {'n':>4} {'mean send':>14} "
          f"{'send=0':>7} {'0<send<=1.5':>12} {'send>=4':>8}")
    for key in sorted(decisions):
        v = decisions[key]
        print(f"{key[0]:16} {key[1]:10} {key[2]:10} {len(v):4d} "
              f"{statistics.mean(v):6.2f} (±{statistics.stdev(v):.2f}) "
              f"{sum(x == 0 for x in v) / len(v):7.2f} "
              f"{sum(0 < x <= 1.5 for x in v) / len(v):12.2f} "
              f"{sum(x >= 4 for x in v) / len(v):8.2f}")
    print("\nShare of sends to defectors that are exactly 0, per run:")
    for key in sorted(per_run):
        v = per_run[key]
        print(f"{key[0]:16} {key[1]:10} runs={len(v):2d} "
              f"mean {statistics.mean(v):.2f} (±{statistics.stdev(v):.2f})")


if __name__ == "__main__":
    main()

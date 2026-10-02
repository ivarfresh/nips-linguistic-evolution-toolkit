"""Build a decision-level table for the partner-responsiveness analysis.

Question: when a myth task is added, do agents react more or less to what their
partner does in the game? Each row is one game decision with the partner signal
the agent had just been shown.

Two sources of chance variation in that signal are used:

* Communication noise. The trustee is shown the investor's send minus a draw
  from U(-range, 0); the investor is shown the trustee's return minus another
  draw. Draws are seeded by (noise_seed, round, dyad_id, action), so the same
  replicate faces the same draws in every task order. The raw draw is
  recomputed here, which also gives a placebo draw for no-noise runs.
* Random forced defection (2026-09-09 dyads, 25% / 50%). The schedule is keyed
  by (seed, round, agent, role) and is identical across task orders.

Run sets (selected from their manifests, never by globbing):
  figure2  docs/figures/figure2_noise_comparison_20260916/run_values.csv
           (270 no-defector runs: no noise, uninformed noise, informed noise)
  random   docs/figures/negative_only_crossmodel_reasoning_rerun_20260909/run_manifest.csv
           (dyads with 25% / 50% random forced defection, informed noise)
  frontier docs/figures/frontier_rerun_20260918/provenance.json
           (Opus 5 / Gemini 3.1 Pro / GPT-5.6 Sol, informed noise)

Usage (from repo root):
  python analyses/partner_responsiveness_extract.py
"""

import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "figures" / "partner_responsiveness_20260930"
ENDOWMENT = 5.0
MULTIPLIER = 3.0


def raw_noise_draw(noise_seed, turn, dyad_id, action_type, noise_range=1.0):
    """Recompute the U(-range, 0) draw of TrustGameNoisy._apply_noise_for_event."""
    material = f"{noise_seed}|{turn}|{dyad_id}|{action_type}".encode("utf-8")
    event_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(event_seed).uniform(-noise_range, 0)


def model_label(model_id):
    return {
        "anthropic/claude-sonnet-4.5": "Sonnet 4.5",
        "openai/gpt-5-nano": "GPT-5 Nano",
        "google/gemini-3.7-flash": "Gemini 3.7 Flash",
        "anthropic/claude-opus-5": "Opus 5",
        "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
        "openai/gpt-5.6-sol": "GPT-5.6 Sol",
    }.get(model_id, model_id)


def run_specs():
    specs = []
    for row in csv.DictReader(open(ROOT / "docs/figures/figure2_noise_comparison_20260916/run_values.csv")):
        specs.append({
            "run_set": "figure2",
            "path": row["source_path"],
            "noise": row["noise"],
            "treatment": "control",
            "replicate": int(row["replicate"]),
        })
    manifest = ROOT / "docs/figures/negative_only_crossmodel_reasoning_rerun_20260909/run_manifest.csv"
    for row in csv.DictReader(open(manifest)):
        if row["num_agents"] == "2" and row["treatment"].startswith("random"):
            specs.append({
                "run_set": "random",
                "path": row["path"],
                "noise": "noise_informed",
                "treatment": row["treatment"],
                "replicate": int(row["replicate_id"]),
            })
    prov = json.load(open(ROOT / "docs/figures/frontier_rerun_20260918/provenance.json"))
    for run in prov["runs"]:
        if "noise_experiments/frontier_rerun_20260918/" not in run["path"]:
            continue  # the September reference runs are listed alongside
        specs.append({
            "run_set": "frontier",
            # The manifest records the absolute path of the worktree that built it.
            "path": "data/json/" + run["path"].split("data/json/", 1)[1],
            "noise": "noise_informed",
            "treatment": "control",
            "replicate": None,
        })
    return specs


def decision_rows(spec):
    data = json.load(open(ROOT / spec["path"]))
    meta = data["run_metadata"]
    noise_cfg = meta.get("noise_config") or {}
    noise_range = float(noise_cfg.get("range", 1.0)) if noise_cfg else 1.0
    noise_seed = meta["noise_seed"]
    num_agents = int(meta["num_agents"])
    task_order = "_".join(data["task_order"])
    replicate = spec["replicate"]
    if replicate is None:
        replicate = int(noise_seed) - 202608250
    base = {
        "run_set": spec["run_set"],
        "model": model_label(meta["model"]),
        "num_agents": num_agents,
        "noise": spec["noise"],
        "treatment": spec["treatment"],
        "task_order": task_order,
        "replicate": replicate,
        "path": spec["path"],
    }

    # Index every dyad by (round, agent) for lag lookups.
    by_agent_round = {}
    for rec in data["conversation_history"]:
        responses = rec.get("game_responses", {})
        for dyad in rec.get("dyads", []):
            inv, tru = dyad["investor"], dyad["trustee"]
            info = {
                "round": dyad["round"],
                "dyad_id": dyad["dyad_id"],
                "investor": inv,
                "trustee": tru,
                "sent": dyad["sent_decision"] if "sent_decision" in dyad else dyad["sent"],
                "sent_shown": dyad.get("sent_communicated", dyad["sent"]),
                "received_shown": dyad.get("received_communicated", dyad["received"]),
                "received": dyad["received"],
                "returned": dyad["returned_decision"] if "returned_decision" in dyad else dyad["returned"],
                "returned_shown": dyad.get("returned_communicated", dyad["returned"]),
                "inv_source": responses.get(inv, {}).get("response_source", "llm"),
                "tru_source": responses.get(tru, {}).get("response_source", "llm"),
                "inv_text": responses.get(inv, {}).get("content") or "",
                "tru_text": responses.get(tru, {}).get("content") or "",
                "u_sent": raw_noise_draw(noise_seed, dyad["round"], dyad["dyad_id"], "sent", noise_range),
                "u_returned": raw_noise_draw(noise_seed, dyad["round"], dyad["dyad_id"], "returned", noise_range),
            }
            if noise_cfg:
                # The recorded noise must equal the recomputed draw unless clamped at 0.
                if info["sent"] >= noise_range and abs((info["sent_shown"] - info["sent"]) - info["u_sent"]) > 0.011:
                    raise ValueError(f"noise draw mismatch in {spec['path']} round {dyad['round']}")
            by_agent_round[(inv, dyad["round"])] = ("investor", info)
            by_agent_round[(tru, dyad["round"])] = ("trustee", info)

    rows = []
    for (agent, rnd), (role, info) in sorted(by_agent_round.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        row = dict(base, round=rnd, agent=agent, role=role)
        if role == "trustee":
            row.update({
                "own_source": info["tru_source"],
                "text": info["tru_text"],
                "decision_frac": (info["returned"] / info["received_shown"]) if info["received_shown"] > 0 else None,
                "return_frac_actual": (info["returned"] / info["received"]) if info["received"] > 0 else None,
                "sig_actual": info["sent"] / ENDOWMENT,
                "sig_shown": info["sent_shown"] / ENDOWMENT,
                "sig_u": info["u_sent"] if noise_cfg else None,
                "placebo_u": None if noise_cfg else info["u_sent"],
                "sig_partner_forced": info["inv_source"] != "llm",
            })
            prev = by_agent_round.get((agent, rnd - 1)) if num_agents == 2 else None
            if prev and prev[0] == "investor":
                # Dyad trustees were investors last round: the partner's return.
                # prev_ret_shown is the noisy return; dyad prompts show it only
                # through the running earnings total, not as its own number.
                p = prev[1]
                row.update({
                    "prev_ret_actual": (p["returned"] / p["received"]) if p["received"] > 0 else None,
                    "prev_ret_shown": (p["returned_shown"] / p["received"]) if p["received"] > 0 else None,
                    "prev_ret_u": p["u_returned"] if noise_cfg else None,
                    "prev_ret_partner_forced": p["tru_source"] != "llm",
                    "own_prev_forced": p["inv_source"] != "llm",
                })
        else:
            row.update({
                "own_source": info["inv_source"],
                "text": info["inv_text"],
                "decision_frac": info["sent"] / ENDOWMENT,
            })
            if num_agents == 2:
                # Dyad investors were trustees last round: the latest partner
                # signal is the partner's previous send, shown with noise.
                prev = by_agent_round.get((agent, rnd - 1))
                prev2 = by_agent_round.get((agent, rnd - 2))
                if prev and prev[0] == "trustee":
                    p = prev[1]
                    row.update({
                        "sig_actual": p["sent"] / ENDOWMENT,
                        "sig_shown": p["sent_shown"] / ENDOWMENT,
                        "sig_u": p["u_sent"] if noise_cfg else None,
                        "placebo_u": None if noise_cfg else p["u_sent"],
                        "sig_partner_forced": p["inv_source"] != "llm",
                        "own_prev_forced": p["tru_source"] != "llm",
                    })
                if prev2 and prev2[0] == "investor":
                    p = prev2[1]
                    row.update({
                        "ret2_actual": (p["returned"] / p["received"]) if p["received"] > 0 else None,
                        "ret2_shown_amount": p["returned_shown"],
                        "ret2_u": p["u_returned"] if noise_cfg else None,
                        "ret2_partner_forced": p["tru_source"] != "llm",
                        "own_prev2_forced": p["inv_source"] != "llm",
                        "own_prev2_sent": p["sent"] / ENDOWMENT,
                    })
        rows.append(row)
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    specs = run_specs()
    for spec in specs:
        rows.extend(decision_rows(spec))
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    out = OUT_DIR / "decisions.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(specs)} runs, {len(rows)} decisions -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

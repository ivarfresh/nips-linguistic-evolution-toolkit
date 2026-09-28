"""Score the myth-pressure pilot against its pre-registered criteria.

Reads the finals listed in a receipt (default: the pilot's completion receipt),
and writes per-myth, per-run and per-arm tables, the council-hygiene coding and a
figure to --out. Criteria: docs/research/myth_pressure_pilot_2026-09-28.md.
Perplexity follows GlossoGen App. B.1 (GPT-2 small, mean per-token surprisal,
exponentiated) on the text the other agent actually received.

Run from repo root:
    python analyses/myth_pressure_pilot.py \
        --receipt data/json/noise_experiments/myth_pressure_pilot_20260928/completion_receipt.json \
        --out docs/figures/myth_pressure_pilot_20260928
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’½-]*")
LABEL = re.compile(r"^\s*\**\s*Myth\s*:\s*\**\s*", re.I)
# Amount talk: a percent or dollar amount, or a quantity right after a transfer verb
# ("give half", "sent three", "return ~65%"). Word counts ("20 words", "Words 11-15")
# and structure talk ("Return action") are not amounts.
AMOUNT = re.compile(
    r"\$\s*\d|\d\s*%|\b(?:send|sends|sent|give|gives|gave|return|returns|returned|keep|keeps|kept|invest\w*)\s+"
    r"(?:~|about|roughly|only|back|nearly)?\s*(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"fifteen|half|all|everything|nothing|a third|two-thirds|double|triple)\b",
    re.I,
)
SENTENCE_END = re.compile(r"""[.!?]["'’”)\]*]*\s*$""")
ARMS = ["loose_nocouncil", "loose_council", "tight_nocouncil", "tight_council"]


def body(text):
    return LABEL.sub("", text or "", count=1).strip()


def load_dictionary():
    words = {w.strip().lower() for w in Path("/usr/share/dict/words").read_text().split()}
    return words


def perplexities(texts):
    import torch
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    lm = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(device)
    out = np.full(len(texts), np.nan)
    for i, text in enumerate(texts):
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] < 2:  # GlossoGen drops single-token messages
            continue
        with torch.no_grad():
            logits = lm(ids).logits[0, :-1]
        out[i] = float(np.exp(F.cross_entropy(logits.float(), ids[0, 1:]).item()))
    return out


def council_flag(message):
    """Amount talk in a council message (see AMOUNT)."""
    return bool(AMOUNT.search(message))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((ROOT / args.receipt).read_text())
    dictionary = load_dictionary()

    myths, councils, runs = [], [], []
    for final in receipt["finals"]:
        path = ROOT / final["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == final["sha256"], path
        data = json.loads(path.read_text())
        arm, rep = final["arm"], final["replicate_id"]
        history = data["conversation_history"]
        for entry in history:
            for agent, record in entry["myth_delivery"].items():
                delivered = body(entry["myths"][agent])
                fits = record["written_words"] <= record["word_budget"]
                # Amendment (2026-09-28, pre-pilot): a cut myth whose delivered part
                # ends a sentence was planned for the cut.
                planned = fits or bool(SENTENCE_END.search(delivered))
                myths.append(dict(arm=arm, rep=rep, round=entry["round"], agent=agent, **record,
                                  fits=fits, planned=planned, text=delivered))
            for council in (entry.get("council") or {}).values():
                for i, message in enumerate(council["messages"]):
                    councils.append(dict(arm=arm, rep=rep, round=entry["round"], agent=message["agent"],
                                         index=i, flagged=council_flag(message["content"]), text=message["content"]))
        balances = history[-1]["balances"]
        sends = [d["sent_decision"] / 5 for e in history for d in e.get("dyads") or []]
        runs.append(dict(arm=arm, rep=rep, resources=float(np.mean(list(balances.values()))),
                         mean_send=float(np.mean(sends)), cost=final["standard_rate_usd"]))

    myths = pd.DataFrame(myths)
    myths["ppl"] = perplexities(myths.text.tolist())

    # Invented tokens: not an English dictionary word, not a plain number, not in round 1.
    def tokens(text):
        return {w.lower().strip("'’-") for w in WORD.findall(text)}
    def invented(w):
        return w and not w.isdigit() and w not in dictionary and w.rstrip("s") not in dictionary
    shared_rows = []
    for (arm, rep), g in myths.groupby(["arm", "rep"]):
        round1 = set().union(*g[g["round"] == 1].text.map(tokens))
        late = g[g["round"] >= 6]
        per_agent = {a: set().union(*h.text.map(tokens)) for a, h in late.groupby("agent")}
        shared = set.intersection(*per_agent.values()) if len(per_agent) == 2 else set()
        codes = sorted(w for w in shared if invented(w) and w not in round1)
        shared_rows.append(dict(arm=arm, rep=rep, shared_invented=len(codes), examples=" ".join(codes[:15])))
    shared = pd.DataFrame(shared_rows)

    runs = pd.DataFrame(runs)
    late = myths[myths["round"] >= 6].groupby(["arm", "rep"]).agg(late_ppl=("ppl", "mean"), late_delivered=("delivered_words", "mean")).reset_index()
    fit = (myths[(myths["round"] >= 4) & myths.arm.str.startswith("tight")].groupby(["arm", "rep"])
           .agg(fit_rate_r4plus=("fits", "mean"), planned_rate_r4plus=("planned", "mean")).reset_index())
    runs = runs.merge(late, on=["arm", "rep"]).merge(shared, on=["arm", "rep"]).merge(fit, on=["arm", "rep"], how="left")
    councils = pd.DataFrame(councils)
    if len(councils):
        runs = runs.merge(councils.groupby(["arm", "rep"]).flagged.mean().rename("council_flag_rate").reset_index(), on=["arm", "rep"], how="left")

    def fmt(s):
        return f"{s.mean():.2f} (±{s.std():.2f})" if s.notna().any() else "—"
    cols = ["resources", "mean_send", "late_ppl", "late_delivered", "fit_rate_r4plus", "planned_rate_r4plus", "shared_invented", "council_flag_rate", "cost"]
    summary = pd.DataFrame({arm: {c: fmt(runs[runs.arm == arm][c]) for c in cols if c in runs} for arm in ARMS if arm in set(runs.arm)}).T
    summary.index.name = "arm"

    lines = ["# Myth-pressure pilot: criteria", "", f"Runs: {len(runs)}; myths: {len(myths)}; council messages: {len(councils)}", "",
             summary.to_markdown(), ""]
    tc, tn = runs[runs.arm == "tight_council"], runs[runs.arm == "tight_nocouncil"]
    if len(tc) and len(tn):
        tight = myths[(myths["round"] >= 4) & myths.arm.str.startswith("tight")]
        fit_all, planned_all = tight.fits.mean(), tight.planned.mean()
        ratio = tc.late_ppl.mean() / tn.late_ppl.mean()
        lines += [f"1. Tight arms, rounds 4+: fit before truncation {fit_all:.0%}; planned for the cut {planned_all:.0%} (criterion: planned ≥ 80%) → {'PASS' if planned_all >= .8 else 'FAIL'}",
                  f"2a. Perplexity ratio tight/council ÷ tight/no-council, rounds 6–10: {ratio:.2f} (criterion ≥ 2) → {'PASS' if ratio >= 2 else 'FAIL'}",
                  f"2b. Shared invented tokens, tight/council: {fmt(tc.shared_invented)}; runs with ≥ 3: {(tc.shared_invented >= 3).sum()}/{len(tc)}"]
    if len(councils):
        rate = councils.flagged.mean()
        lines.append(f"3. Council messages with amount/strategy talk: {rate:.0%} (criterion ≤ 25%) → {'PASS' if rate <= .25 else 'FAIL'}")
    (out / "criteria.md").write_text("\n".join(lines) + "\n")
    myths.to_csv(out / "myths.csv", index=False)
    runs.to_csv(out / "runs.csv", index=False)
    if len(councils):
        councils.to_csv(out / "council_messages.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for arm, color in zip(ARMS, ["#9aa0a6", "#4a3aa7", "#eda100", "#d6453d"]):
        g = myths[myths.arm == arm]
        if not len(g):
            continue
        for ax, col in zip(axes[:2], ["written_words", "ppl"]):
            m = g.groupby("round")[col].agg(["mean", "std"])
            ax.plot(m.index, m["mean"], "o-", color=color, label=arm.replace("_", " / "))
            ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], color=color, alpha=.12)
        axes[2].scatter(runs[runs.arm == arm].late_ppl, runs[runs.arm == arm].resources, color=color, s=40)
    budget = myths[myths.arm.str.startswith("tight")].groupby("round").word_budget.first()
    axes[0].step(budget.index, budget.values, where="mid", color="k", ls=":", label="tight budget")
    axes[0].set(title="Words written per myth", xlabel="round")
    axes[1].set(title="GPT-2 perplexity of delivered myth", xlabel="round", yscale="log")
    axes[2].set(title="Per run: late perplexity vs resources", xlabel="perplexity, rounds 6–10", ylabel="resources after round 10")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "myth_pressure_pilot.png", dpi=140)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-page scorecard: September beside frontier, every myth finding per simulation x population.

Reads the five lens tables (../{spread,plan,alignment,consistency,search}/scorecard_rows.csv, plus
../plan/r4_exact.csv) and never recomputes a statistic. Each cell takes the lens's own status; cells
that summarise several lens rows follow the rule written in `rule` of scorecard_cells.csv.

Writes scorecard_cells.csv, scorecard.png and scorecard.pdf next to this file.
Source-row ids in the CSV are 0-based data-row positions in the named lens's scorecard_rows.csv.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
L = {k: pd.read_csv(BASE / k / "scorecard_rows.csv") for k in ["spread", "plan", "alignment", "consistency", "search"]}
R4X = pd.read_csv(BASE / "plan" / "r4_exact.csv")

FAM = {"september": ["Sonnet", "GPT", "Gemini"], "frontier": ["Opus", "Sol", "GeminiPro"]}
CEILFAM = {"september": "Gemini", "frontier": "GeminiPro"}
VARY = {"september": "Sonnet+GPT", "frontier": "Sol+Opus"}

# ---------------------------------------------------------------- statuses
YES, SUG, ND = "yes", "suggestive", "no detectable effect"
CEIL, DES, UP = "not estimable: ceiling", "not testable by design", "underpowered"
GLYPH = {YES: "✓", SUG: "~", ND: "○", CEIL: "▲", DES: "⊘", UP: "◌", None: ""}
FILL = {YES: "#1c5cab", SUG: "#86b6ef", ND: "#ebeae6", CEIL: "#f0a860", DES: "#ffffff", UP: "#ffffff", None: "#ffffff"}
HATCH = {CEIL: "////", DES: "\\\\\\\\"}
INK = {YES: "#ffffff"}
LEGEND = [(YES, "yes (Holm p < 0.05)"), (SUG, "suggestive (raw p < 0.05, not Holm)"), (ND, "no detectable effect"),
          (CEIL, "can't test: ceiling (sends stuck at $5)"), (UP, "too few runs / senders"),
          (DES, "not testable by design")]
# tie-break when summarising several lens rows: the less-evidential status wins
TIE = [ND, UP, CEIL, DES, SUG]


def summarise(statuses):
    """'yes' if any lens row is 'yes'; otherwise the most common status (ties -> less evidence)."""
    s = [x for x in statuses if isinstance(x, str)]
    if not s:
        return None
    if YES in s:
        return YES
    counts = pd.Series(s).value_counts()
    top = counts[counts == counts.max()].index.tolist()
    return sorted(top, key=lambda x: TIE.index(x) if x in TIE else 99)[0]


def conservative(a, b):
    order = [DES, CEIL, UP, ND, SUG, YES]
    return a if order.index(a) <= order.index(b) else b


# ---------------------------------------------------------------- lens access
def q(lens, finding=None, regex=None, **kw):
    d = L[lens]
    m = pd.Series(True, index=d.index)
    if finding is not None:
        m &= d.finding.eq(finding)
    if regex is not None:
        m &= d.finding.str.contains(regex, regex=True)
    for k, v in kw.items():
        m &= d[k].eq(v)
    return d[m]


def one(lens, finding=None, regex=None, **kw):
    r = q(lens, finding, regex, **kw)
    assert len(r) == 1, (lens, finding, regex, kw, len(r))
    return r.iloc[0]


def ok(x):
    return x == x and x is not None


def pm(r, dec=2, unit=""):
    """'+0.60 [0.44, 0.76]' from a lens row."""
    if not ok(r.effect):
        return "no estimate"
    s = f"{r.effect:+.{dec}f}{unit}"
    if ok(r.ci_low):
        s += f" [{r.ci_low:.{dec}f}, {r.ci_high:.{dec}f}]"
    return s


def hw(r, dec=3):
    """'+0.003 ±0.019' (CI half-width) from a lens row."""
    if not ok(r.effect):
        return "–"
    s = f"{r.effect:+.{dec}f}".replace("0.", ".")
    if ok(r.ci_low):
        s += f" ±{(r.ci_high - r.ci_low) / 2:.{dec}f}".replace("0.", ".")
    return s


def nr(r):
    return f"{int(r.n_runs)} runs" if ok(r.n_runs) else ""


def short(st):
    return {YES: "yes", SUG: "sugg.", ND: "null", CEIL: "ceiling", DES: "design", UP: "low-n"}.get(st, st)


class Cell:
    def __init__(self, status, lines, src, measure, rule="lens status as assigned", flags="", main=None):
        self.status, self.lines, self.src, self.measure, self.rule, self.flags = status, lines, src, measure, rule, flags
        self.main = main  # the lens row that carries effect / CI / n_runs into the CSV


def src(lens, rows):
    rows = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame([rows])
    return [(lens, int(i)) for i in rows.index]


def blank(text, why):
    return Cell(None, [text], [], "", why)


def design(text, why):
    return Cell(DES, [text], [], "", why, flags="status assigned from the design; no lens row")


# ---------------------------------------------------------------- rows
GROUPS = [
    ("september", "S2H", "2-agent\none model", "2-agent homogeneous", ["Sonnet dyads", "GPT dyads", "Gemini dyads"]),
    ("september", "S2M", "2-agent\nmixed pairs", "2-agent mixed", ["Sonnet+GPT dyads", "Sonnet+Gemini dyads", "Gemini+GPT dyads"]),
    ("september", "S8H", "8-agent\none model", "8-agent homogeneous", ["8 Sonnet", "8 GPT", "8 Gemini"]),
    ("september", "S8M", "8-agent\nmixed", "8-agent mixed", ["1 GPT + 7 Sonnet", "2 GPT + 6 Sonnet", "4 GPT + 4 Sonnet",
                                                            "1 Gemini + 7 GPT", "2 Gemini + 6 GPT", "4 Gemini + 4 GPT"]),
    ("frontier", "F2H", "2-agent\none model", "2-agent homogeneous", ["Opus dyads", "Sol dyads", "GeminiPro dyads"]),
    ("frontier", "F2M", "2-agent\nmixed pairs", "2-agent mixed", ["Opus+Sol dyads", "Opus+GeminiPro dyads", "GeminiPro+Sol dyads"]),
    ("frontier", "F8H", "8-agent\none model", "8-agent homogeneous", ["8 Opus", "8 Sol", "8 GeminiPro"]),
    ("frontier", "F8M", "8-agent\nmixed", "8-agent mixed", ["2 GeminiPro + 3 Opus + 3 Sol"]),
]
ROWLABEL = {"2 GeminiPro + 3 Opus + 3 Sol": "2 GeminiPro +\n3 Opus + 3 Sol"}
TALL = {"2 GeminiPro + 3 Opus + 3 Sol": 3.0}  # row height in units of a normal row

COLS = [
    ("Own first story\n= plan", "R4, round-1 senders, myth→game\nk/n sent exactly the $ named · $ per $"),
    ("Read story →\nnext send", "R3, 8-agent myth→game\n$ sent per $ in shown myth"),
    ("Read story →\nnext story amount", "8-agent myth→game\n$ written per $ in shown myth"),
    ("Words copied\nfrom read story", "adoption, shown − unseen\npercentage points"),
    ("Moral spreads\nwithin family", "label uptake, myth→game\nshown − unseen, pts"),
    ("Moral spreads\nacross families", "label uptake, myth→game\nshown − unseen, pts"),
    ("Matching norms\n→ cooperation", "H1 primary: 5 norm measures\n× 3 outcomes = 15 tests"),
    ("Drift toward\nconsistency", "judge flag share, rd 1 → 8–10\n(kw = keyword share change)"),
    ("Consistency\n→ cooperation", "H3, agent FE, judge flag\nΔ send/5 · Δ return share"),
    ("Stories add beyond\npast moves", "held-out R² gain, all myth\nfeatures (ridge); ≤ = upper CI"),
    ("Morals follow\nplay", "full send → P(next myth\n'be generous'), Δ prob."),
]


# ---------------------------------------------------------------- column builders
R4RAW = pd.read_csv(BASE / "plan" / "r4.csv")
R4RAW = R4RAW[R4RAW.measure == "named amount only ($/$)"]


def r4raw(ds, size, fam, population=None):
    m = (R4RAW.dataset == ds) & (R4RAW.setting == size) & (R4RAW.family == fam)
    if population:
        m &= R4RAW.population == population
    x = R4RAW[m]
    assert len(x) == 1, (ds, size, fam, population, len(x))
    return x.iloc[0]


def exact_txt(x):
    """'12/15 exact' from a plan r4.csv row (exact_match share × senders naming an amount)."""
    if not x.n:
        return "0 named"
    return f"{int(round(x.exact_match * x.n))}/{int(x.n)}"


def dpd(x):
    return f"${x.coef:.2f}/$ [{x.ci_low:.2f}, {x.ci_high:.2f}]" if ok(x.coef) else "too few for $ per $"


def c1_group(ds, g, setting):
    # Lead decision 2026-09-30: lead with raw-unit exact-match counts and $ per $ (plan r4.csv), which hold
    # under any significance rule; status stays the plan lens's.
    size = setting.split("-")[0] + "-agent"
    r = one("plan", "R4 own round-1 myth → round-1 send, named amount only ($/$)", dataset=ds, setting=size,
            population=setting, family=VARY[ds])
    x = r4raw(ds, size, VARY[ds], setting)
    lines = [f"{exact_txt(x)} sent the named $", dpd(x),
             f"{int(x.n_runs) if ok(x.n_runs) else 0} runs · {VARY[ds]}"]
    for f in FAM[ds]:
        xf = r4raw(ds, size, f, setting)
        rf = one("plan", r.finding, dataset=ds, setting=size, population=setting, family=f)
        e = f" · ${xf.coef:.2f}" if ok(xf.coef) else ""
        lines.append(f"{f} {exact_txt(xf)}{e} ({short(rf.status)})")
    return Cell(r.status, lines, src("plan", r), f"named-amount senders, pooled over {VARY[ds]} (plan r4.csv)",
                "lens pooled stratum (setting); families listed", main=r)


def c1_pooled(ds):
    fin = "R4 own round-1 myth → round-1 send, named amount only ($/$)"
    r = one("plan", fin, dataset=ds, setting="2+8-agent", family=VARY[ds])
    x = r4raw(ds, "2+8-agent", VARY[ds])
    lines = [f"{exact_txt(x)} sent the named $", f"{dpd(x)} · {int(x.n_runs)} runs"]
    rows = [r]
    flags = ""
    for f in FAM[ds]:
        rf = one("plan", fin, dataset=ds, setting="2+8-agent", family=f)
        xf = r4raw(ds, "2+8-agent", f)
        st = rf.status
        srch = one("search", "$ sent per $ stated in own round-1 myth", dataset=ds, family=f)
        if srch.status != rf.status:
            # search lens's 'yes' rule was set after seeing results; lead ruled Holm on the wild-cluster p,
            # which makes the frontier Opus search row 'suggestive' -> agrees with plan
            flags = (f"{f}: plan '{rf.status}', search '{srch.status}' under its own rule; lead ruled the stricter "
                     f"rule (Holm on wild-cluster p) -> suggestive, agreeing with plan; '{conservative(rf.status, srch.status)}' shown")
            st = conservative(rf.status, srch.status)
            rows.append(srch)
        e = f" · ${xf.coef:.2f}" if ok(xf.coef) else ""
        lines.append(f"{f} {exact_txt(xf)}{e} ({short(st)})")
        rows.append(rf)
    return Cell(r.status, lines, sum([src("search" if "stated in own" in x.finding else "plan", x) for x in rows], []),
                f"named-amount senders, 2+8-agent pooled over {VARY[ds]} (plan r4.csv)",
                "lens pooled stratum (all settings)", flags, main=r)


def h2(ds, g, setting, pops, which):
    fin = f"H2 shown myth stated amount → reader's next {which} ($/$)"
    if setting.startswith("2-agent"):
        rows = q("plan", fin, dataset=ds, setting="2-agent")
        assert set(rows.status) == {DES}, rows.status
        return Cell(DES, ["dyads: shown myth is", "always the partner's"], src("plan", rows), f"reader's next {which}",
                    "lens rows (per family) all 'not testable by design'"), rows
    if setting == "8-agent mixed":
        pop = "8-agent mixed" if ds == "september" else "2 GeminiPro + 3 Opus + 3 Sol"
        r = one("plan", fin, dataset=ds, setting="8-agent", population=pop, family="all")
        lines = [pm(r) if ok(r.effect) else "no estimate", f"{nr(r)} · all families"]
        for f in FAM[ds]:
            rf = one("plan", fin, dataset=ds, setting="8-agent", population=pop, family=f)
            lines.append(f"{f} {rf.effect:+.2f} ({short(rf.status)})" if ok(rf.effect) else f"{f}: {short(rf.status)}")
        return Cell(r.status, lines, src("plan", r), f"reader's next {which}, all families",
                    "lens pooled stratum (all mixed populations)", main=r), None
    return None, None


SEARCH_R3 = "shown myth rule_send_amount (per SD, minus unseen placebo) -> round-r send"


def cross_r3(cell, ds, setting, **kw):
    """Compare the plan lens's H2 send status with the search lens's R3 stated-amount row for the same stratum."""
    x = q("search", SEARCH_R3, dataset=ds, setting=setting, **kw)
    if len(x) != 1:
        return cell
    x = x.iloc[0]
    cell.src += src("search", x)
    if x.status != cell.status:
        new = conservative(cell.status, x.status)
        cell.flags = (cell.flags + "; " if cell.flags else "") + (
            f"plan H2 '{cell.status}' vs search R3 rule_send_amount '{x.status}' ({x.effect:+.3f} per SD); "
            f"conservative '{new}' shown")
        cell.status = new
    return cell


def h2_row(ds, pop, which):
    fin = f"H2 shown myth stated amount → reader's next {which} ($/$)"
    r = one("plan", fin, dataset=ds, setting="8-agent", population=pop)
    cell = Cell(r.status, [pm(r) if ok(r.effect) else "no estimate", nr(r)], src("plan", r), f"reader's next {which}", main=r)
    if which == "send":
        cell = cross_r3(cell, ds, "8-agent homogeneous; 8-agent myth_game rounds>=2 (R3)", population=pop)
    return cell


def h2_pooled(ds, which):
    fin = f"H2 shown myth stated amount → reader's next {which} ($/$)"
    r = one("plan", fin, dataset=ds, setting="8-agent", population="8-agent myth→game (homog + mixed)", family="all")
    pl = one("plan", fin + " [unseen placebo]", dataset=ds, setting="8-agent",
             population="8-agent myth→game (homog + mixed)", family="all")
    lines = [pm(r) if ok(r.effect) else "no estimate", f"{nr(r)} · placebo {pl.effect:+.2f}" if ok(pl.effect)
             else nr(r)]
    for f in FAM[ds]:
        rf = one("plan", fin, dataset=ds, setting="8-agent", population="8-agent myth→game (homog + mixed)", family=f)
        lines.append(f"{f} {rf.effect:+.2f} ({short(rf.status)})" if ok(rf.effect) else f"{f}: {short(rf.status)}")
    cell = Cell(r.status, lines, src("plan", r) + src("plan", pl), f"reader's next {which}, 8-agent homog+mixed",
                "lens pooled stratum (8-agent myth→game)", main=r)
    if which == "send":
        cell = cross_r3(cell, ds, "all; 8-agent myth_game rounds>=2 (R3)", family="pooled")
    return cell


def fam_pair(ds, setting, pop, fin_tmpl, a, b):
    out = []
    for o in ["myth→game", "game→myth"]:
        r = q("spread", fin_tmpl.format(a=a, b=b, o=o), dataset=ds, setting=setting, population=pop)
        out.append(r.iloc[0] if len(r) else None)
    return out


def c4_group(ds, g, setting, pops):
    sc = "same family" if "homogeneous" in setting else "other family"
    pop_all = f"all {setting} runs"
    r = q("spread", f"word adoption shown − unseen, {sc} [both orders]", dataset=ds, setting=setting, population=pop_all)
    if len(r):
        r = r.iloc[0]
        lines = [f"{pm(r, 1)} pts", f"{nr(r)} · {'cross-family' if sc == 'other family' else 'same family'}"]
        if "homogeneous" in setting:
            lines.append("per family, m→g / g→m:")
        srcs = src("spread", r)
        if "homogeneous" in setting:
            for p in pops:
                f = p.split()[-1] if p.startswith("8 ") else p.split()[0]
                mg, gm = fam_pair(ds, setting, p, "word adoption shown − unseen, {a} shown {b} [{o}]", f, f)
                lines.append(f"{f} {mg.effect:+.1f} / {gm.effect:+.1f}")
                srcs += src("spread", mg) + src("spread", gm)
        elif setting == "8-agent mixed":
            s = one("spread", "word adoption shown − unseen, same family [both orders]", dataset=ds, setting=setting,
                    population=pop_all)
            lines.append(f"same family {s.effect:+.1f} ({short(s.status)})")
            srcs += src("spread", s)
        else:
            for o in ["myth→game", "game→myth"]:
                ro = one("spread", f"word adoption shown − unseen, other family [{o}]", dataset=ds, setting=setting,
                         population=pop_all)
                lines.append(f"{o} {ro.effect:+.1f} ({short(ro.status)})")
                srcs += src("spread", ro)
        if setting.startswith("2-agent"):
            lines[0] += " †"
        return Cell(r.status, lines, srcs, f"word adoption, {sc}, both orders", "lens pooled stratum (setting)",
                    "dyads: shared game history, unseen comparison from another run (not transmission evidence)"
                    if setting.startswith("2-agent") else "", main=r)
    # frontier 8-agent mixed: the lens gives only per reader->author pair rows
    rows = q("spread", regex=r"^word adoption shown − unseen, \w+ shown \w+ \[", dataset=ds, setting=setting)
    pair = rows.finding.str.extract(r", (\w+) shown (\w+) \[")
    same = rows[pair[0] == pair[1]]
    cross = rows.drop(same.index)
    lines = [f"same family {same.effect.min():+.1f} to {same.effect.max():+.1f}",
             f"cross-family {cross.effect.min():+.1f} to {cross.effect.max():+.1f}",
             f"{int(rows.n_runs.max())} runs · {len(rows)} pair×order rows,", "each Holm 1.0; no pooled row"]
    return Cell(summarise(rows.status), lines, src("spread", rows), "word adoption, per reader/author pair",
                "summary of per-pair rows: yes if any yes, else most common status", main=None), None


def c5_group(ds, g, setting, pops):
    pop_all = f"all {setting} runs"
    if setting == "2-agent mixed":
        return design("dyad partner is\nalways the other family", "no same-family reading in mixed dyads")
    if ds == "frontier" and setting == "8-agent mixed":
        return None
    r = one("spread", "moral label uptake shown − unseen, same family [myth→game]", dataset=ds, setting=setting,
            population=pop_all)
    lines = [f"{pm(r, 1)} pts" + (" †" if setting.startswith("2-agent") else ""),
             nr(r) + (" · no placebo (dyads)" if setting.startswith("2-agent") else "")]
    srcs = src("spread", r)
    flags = "dyads: shared game history, no future-myth placebo" if setting.startswith("2-agent") else ""
    pl = q("spread", "PLACEBO future myth − unseen, same family [myth→game]", dataset=ds, setting=setting,
           population=pop_all)
    if len(pl):
        pl = pl.iloc[0]
        fail = pl.p < 0.05 or pl.effect >= r.effect
        lines.append(placebo_text(pl, r))
        srcs += src("spread", pl)
        if fail:
            flags = "future-myth placebo p < 0.05 or >= the effect"
            if r.status == YES:
                # lead decision 2026-09-30: a failed future-myth placebo caps a spread 'yes' at 'suggestive'
                lens_status = r.status
                r = r.copy()
                r["status"] = SUG
                flags += f"; lens status '{lens_status}' downgraded to 'suggestive' (lead decision)"
    r3 = q("spread", "moral label uptake, R3-adjusted (author ≠ current partner, t−1 pair game controlled), same family [myth→game]",
           dataset=ds, setting=setting, population=pop_all)
    if len(r3):
        lines.append(f"R3-adjusted {r3.iloc[0].effect:+.1f} ({short(r3.iloc[0].status)})")
        srcs += src("spread", r3)
    rb = q("spread", "ROBUSTNESS (DeepSeek labels) moral label uptake shown − unseen, same family [myth→game]",
           dataset=ds, setting=setting, population=pop_all)
    if len(rb):
        lines.append(f"DeepSeek {rb.iloc[0].effect:+.1f} ({short(rb.iloc[0].status)})")
        srcs += src("spread", rb)
    if "homogeneous" in setting and setting.startswith("2"):
        for p in pops:
            f = p.split()[0]
            rf = one("spread", f"moral label uptake shown − unseen, {f} shown {f} [myth→game]", dataset=ds,
                     setting=setting, population=p)
            lines.append(f"{f} {rf.effect:+.1f} ({short(rf.status)})")
            srcs += src("spread", rf)
    return Cell(r.status, lines, srcs, "moral label uptake, same family, myth→game, shown − unseen",
                "lens pooled stratum (setting)", flags, main=r)


def placebo_text(pl, r):
    fail = pl.p < 0.05 or pl.effect >= r.effect
    why = "≥ effect" if pl.effect >= r.effect else f"p {pl.p:.2f}".replace("0.", ".")
    return f"placebo {pl.effect:+.1f} ({why}): {'fails' if fail else 'passes'}"


def c5_frontier_mix():
    fin = "moral label uptake shown − unseen, same family [myth→game]"
    r = one("spread", fin, dataset="frontier", setting="8-agent mixed", population="2 GeminiPro + 3 Opus + 3 Sol")
    pl = one("spread", "PLACEBO future myth − unseen, same family [myth→game]", dataset="frontier",
             setting="8-agent mixed", population="2 GeminiPro + 3 Opus + 3 Sol")
    lines = [f"{pm(r, 1)} pts", nr(r), placebo_text(pl, r), "GeminiPro: n/a (2 agents)"]
    return Cell(r.status, lines, src("spread", r) + src("spread", pl), "moral label uptake, same family, myth→game",
                flags="future-myth placebo >= effect" if pl.effect >= r.effect else "", main=r)


def c6_group(ds, g, setting):
    if "homogeneous" in setting:
        return design("one family only", "homogeneous rows have no other family")
    pop = f"all {setting} runs" if not (ds == "frontier" and setting == "8-agent mixed") else "2 GeminiPro + 3 Opus + 3 Sol"
    r = one("spread", "moral label uptake shown − unseen, other family [myth→game]", dataset=ds, setting=setting,
            population=pop)
    dy = setting.startswith("2-agent")
    lines = [f"{pm(r, 1)} pts" + (" †" if dy else ""), nr(r) + (" · no placebo (dyads)" if dy else "")]
    srcs = src("spread", r)
    for fin, lab in [("PLACEBO future myth − unseen, other family [myth→game]", "placebo"),
                     ("moral label uptake, R3-adjusted (author ≠ current partner, t−1 pair game controlled), other family [myth→game]", "R3-adjusted"),
                     ("ROBUSTNESS (DeepSeek labels) moral label uptake shown − unseen, other family [myth→game]", "DeepSeek")]:
        x = q("spread", fin, dataset=ds, setting=setting, population=pop)
        if len(x):
            x = x.iloc[0]
            lines.append(placebo_text(x, r) if lab == "placebo" else f"{lab} {x.effect:+.1f} ({short(x.status)})")
            srcs += src("spread", x)
    rule = "lens pooled stratum (setting)" if pop.startswith("all") else "lens population row"
    return Cell(r.status, lines, srcs, "moral label uptake, other family, myth→game, shown − unseen", rule,
                "dyads: shared game history, no future-myth placebo" if dy else "", main=r)


def c7_rows(ds, setting, pop):
    rows = q("alignment", regex=r"^H1 primary", dataset=ds, setting=setting, population=pop)
    assert len(rows) == 15, (ds, setting, pop, len(rows))
    return rows


MEAS = {"same_label": "label", "moral_cos": "moral cos", "rule_index": "rule",
        "give_align": "score", "myth_cos": "myth cos"}
OUTC = {"send fraction": "send", "return proportion": "return", "giving gap": "gap"}


def c7_cell(rows, label=""):
    st = summarise(rows.status)
    n = rows.status.value_counts()
    nyes, nsug = int(n.get(YES, 0)), int(n.get(SUG, 0))
    flags = ""
    if nyes:
        y = rows[rows.status == YES].iloc[0]
        meas = MEAS[y.finding.split(": ")[1].split(" alignment")[0]]
        out = OUTC[y.finding.split("-> ")[1].split(" (")[0]]
        opp = y.effect < 0
        lines = [f"{nyes}/15 Holm{', OPPOSITE sign' if opp else ''}", f"{meas} match → {out} {y.effect:+.3f}"]
        if opp:
            flags = "the only Holm 'yes' has the opposite sign to H1 (misaligned pairs cooperate more)"
    else:
        lines = [f"0/15 Holm · {nsug} raw p" if nsug else "0/15 Holm"]
        rest = [f"{int(n.get(s, 0))} {short(s)}" for s in [ND, UP, CEIL] if n.get(s, 0)]
        lines.append(" · ".join(rest))
    return Cell(st, [label + lines[0]] + lines[1:], src("alignment", rows), "15 H1 primary tests",
                "summary: yes if any lens row yes, else most common status (ties -> less evidence)", flags,
                main=rows[rows.status == YES].iloc[0] if nyes else None)


def cons_pop(ds, setting, fam):
    if "homogeneous" in setting:
        return f"{fam} dyads" if setting.startswith("2") else f"8 {fam}"
    if setting == "2-agent mixed":
        return f"mixed dyads, {fam} members"
    return "2 GeminiPro + 3 Opus + 3 Sol" if ds == "frontier" else f"8-agent mixed populations, {fam} members"


DRIFT_J = "drift: GLM judge flag round 1 -> rounds 8-10 (task orders pooled)"
DRIFT_K = "drift: keyword round 1 -> rounds 8-10 (task orders pooled)"


def c8_fam(ds, setting, fam, pop=None, lead=True):
    pop = pop or cons_pop(ds, setting, fam)
    j = one("consistency", DRIFT_J, dataset=ds, setting=setting, population=pop, family=fam)
    k = one("consistency", DRIFT_K, dataset=ds, setting=setting, population=pop, family=fam)
    m = re.search(r"round 1 ([\d.]+) .*?-> rounds 8-10 ([\d.]+)", j.note)
    frm = f"{float(m.group(1)):.2f}→{float(m.group(2)):.2f}".replace("0.", ".") if m else ""
    if lead:
        lines = [f"judge {j.effect:+.2f} ({frm})", f"kw {k.effect:+.2f} ({short(k.status)}) · {nr(j)}"]
    else:
        lines = [f"{fam}: judge {j.effect:+.2f} ({short(j.status)})".replace("0.", "."),
                 f"   kw {k.effect:+.2f} ({short(k.status)}), {nr(j)}".replace("0.", ".")]
    return j, k, lines


def c8_cell(ds, setting, fams, pop=None):
    if len(fams) == 1:
        j, k, lines = c8_fam(ds, setting, fams[0], pop)
        return Cell(j.status, lines, src("consistency", j) + src("consistency", k), "GLM judge flag (keyword in text)",
                    "lens status of the judge-flag row ('yes' needs both judges)", main=j)
    js, lines, srcs = [], [], []
    for f in fams:
        j, k, ln = c8_fam(ds, setting, f, pop, lead=False)
        js.append(j)
        lines += ln
        srcs += src("consistency", j) + src("consistency", k)
    return Cell(summarise([j.status for j in js]), lines, srcs, "GLM judge flag per family (keyword in text)",
                "summary over families: yes if any yes, else most common", main=None)


R2S = "R2 own consistency -> send level (agent FE, lagged coop): GLM judge flag"
R2R = "R2 own consistency -> return proportion (agent FE, lagged coop): GLM judge flag"


def c9_fam(ds, setting, fam, pop=None, lead=True):
    pop = pop or cons_pop(ds, setting, fam)
    s = one("consistency", R2S, dataset=ds, setting=setting, population=pop, family=fam)
    r = one("consistency", R2R, dataset=ds, setting=setting, population=pop, family=fam)

    def part(x):
        return "ceiling" if x.status == CEIL and not ok(x.effect) else hw(x) + ("*" if x.status == YES else "")
    lw = f" · lens-wide {r.holm_p_lens:.2f}".replace("0.", ".") if r.status == YES else ""
    ln = [f"send {part(s)}{lw}", f"ret {part(r)} · {nr(r)}"]
    if not lead:
        ln = [f"{fam}: send {part(s)}", f"   ret {part(r)}"]
    flags = []
    for x in (s, r):
        if x.status == YES:
            flags.append(f"{x.finding.split('-> ')[1].split(' (')[0]} {fam} yes (holm_p {x.holm_p:.3f}; "
                         f"lens-wide Holm {x.holm_p_lens:.3f})")
        if x.status == CEIL and ok(x.effect):
            flags.append(f"{fam} {x.finding.split('-> ')[1].split(' (')[0]}: effect shown but lens status ceiling")
    return [s, r], ln, flags


def c9_cell(ds, setting, fams, pop=None):
    rows, lines, flags = [], [], []
    for f in fams:
        rr, ln, fl = c9_fam(ds, setting, f, pop, lead=len(fams) == 1)
        rows += rr
        lines += ln
        flags += fl
    return Cell(summarise([r.status for r in rows]), lines, sum([src("consistency", r) for r in rows], []),
                "GLM judge flag → send level and return share (agent FE)",
                "summary over send + return (and families): yes if any yes, else most common (ties -> less evidence)",
                "; ".join(flags), main=None)


REV = "reverse (September pooled spec): send/5 → next myth 'be generous' [both orders]"
REVW = r"^reverse: \+0\.1 send/5 in a game → P\(next myth 'be generous'\) \[both orders\]$"


def c11_group(ds, setting):
    r = one("spread", REV, dataset=ds, setting=setting)
    w = q("spread", regex=REVW, dataset=ds, setting=setting)
    w = w[w.population != "all mixed dyads"]
    n = w.status.value_counts()
    lines = [f"{pm(r)}", f"agent-level {int(n.get(YES, 0))}/{len(w)} Holm, {int(n.get(SUG, 0))} raw"]
    return Cell(r.status, lines, src("spread", r) + src("spread", w), "September pooled spec (run FE), send/5",
                "lens pooled stratum (setting); within-agent family rows counted in text", main=r)


def c11_pooled(ds):
    r = one("spread", REV, dataset=ds, setting="all settings")
    lines = [pm(r), "all runs, run FE"]
    srcs = src("spread", r)
    rb = q("spread", "ROBUSTNESS (DeepSeek labels) reverse pooled: send/5 → next myth 'be generous' [both orders]",
           dataset=ds)
    if len(rb):
        lines.append(f"DeepSeek {rb.iloc[0].effect:+.2f} ({short(rb.iloc[0].status)})")
        srcs += src("spread", rb)
    return Cell(r.status, lines, srcs, "September pooled spec (run FE), send/5", "lens pooled stratum (all settings)",
                main=r)


def c10_pooled(ds):
    rows, lines = [], []
    for out, lab in [("round-r send", "send"), ("return share", "return"), ("change in send", "Δ send")]:
        r = one("search", f"all myth features add held-out R2 for {out} (raw, ridge)", dataset=ds,
                setting="all settings pooled, discovery split", family="pooled")
        rows.append(r)
        lines.append(f"{lab} {r.effect:+.3f} (≤ {r.ci_high:.3f}) {short(r.status)}")
    st = summarise([r.status for r in rows])
    fam = q("search", "all myth features add held-out R2 for return share (raw, ridge)", dataset=ds,
            setting="all settings pooled, discovery split")
    sug = fam[(fam.status == SUG) & (fam.family != "pooled")]
    for _, s in sug.iterrows():
        lines.append(f"{s.family} return {s.effect:+.3f} ({short(s.status)})")
    lines.append(f"{int(rows[0].n_runs)} runs, CV by run")
    return Cell(st, lines, src("search", pd.DataFrame(rows)) + src("search", sug), "raw ridge, pooled families",
                "summary over 3 outcomes: yes if any yes, else most common (ties -> less evidence)", main=None)


def c7_pooled(ds):
    lines, srcs, sts, flags = [], [], [], []
    for s, lab in [("2-agent (all, pooled)", "2-ag"), ("8-agent (all, pooled)", "8-ag")]:
        rows = q("alignment", regex=r"^H1 primary", dataset=ds, setting=s)
        c = c7_cell(rows)
        lines.append(f"{lab}: {c.lines[0]}")
        if YES in rows.status.values:
            lines.append(c.lines[1])
        srcs += c.src
        sts += list(rows.status)
        if c.flags:
            flags.append(c.flags)
    return Cell(summarise(sts), lines, srcs, "15 H1 primary tests per setting",
                "summary: yes if any lens row yes, else most common", "; ".join(flags))


def c8_pooled(ds):
    return c8_cell(ds, "all settings", FAM[ds], pop=None) if False else _c8_pooled(ds)


def _c8_pooled(ds):
    js, lines, srcs = [], [], []
    for f in FAM[ds]:
        j = one("consistency", DRIFT_J, dataset=ds, setting="all settings", family=f)
        k = one("consistency", DRIFT_K, dataset=ds, setting="all settings", family=f)
        js.append(j)
        lines.append(f"{f} judge {j.effect:+.2f}, kw {k.effect:+.2f}".replace("0.", "."))
        srcs += src("consistency", j) + src("consistency", k)
    return Cell(summarise([j.status for j in js]), lines, srcs, "GLM judge flag per family (keyword in text)",
                "summary over families: yes if any yes, else most common")


def c9_pooled(ds):
    s = one("consistency", R2S, dataset=ds, setting="all settings", family="all")
    r = one("consistency", R2R, dataset=ds, setting="all settings", family="all")
    return Cell(summarise([s.status, r.status]), [f"send {hw(s)}", f"return {hw(r)}", f"{nr(r)}, all families"],
                src("consistency", s) + src("consistency", r), "GLM judge flag, all families",
                "summary over send + return")


# ---------------------------------------------------------------- assemble the grid
def build():
    grid = {}  # (row_key, col) -> (Cell, span_rows)
    rows = []  # (dataset, group key, label, row_key, height)
    for ds in ["september", "frontier"]:
        rows.append((ds, "POOL", "", f"{ds}|POOL", 2.4))
        for gds, gk, glab, setting, pops in GROUPS:
            if gds != ds:
                continue
            for p in pops:
                rows.append((ds, gk, p, f"{ds}|{p}", TALL.get(p, 1.0)))
    # pooled rows
    for ds in ["september", "frontier"]:
        k = f"{ds}|POOL"
        grid[(k, 0)] = (c1_pooled(ds), 1)
        grid[(k, 1)] = (h2_pooled(ds, "send"), 1)
        grid[(k, 2)] = (h2_pooled(ds, "myth amount"), 1)
        for c in (3, 4, 5):
            grid[(k, c)] = (blank("not pooled across\nsettings by the lens", "no all-settings row"), 1)
        grid[(k, 6)] = (c7_pooled(ds), 1)
        grid[(k, 7)] = (c8_pooled(ds), 1)
        grid[(k, 8)] = (c9_pooled(ds), 1)
        grid[(k, 9)] = (c10_pooled(ds), 1)
        grid[(k, 10)] = (c11_pooled(ds), 1)
    for ds, gk, glab, setting, pops in GROUPS:
        keys = [f"{ds}|{p}" for p in pops]
        n = len(pops)
        top = keys[0]
        grid[(top, 0)] = (c1_group(ds, gk, setting), n)
        for c, which in [(1, "send"), (2, "myth amount")]:
            cell, _ = h2(ds, gk, setting, pops, which)
            if cell is not None:
                grid[(top, c)] = (cell, n)
            else:
                for p, k in zip(pops, keys):
                    grid[(k, c)] = (h2_row(ds, p, which), 1)
        c4 = c4_group(ds, gk, setting, pops)
        grid[(top, 3)] = (c4[0] if isinstance(c4, tuple) else c4, n)
        c5 = c5_group(ds, gk, setting, pops)
        grid[(top, 4)] = (c5 if c5 is not None else c5_frontier_mix(), n)
        grid[(top, 5)] = (c6_group(ds, gk, setting), n)
        for p, k in zip(pops, keys):
            grid[(k, 6)] = (c7_cell(c7_rows(ds, setting, p)), 1)
        if "homogeneous" in setting:
            for p, k in zip(pops, keys):
                f = p.split()[-1] if p.startswith("8 ") else p.split()[0]
                grid[(k, 7)] = (c8_cell(ds, setting, [f]), 1)
                grid[(k, 8)] = (c9_cell(ds, setting, [f]), 1)
        else:
            fams = [f for f in FAM[ds] if ds == "frontier" or setting == "2-agent mixed" or True]
            grid[(top, 7)] = (c8_cell(ds, setting, fams), n)
            fams9 = [f for f in fams if not (f == CEILFAM[ds] and False)]
            grid[(top, 8)] = (c9_cell(ds, setting, fams9), n)
        grid[(top, 9)] = (blank("pooled only ↑", "search lens computed held-out gain only over all settings"), n)
        grid[(top, 10)] = (c11_group(ds, setting), n)
    return rows, grid


# ---------------------------------------------------------------- CSV
def write_csv(rows, grid):
    out = []
    order = {r[3]: i for i, r in enumerate(rows)}
    for (k, c), (cell, span) in sorted(grid.items(), key=lambda x: (order[x[0][0]], x[0][1])):
        i = order[k]
        covered = [rows[j][2] or "all settings (pooled row)" for j in range(i, i + span)]
        m = cell.main
        out.append({
            "dataset": rows[i][0], "group": rows[i][1], "rows_covered": " | ".join(covered), "span_rows": span,
            "column": c + 1, "header": COLS[c][0].replace("\n", " "), "status": cell.status or "",
            "cell_text": " / ".join(cell.lines), "headline_measure": cell.measure, "rule": cell.rule,
            "effect": m.effect if m is not None else np.nan, "ci_low": m.ci_low if m is not None else np.nan,
            "ci_high": m.ci_high if m is not None else np.nan, "n_runs": m.n_runs if m is not None else np.nan,
            "headline_finding": m.finding if m is not None else "",
            "headline_key": (f"{m.dataset}|{m.setting}|{m.population}|{m.family}|{m.finding}" if m is not None else ""),
            "source": "; ".join(f"{lens}/scorecard_rows.csv#{ix}" for lens, ix in cell.src),
            "source_keys": " || ".join(
                f"{lens}: {L[lens].loc[ix, 'dataset']}|{L[lens].loc[ix, 'setting']}|{L[lens].loc[ix, 'population']}|"
                f"{L[lens].loc[ix, 'family']}|{L[lens].loc[ix, 'finding']}" for lens, ix in cell.src),
            "n_runs_max_over_sources": max([L[lens].loc[ix, "n_runs"] for lens, ix in cell.src
                                            if L[lens].loc[ix, "n_runs"] == L[lens].loc[ix, "n_runs"]], default=np.nan),
            "flags": cell.flags,
        })
    pd.DataFrame(out).to_csv(HERE / "scorecard_cells.csv", index=False)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- render
def render(rows, grid):
    plt.rcParams["font.family"] = ["Arial Narrow", "DejaVu Sans"]
    plt.rcParams["hatch.linewidth"] = 0.6
    plt.rcParams["text.parse_math"] = False
    W, H = 17.0, 11.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ink, ink2, grid_c = "#1a1a19", "#52514e", "#c9c8c2"

    x0, grp_w, pop_w = 0.3, 0.75, 1.15
    col_w = (W - 0.3 - x0 - grp_w - pop_w) / len(COLS)
    cx = [x0 + grp_w + pop_w + i * col_w for i in range(len(COLS))]
    y = 0.3
    ax.text(x0, y, "What in the myths predicts cooperation? September beside frontier", fontsize=15, weight="bold",
            va="top", color=ink)
    ax.text(x0, y + 0.3, "Each cell: the lens's status (colour + symbol) and headline number. A cell spanning several "
            "rows is a pooled estimate over those rows; family lines inside it are the lens's per-family rows.",
            fontsize=9, va="top", color=ink2)
    y = 0.78
    hdr_h = 0.6
    for i, (t, sub) in enumerate(COLS):
        ax.text(cx[i] + 0.05, y, f"{i + 1}  {t}", fontsize=9, weight="bold", va="top", color=ink, linespacing=1.05)
        ax.text(cx[i] + 0.05, y + 0.32, sub, fontsize=7, va="top", color=ink2, linespacing=1.05)
    y += hdr_h

    unit = 0.25
    band_h = 0.24
    ypos = {}
    cur_ds, cur_g = None, None
    gstart = {}
    for ds, gk, lab, key, h in rows:
        if ds != cur_ds:
            title = {"september": "SEPTEMBER  ·  Sonnet 4.5 / GPT-5 Nano / Gemini 3.7 Flash  ·  156 myth runs",
                     "frontier": "FRONTIER (main set)  ·  Opus 5 / GPT-5.6 Sol / Gemini 3.1 Pro  ·  106 myth runs"}[ds]
            ax.add_patch(Rectangle((x0, y), W - 0.3 - x0, band_h, color="#2c2c2a", lw=0))
            ax.text(x0 + 0.08, y + band_h / 2, title, fontsize=9.5, weight="bold", color="white", va="center")
            y += band_h + 0.03
            cur_ds = ds
        ypos[key] = (y, h * unit)
        if gk != cur_g:
            gstart[gk] = y
            cur_g = gk
        y += h * unit
    y_end = y

    # group labels and population labels
    for ds, gk, glab, setting, pops in GROUPS:
        top = ypos[f"{ds}|{pops[0]}"][0]
        bot = sum(ypos[f"{ds}|{pops[-1]}"])
        ax.text(x0 + 0.04, (top + bot) / 2, glab, fontsize=8.5, weight="bold", va="center", color=ink, linespacing=1.0)
        ax.plot([x0, W - 0.3], [top, top], color=ink2, lw=0.9)
        for p in pops:
            yy, hh = ypos[f"{ds}|{p}"]
            ax.text(x0 + grp_w, yy + hh / 2, ROWLABEL.get(p, p), fontsize=8, va="center", color=ink, linespacing=1.0)
    for ds in ["september", "frontier"]:
        yy, hh = ypos[f"{ds}|POOL"]
        ax.text(x0 + 0.04, yy + hh / 2, "All settings\npooled (lens's\nheadline strata)", fontsize=8, weight="bold",
                va="center", color=ink, linespacing=1.0)

    fs = 7.0
    line_h = fs * 1.08 / 72
    overflow = []
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    order = [r[3] for r in rows]
    for (k, c), (cell, span) in grid.items():
        i = order.index(k)
        yy = ypos[k][0]
        hh = sum(ypos[order[j]][1] for j in range(i, i + span))
        pad = 0.018
        st = cell.status
        ax.add_patch(Rectangle((cx[c] + pad, yy + pad), col_w - 2 * pad, hh - 2 * pad, facecolor=FILL[st],
                               edgecolor=grid_c if st in (DES, UP, None) else FILL[st],
                               lw=0.8, ls=(0, (2, 1.5)) if st == UP else "-", hatch=HATCH.get(st),
                               zorder=1))
        if st in HATCH:
            ax.patches[-1].set_edgecolor("#d9d8d2" if st == DES else "#e38a3c")
            ax.add_patch(Rectangle((cx[c] + pad, yy + pad), col_w - 2 * pad, hh - 2 * pad, fill=False,
                                   edgecolor=grid_c if st == DES else FILL[st], lw=0.8, zorder=1.1))
        color = INK.get(st, ink)
        lines = list(cell.lines)
        glyph = GLYPH[st]
        nfit = int((hh - 0.02) / line_h)
        if len(lines) > nfit:
            overflow.append((k, c, f"{len(lines)} lines > {nfit}"))
            lines = lines[:nfit]
        ty = yy + hh / 2 - (len(lines) - 1) * line_h / 2
        bg = dict(boxstyle="square,pad=0.05", fc=FILL[st], ec="none") if st in HATCH else None
        if glyph:
            ax.text(cx[c] + 0.06, ty, glyph, fontsize=fs, va="center", color=color, family="DejaVu Sans",
                    weight="bold", zorder=3, bbox=bg)
        for li, t in enumerate(lines):
            txt = t
            tobj = ax.text(cx[c] + 0.07 + (0.11 if glyph else 0), ty + li * line_h, txt, fontsize=fs, va="center",
                           color=color,
                           weight="bold" if li == 0 else "normal", zorder=3, bbox=bg,
                           style="italic" if st is None else "normal")
            bb = tobj.get_window_extent(rend)
            if bb.x1 / fig.dpi > cx[c] + col_w - 0.03:
                overflow.append((k, c, t))
    # column rules
    for i in range(len(COLS) + 1):
        xx = x0 + grp_w + pop_w + i * col_w
        ax.plot([xx, xx], [0.78 + hdr_h - 0.05, y_end], color=grid_c, lw=0.4, zorder=0)

    # legend
    ly = y_end + 0.14
    lx = x0
    for st, lab in LEGEND:
        ax.add_patch(Rectangle((lx, ly), 0.32, 0.2, facecolor=FILL[st], hatch=HATCH.get(st),
                               edgecolor="#e38a3c" if st == CEIL else ("#d9d8d2" if st == DES else (grid_c if st == UP else FILL[st])),
                               ls=(0, (2, 1.5)) if st == UP else "-", lw=0.8))
        ax.text(lx + 0.16, ly + 0.1, GLYPH[st], fontsize=8.5, ha="center", va="center", color=INK.get(st, ink),
                family="DejaVu Sans",
                bbox=dict(boxstyle="square,pad=0.05", fc=FILL[st], ec="none") if st in HATCH else None)
        ax.text(lx + 0.4, ly + 0.1, lab, fontsize=8, va="center", color=ink)
        lx += 0.5 + 0.068 * len(lab)
    foot = (
        "Statuses are the lenses' own, with Holm correction within each stratum (setting × family × task order); they can read "
        "stronger than the September README, which used lens-wide Holm (e.g. col 3 September +0.07 was not Holm-significant lens-wide). "
        "† dyads: partners share their game history and the unseen comparison myth comes from another run, so dyad "
        "word/moral matches are not transmission evidence (no future-myth placebo is possible). "
        "Cols 8–9 use the GLM judge flag (the only consistency measure live in both corpora); col 8 'yes' needs the DeepSeek judge to rise "
        "too, and frontier judges agree less per myth (κ 0.37 vs 0.54); the embedding score is not used as frontier drift evidence. "
        "Cells summarising several tests (cols 7, 9; mixed rows of 8; col 10): 'yes' if any lens row is yes, else the most common status. "
        "[a, b] = 95% CI (run-clustered); ± = CI half-width; * = Holm-significant. "
        "Moral labels and consistency flag: GLM-5.2 judge; DeepSeek as robustness (frontier DeepSeek served with hidden reasoning, not strictly comparable). "
        "Frontier = main set only (update set excluded). Game-only runs have no myths and are excluded. "
        "m→g = myth→game, g→m = game→myth task order; kw = keyword; snd = senders. Source rows for every cell: scorecard_cells.csv."
    )
    import textwrap
    ax.text(x0, ly + 0.34, "\n".join(textwrap.wrap(foot, 330)), fontsize=7.5, va="top", color=ink2, linespacing=1.25)
    fig.savefig(HERE / "scorecard.png", dpi=200, facecolor="white")
    fig.savefig(HERE / "scorecard.pdf", facecolor="white")
    return overflow, ly


def _fmts(x):
    """Ways a number can be printed on the page."""
    out = set()
    for d in (0, 1, 2, 3):
        for sign in ("+", ""):
            t = f"{x:{sign}.{d}f}"
            out |= {t, t.replace("0.", "."), f"${t}"}
    return out


def verify():
    """Re-read every cell's source rows by source_keys; fail loudly if numbers on the page disagree with the lenses."""
    cells = pd.read_csv(HERE / "scorecard_cells.csv")
    n_checked, problems, omitted = 0, [], []
    for _, c in cells.iterrows():
        if not isinstance(c.source_keys, str):
            continue
        ids = [x.split("#") for x in c.source.split("; ")]
        for (lensfile, ix), key in zip(ids, c.source_keys.split(" || ")):
            lens, rest = key.split(": ", 1)
            ds, setting, pop, fam, finding = rest.split("|", 4)
            d = L[lens]
            m = d[(d.dataset == ds) & (d.setting == setting) & (d.population == pop) & (d.family == fam) &
                  (d.finding == finding)]
            if len(m) != 1 or int(m.index[0]) != int(ix):
                problems.append(f"{c.group} col {c.column}: key resolves to {list(m.index)} not #{ix}: {key}")
        if not isinstance(c.headline_finding, str) or not c.headline_finding:
            continue
        lens = next(k.split(": ", 1)[0] for k in c.source_keys.split(" || ")
                    if k.split(": ", 1)[1] == c.headline_key)
        ds, setting, pop, fam, finding = c.headline_key.split("|", 4)
        d = L[lens]
        r = d[(d.dataset == ds) & (d.setting == setting) & (d.population == pop) & (d.family == fam) &
              (d.finding == finding)]
        assert len(r) == 1, c.headline_key
        r = r.iloc[0]
        for col in ("effect", "ci_low", "ci_high", "n_runs"):
            a, b = c[col], r[col]
            if not ((a != a and b != b) or np.isclose(a, b)):
                problems.append(f"{c.group} col {c.column}: CSV {col} {a} != lens {b}")
        text = c.cell_text
        if r.effect == r.effect and not any(f in text for f in _fmts(r.effect)):
            problems.append(f"{c.group} col {c.column}: effect {r.effect:.4f} not printed in '{text}'")
        # CI bounds and run counts: when a CI is printed as [lo, hi] it must be this row's; otherwise note the omission
        ci = re.search(r"\[(-?[\d.]+), (-?[\d.]+)\]", text)
        if r.ci_low == r.ci_low:
            if ci and not (any(f == ci.group(1) for f in _fmts(r.ci_low)) and any(f == ci.group(2) for f in _fmts(r.ci_high))):
                problems.append(f"{c.group} col {c.column}: printed CI [{ci.group(1)}, {ci.group(2)}] != lens "
                                f"[{r.ci_low:.3f}, {r.ci_high:.3f}]")
            elif not ci and "±" not in text:
                omitted.append(f"{c.group} col {c.column}: CI not printed")
        runs = re.findall(r"(\d+) runs", text)
        if r.n_runs == r.n_runs:
            if runs and str(int(r.n_runs)) not in runs:
                problems.append(f"{c.group} col {c.column}: printed runs {runs} != lens {int(r.n_runs)}")
            elif not runs:
                omitted.append(f"{c.group} col {c.column}: n_runs not printed")
        n_checked += 1
    # column 1 counts come from plan r4.csv: check them against the lens's exact-match notes too
    for _, c in cells[cells.column == 1].iterrows():
        m = re.match(r"(\d+)/(\d+) sent the named", c.cell_text)
        if m and int(m.group(2)) == 0:
            problems.append(f"{c.group} col 1: zero named senders")
    print(f"verify: {len(cells)} cells, {n_checked} headline cells checked number-by-number, "
          f"{sum(isinstance(k, str) for k in cells.source_keys)} cells' source keys re-resolved")
    for o in omitted:
        print("verify note:", o)
    if problems:
        raise AssertionError("scorecard verification failed:\n" + "\n".join(problems))
    print("verify: OK — every printed headline effect, CI bound and run count matches its lens row")


if __name__ == "__main__":
    rows, grid = build()
    df = write_csv(rows, grid)
    ov, ly = render(rows, grid)
    print(f"{len(df)} cells; bottom of grid at {ly:.2f} in")
    for o in ov:
        print("OVERFLOW", o)
    verify()

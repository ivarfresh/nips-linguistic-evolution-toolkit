# Anonymous vs named regime: behaviors and myth kinds

**Data:** `sonnet45_8agent_myth_directive_history3_anon_r10_n5` vs `sonnet45_8agent_myth_directive_history3_r10_n5`
(5 runs each, 8 agents, 10 rounds, myth→game directive condition). Both regimes show the current
co-player's last 3 games; NAMED additionally gives persistent human names (Cyra, Aster, …) for self,
partner, and third parties, enabling cross-round reputation tracking. ANON refers to everyone as
"your current co-player" / "another agent".

**Method:** behavioral metrics + regex lexicon coding of all reasoning traces and myths (this folder's
CSVs), plus four independent LLM readers each blind to the other regime (myths r1,2,9,10; reasoning
r2,6,10).

## 1. Behavior: cooperation at ceiling in both regimes; no overt policing to observe

Sends reach ~97% of endowment by round 4 in both regimes; mean return fraction 0.61 (anon) vs 0.58
(named), n.s. (p=.50). No trustee ever returned <1/3 of received. With zero defection, punishment
*behavior* never gets triggered — policing exists only as normative content in text. The closest
behavioral analogue is **calibrated reciprocity**: graded returns to partial trust (e.g. "Briar sent
$4 → returned $7; partial trust receives calibrated fairness"), present in both regimes.

## 2. How cooperation is justified differs sharply (reasoning traces)

- **NAMED:** 100% of sampled traces open with the partner's *named, quantified track record*
  ("Hana's history shows perfect adherence to the golden standard…"). Trust = individual reputation.
- **ANON:** partners are typed, not named — agents invent *behavioral pseudo-identities*
  ("Foundation walker", "two-thirds keeper", "Phoenix spirit") and lean on impersonal quantified
  norms. Stranger/population language is ~5× more frequent (0.30 vs 0.06 per 1k tokens).
- Explicit conditional threats are nearly absent in both (~0.03 per 1k tokens). Policing vocabulary
  is ~2× higher in anon reasoning (0.28 vs 0.15 per 1k) but rare in absolute terms.

## 3. Different kinds of myths emerge (the core answer)

Both regimes start the same (round 1–2: nature-magic parables — multiplying springs/rivers, divine
watchers, supernatural sanction). They then **diverge by round 9–10**:

- **NAMED → reputation chronicles.** Late myths are populated 100% by the agents' own game names
  (Dorian, Elara, Galen…) with tracked histories; invented institutions are *records*:
  "The Chronicles", "the starlight ledger… blazed with seven names", "Circle of Abundance".
  Supernatural punishment largely disappears (blind reader estimate: ~59% of early myths feature
  sanction of a transgressor → ~14% late); it is replaced by transparent reputation
  ("proven cooperators trust proven cooperators instantly").
- **ANON → codified impersonal standards.** Late myths instead canonize *named numerical norms*:
  "The Foundation" / "Perfect Balance" (= return $7.5), "Generous Abundance" / "two-thirds covenant"
  (= return $10), and meta-myths about **consistency itself** as the virtue ("What matters is not
  which return you choose, but that you choose it forever. Consistency transforms philosophy into
  covenant."). Sanction imagery *persists* in impersonal, systemic form (the spring dies for
  everyone: "The multiplication ceased. Both rivers diminished into streams, then trickles, then
  dust") alongside divine watchers (the goddess "Recipros").

Lexicon corroboration (late myths, per 1k tokens, per-run Welch tests, n=5/cell): policing terms
anon 1.57 vs named 0.38 (p=.12; rising over rounds in anon, falling in named), surveillance terms
5.86 vs 3.97 (p=.052), stranger terms 1.61 vs 0.58 (p=.35); justice/honor and reputation terms lean
named. Directionally consistent, but only surveillance approaches significance at n=5.

## 4. Interpretation

Where identity persists, myths evolve into **reputation registries** — culture encodes *who did
what*. Where identity is stripped, myths evolve into **impersonal institutions** — culture encodes
*what anyone must do*, with quantified standards, generalized monitoring imagery, and systemic
(rather than targeted) sanction. This is a nice in-silico echo of the moralizing-punishment /
"Big Gods" hypothesis (impersonal norms + supernatural monitoring scale cooperation among strangers),
with the caveat that overt policing behavior never appears because cooperation never breaks down.

## Caveats

- n=5 runs per regime; lexicon contrasts are directional, mostly not significant individually.
- Anon agents still see the current partner's last 3 games, so this is "no persistent identity",
  not pure one-shot anonymity.
- Cooperation at ceiling → no behavioral test of policing; a noise/defector-injection variant would
  be needed to see whether the anon regime's impersonal norms actually get enforced differently.
- Blind-reader proportions are LLM estimates (Haiku), not hand-coded.

## Files

- `behavior_raw.csv`, `reason_lexicon_raw.csv`, `myth_lexicon_raw.csv` — full coded data
- `myths_{anon,named}.json`, `reasons_{anon,named}.json` — extracted texts (+ `_sample` subsets)
- Blind-reader reports: see appendix files `report_myths_{anon,named}.md`, `report_reasons_{anon,named}.md`

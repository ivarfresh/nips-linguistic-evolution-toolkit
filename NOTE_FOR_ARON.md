# Note for Aron — RETRACTED: the "num_agents" result was a bug artifact

**Status: this branch's headline conclusion is withdrawn.** An earlier version
of this note claimed the 2-vs-8 noise-buffering gap was a pure population-size
effect ("num_agents is the whole story"). **That is wrong** — it was produced on
the broken dyad transfer path, and Aron has since fixed the bug.

## What was wrong

The 2-agent later-round transfer path generated and cached the receiver-visible
**noisy transfer before the sender had decided**, so it defaulted to ~$0 —
receivers were repeatedly told the sender sent about $0. That spurious $0 is a
sufficient explanation for the dyad "collapse" to ~$46 that this branch's whole
decomposition (fixed-8, nocoplayer-8, and the dyad flips C1–C4) was built on. So
those cells **cannot establish a population-size interaction.**

Fixed by Aron on `codex/fix-dyad-transfer-noise` (commit `92406b99`, guarded by
`tests/test_trust_game_noisy_transfer.py`). That branch is the canonical,
corrected line — prefer it over this one.

## The corrected picture (Aron, 2026-08-12)

On the fixed transfer path (small differences, low power so far): **game→myth
slightly outperforms game-only in the 8-agent version, but not in the 2-agent
version.** Aron also standardized private memory to a 3-round horizon for both
population sizes and applies the myth-link instruction exactly once. Figures:
shared deck slides 687–691. Corrected v2 has passed an n=1 smoke; a powered
confirmatory batch is still pending.

## What (if anything) to keep from this branch

- The `pairing_mode: "fixed"` idea — but Aron independently implemented a more
  complete version (with pairing/noise seeds and validation) on his branch, so
  this branch's version is redundant.
- Everything else here (the noise2i dyad triplet, the joint-balance figures, the
  decomposition cells and their data) ran on the buggy path and should be treated
  as **superseded**, not cited.

— Ivar (with Claude). Correcting the record after Aron's fix.

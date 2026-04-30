# V4 Next Runs

Prepared runs use the resumable runner and direct provider routing. Wait for stable
network before launching API-backed batches.

Common environment:

```bash
PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct
```

## 1. Matched No-Noise Baseline

Purpose: 10-turn, memory-3, direct-provider baseline matched to the v4 directional
noise runs.

Total jobs: 90.

```bash
PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py baseline_v4_mem3_direct --workers 2 --output-subdir v4_direct_provider_baseline
```

## 2. Neutral-Framing Pilot

Purpose: test whether replacing trust-game/investor/trustee wording with ROLE A /
ROLE B allocation wording shifts baseline behavior.

Total jobs: 18.

```bash
PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py neutral_framing_v4_pilot --workers 2 --output-subdir v4_direct_provider_neutral
```

## 3. Neutral-Framing Full Run

Run only if the pilot shows a meaningful prompt-framing shift.

Total jobs: 90.

```bash
PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py neutral_framing_v4_mem3 --workers 2 --output-subdir v4_direct_provider_neutral
```

## 4. Smaller Directional Noise

Purpose: test whether the k=5 directional-uniform effects survive with less
floor/ceiling clipping.

k=1 total jobs: 360.

```bash
PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py noise_directional_k1_mem3 --workers 2 --output-subdir v4_direct_provider_k1
```

k=2 total jobs: 360.

```bash
PYENV_VERSION=3.10.14 PYTHONPYCACHEPREFIX=/tmp/nlet-pycache LLM_PROVIDER=direct /opt/homebrew/bin/pyenv exec python scripts/run_noisy_missing.py noise_directional_k2_mem3 --workers 2 --output-subdir v4_direct_provider_k2
```


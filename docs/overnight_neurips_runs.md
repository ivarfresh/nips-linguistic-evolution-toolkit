# Overnight NeurIPS Run Queue

Prepared but not launched.

## Current Status

- `baseline_v4_mem3_direct`: 47/90 complete; all 45 GPT-5-Nano baseline runs are complete, plus 2 Claude runs; 43 Claude runs remain.
- `neutral_framing_v4_pilot`: 0/18 complete.
- `noise_directional_k1_mem3`: 0/360 complete; optional after mandatory runs.
- Gemini and k=2 are intentionally excluded from this queue.

Check exact counts:

```bash
python scripts/noisy_run_status.py baseline_v4_mem3_direct --output-subdir v4_direct_provider_baseline
python scripts/noisy_run_status.py neutral_framing_v4_pilot --output-subdir v4_direct_provider_neutral
python scripts/noisy_run_status.py noise_directional_k1_mem3 --output-subdir v4_direct_provider_k1
```

## Recommended Overnight Launch

Run this when the laptop is on stable power/network:

```bash
WORKERS=2 RUN_K1=0 ./scripts/launch_neurips_overnight_runs.sh
```

This resumes the matched baseline, runs the neutral-framing pilot, then regenerates draft tables/figures and renders manuscript HTML.

## Extended Launch

If there is enough time/API budget, append the k=1 sensitivity run:

```bash
WORKERS=2 RUN_K1=1 ./scripts/launch_neurips_overnight_runs.sh
```

The k=1 run is 360 jobs. It should be treated as a sensitivity/appendix result unless it materially changes the story.

## Notes

- The launcher sources `.env` if present and uses `LLM_PROVIDER=direct`.
- The launcher wraps itself in `caffeinate` on macOS by default; disable with `USE_CAFFEINATE=0`.
- The runner is resumable. If the laptop sleeps or the process is interrupted, rerun the same command and completed final JSONs will be skipped.
- Logs go to `/tmp/nlet-runs` unless `LOG_DIR` is set.
- HTML rendering runs with `HOME=/tmp` to avoid Quarto's macOS cache permission issue in this environment.
- PDF rendering is off by default because TinyTeX package installation may need network. Enable with `RENDER_PDF=1` after TeX is working.

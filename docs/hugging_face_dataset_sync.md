# Hugging Face dataset sync

Generated experiment artifacts can be backed up to a private Hugging Face dataset and shared without committing them to Git.

## One-time setup

1. Create a Hugging Face organization containing both collaborators. A private dataset owned by a personal account cannot be shared; an organization-owned private dataset can.
2. Create a private dataset in that organization and give each uploader their own fine-grained token. Use a write token scoped to this dataset for uploads and a read token for download-only access.
3. Authenticate locally without putting the token in this repository:

   ```bash
   hf auth login
   ```

4. Add these non-secret settings to the repository-root `.env`:

   ```dotenv
   HF_DATASET_AUTO_UPLOAD=1
   HF_DATASET_REPO=organization/dataset
   ```

Never commit a Hugging Face token or add it to `.env.example`.

## What is uploaded

The sync treats a run as complete only when its final full-state JSON exists and contains the expected simulation fields. For each completed `run.json`, it uploads the existing members of this family:

- `run.json`
- `run.results.json`
- `run.log`
- `run.transcript.pdf`

Checkpoints, error snapshots, malformed JSON, results without a matching final JSON, and other partial artifacts are excluded. Remote paths mirror the contents below `data/json/`.

The uploader also rejects symlinks and refuses to write unless Hugging Face reports that the target dataset is private. After deliberate publication, continued uploads require the explicit `HF_DATASET_ALLOW_PUBLIC_UPLOAD=1` override.

The automatic hook runs once, in the parent process, after workers finish in:

- `scripts/run_noisy_missing.py`
- `experiments/run_noisy_batch.py`
- `experiments/run_trust_game_batch.py`

An upload error is reported as a warning but does not change the experiment batch's exit status.

## Preview and backfill

Preview the exact completed-run manifest without contacting Hugging Face:

```bash
python3 scripts/hf_sync_completed_runs.py --dry-run
```

After authentication and configuration, backfill the same manifest:

```bash
python3 scripts/hf_sync_completed_runs.py
```

The backfill is append/update-only: it does not delete remote files.

## Downloading updates

Each collaborator authenticates with their own token, then downloads into a separate local directory:

```bash
hf auth login
hf download organization/dataset --repo-type dataset --local-dir nlet-hf-data
```

Re-running the download updates only changed files. The experiment files will be under `nlet-hf-data/` with the same layout they have below `data/json/` locally.

## Publication

Keep the dataset private during active experiments. Before making it public, add the final dataset card, authors, license, and paper citation. Generate a DOI only after the repository name, visibility, and contents are ready to be treated as a stable release.

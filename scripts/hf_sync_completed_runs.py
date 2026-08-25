#!/usr/bin/env python3
"""Upload artifacts belonging to completed simulation runs to Hugging Face.

Automatic uploads are deliberately opt-in. Batch runners call
``maybe_sync_completed_runs`` after their worker pools have closed; it does
nothing unless both ``HF_DATASET_AUTO_UPLOAD=1`` and
``HF_DATASET_REPO=owner/dataset`` are configured.

The standalone CLI can discover completed runs for a dry run or a historical
backfill. A JSON file counts as a completed simulation only when it is a full
``SimulationData`` state, rather than a results/checkpoint/error artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows only
    _fcntl = None

try:
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - POSIX only
    _msvcrt = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_JSON_ROOT = PROJECT_ROOT / "data" / "json"
AUTO_UPLOAD_ENV = "HF_DATASET_AUTO_UPLOAD"
REPO_ENV = "HF_DATASET_REPO"
ALLOW_PUBLIC_ENV = "HF_DATASET_ALLOW_PUBLIC_UPLOAD"
TRUE_VALUES = {"1", "true", "yes", "on"}
FULL_STATE_KEYS = {"agents", "conversation_history", "game_data", "task_order"}
NON_FINAL_SUFFIXES = (".results.json", ".checkpoint.json", ".error.json")
MAX_UPLOAD_ATTEMPTS = 3


def _load_repo_env() -> None:
    """Load optional repository-local settings without requiring dotenv here."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


_load_repo_env()


@dataclass(frozen=True)
class UploadPlan:
    """An exact, completed-run-only Hugging Face folder upload."""

    repo_id: str
    data_root: Path
    folder_path: Path
    path_in_repo: str | None
    final_paths: tuple[Path, ...]
    artifact_paths: tuple[Path, ...]
    allow_patterns: tuple[str, ...]

    @property
    def remote_paths(self) -> tuple[str, ...]:
        return tuple(
            path.relative_to(self.data_root).as_posix()
            for path in self.artifact_paths
        )


def _warn(message: str) -> None:
    print(f"WARNING: Hugging Face dataset sync: {message}", file=sys.stderr)


def auto_upload_enabled(environ: dict[str, str] | os._Environ[str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(AUTO_UPLOAD_ENV, "").strip().lower() in TRUE_VALUES


def _is_non_final_name(path: Path) -> bool:
    return path.name.endswith(NON_FINAL_SUFFIXES)


def _resolve_below_data_root(path: str | Path, data_root: Path) -> Path | None:
    resolved_root = data_root.resolve()
    resolved_path = Path(path)
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path
    resolved_path = resolved_path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved_path


def is_completed_final_json(path: str | Path, *, data_root: Path = DATA_JSON_ROOT) -> bool:
    """Return whether ``path`` is an atomic full-state simulation JSON."""
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    if source_path.is_symlink():
        return False

    candidate = _resolve_below_data_root(path, data_root)
    if candidate is None or not candidate.is_file():
        return False
    if candidate.suffix != ".json" or _is_non_final_name(candidate):
        return False

    try:
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False

    return isinstance(payload, dict) and FULL_STATE_KEYS.issubset(payload)


def discover_completed_final_jsons(
    paths: Iterable[str | Path],
    *,
    data_root: Path = DATA_JSON_ROOT,
) -> tuple[Path, ...]:
    """Discover completed full-state JSONs under files or directories."""
    candidates: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_dir():
            candidates.update(path.rglob("*.json"))
        elif path.is_file():
            candidates.add(path)

    return tuple(
        sorted(
            resolved
            for candidate in candidates
            if (resolved := _resolve_below_data_root(candidate, data_root)) is not None
            and is_completed_final_json(resolved, data_root=data_root)
        )
    )


def artifacts_for_completed_runs(
    final_paths: Iterable[str | Path],
    *,
    data_root: Path = DATA_JSON_ROOT,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return validated finals and their existing, final-gated companions."""
    finals = tuple(
        sorted(
            {
                resolved
                for raw_path in final_paths
                if (resolved := _resolve_below_data_root(raw_path, data_root)) is not None
                and is_completed_final_json(resolved, data_root=data_root)
            }
        )
    )

    artifacts: set[Path] = set()
    resolved_root = data_root.resolve()
    for final_path in finals:
        base_path = Path(str(final_path)[:-5])
        companions = (
            final_path,
            Path(f"{base_path}.results.json"),
            Path(f"{base_path}.log"),
            Path(f"{base_path}.transcript.pdf"),
        )
        for path in companions:
            if path.is_symlink():
                continue
            resolved = _resolve_below_data_root(path, resolved_root)
            if resolved is not None and resolved.is_file():
                artifacts.add(resolved)

    return finals, tuple(sorted(artifacts))


def _literal_glob(path: str) -> str:
    """Escape glob metacharacters so an allow-pattern names one exact file."""
    replacements = {"*": "[*]", "?": "[?]", "[": "[[]", "]": "[]]"}
    return "".join(replacements.get(character, character) for character in path)


def build_upload_plan(
    final_paths: Iterable[str | Path],
    *,
    repo_id: str,
    data_root: Path = DATA_JSON_ROOT,
) -> UploadPlan:
    """Build the smallest folder upload with an exact artifact allow-list."""
    repo_id = repo_id.strip()
    if not repo_id or "/" not in repo_id:
        raise ValueError("HF_DATASET_REPO must have the form owner/dataset")

    resolved_root = data_root.resolve()
    finals, artifacts = artifacts_for_completed_runs(
        final_paths,
        data_root=resolved_root,
    )

    if artifacts:
        common_parent = Path(
            os.path.commonpath([str(path.parent) for path in artifacts])
        )
        relative_scope = common_parent.relative_to(resolved_root)
        path_in_repo = None if relative_scope == Path(".") else relative_scope.as_posix()
        allow_patterns = tuple(
            _literal_glob(path.relative_to(common_parent).as_posix())
            for path in artifacts
        )
        folder_path = common_parent
    else:
        folder_path = resolved_root
        path_in_repo = None
        allow_patterns = ()

    return UploadPlan(
        repo_id=repo_id,
        data_root=resolved_root,
        folder_path=folder_path,
        path_in_repo=path_in_repo,
        final_paths=finals,
        artifact_paths=artifacts,
        allow_patterns=allow_patterns,
    )


def _default_lock_path(repo_id: str) -> Path:
    digest = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"nlet-hf-dataset-sync-{digest}.lock"


@contextmanager
def local_upload_lock(lock_path: Path) -> Iterator[None]:
    """Serialize Hub commits made by concurrent local batch processes."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        if _fcntl is not None:
            _fcntl.flock(lock_handle.fileno(), _fcntl.LOCK_EX)
        elif _msvcrt is not None:
            lock_handle.seek(0, os.SEEK_END)
            if lock_handle.tell() == 0:
                lock_handle.write(b"\0")
                lock_handle.flush()
            lock_handle.seek(0)
            while True:
                try:
                    _msvcrt.locking(lock_handle.fileno(), _msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
        else:  # pragma: no cover - supported Python platforms provide one
            raise RuntimeError("no supported local file-locking API is available")

        try:
            yield
        finally:
            if _fcntl is not None:
                _fcntl.flock(lock_handle.fileno(), _fcntl.LOCK_UN)
            else:
                lock_handle.seek(0)
                _msvcrt.locking(lock_handle.fileno(), _msvcrt.LK_UNLCK, 1)


def _http_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    direct_status = getattr(exc, "status_code", None)
    return direct_status if isinstance(direct_status, int) else None


def _upload_with_conflict_retry(
    api: Any,
    plan: UploadPlan,
    commit_message: str,
    *,
    allow_public: bool,
) -> None:
    """Retry only optimistic-lock conflicts, refreshing the remote head each time."""
    for attempt in range(MAX_UPLOAD_ATTEMPTS):
        repo_info = api.repo_info(repo_id=plan.repo_id, repo_type="dataset")
        if not getattr(repo_info, "private", False) and not allow_public:
            raise RuntimeError(
                f"refusing to upload to public dataset {plan.repo_id}; "
                f"keep it private or explicitly set {ALLOW_PUBLIC_ENV}=1"
            )
        parent_commit = getattr(repo_info, "sha", None)
        if not parent_commit:
            raise RuntimeError("Hugging Face dataset repo did not report a head commit")

        try:
            api.upload_folder(
                repo_id=plan.repo_id,
                repo_type="dataset",
                folder_path=str(plan.folder_path),
                path_in_repo=plan.path_in_repo,
                allow_patterns=list(plan.allow_patterns),
                commit_message=commit_message,
                parent_commit=parent_commit,
            )
            return
        except Exception as exc:
            is_conflict = _http_status_code(exc) in {409, 412}
            if not is_conflict or attempt == MAX_UPLOAD_ATTEMPTS - 1:
                raise


def sync_completed_runs(
    final_paths: Iterable[str | Path],
    *,
    repo_id: str,
    data_root: Path = DATA_JSON_ROOT,
    lock_path: Path | None = None,
    api: Any | None = None,
    label: str | None = None,
    allow_public: bool = False,
) -> UploadPlan:
    """Upload one exact plan, raising on dependency/authentication/Hub errors."""
    plan = build_upload_plan(
        final_paths,
        repo_id=repo_id,
        data_root=data_root,
    )
    if not plan.artifact_paths:
        return plan

    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed; install it or use the `hf` CLI"
            ) from exc
        api = HfApi()

    commit_label = f" for {label}" if label else ""
    commit_message = (
        f"Sync {len(plan.final_paths)} completed experiment run(s)"
        f"{commit_label}"
    )
    selected_lock_path = lock_path or _default_lock_path(plan.repo_id)
    with local_upload_lock(selected_lock_path):
        _upload_with_conflict_retry(
            api,
            plan,
            commit_message,
            allow_public=allow_public,
        )
    return plan


def maybe_sync_completed_runs(
    final_paths: Iterable[str | Path],
    *,
    label: str | None = None,
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> bool:
    """Best-effort opt-in hook for runners; never raises or changes run status."""
    env = os.environ if environ is None else environ
    if not auto_upload_enabled(env):
        return False

    repo_id = env.get(REPO_ENV, "").strip()
    if not repo_id:
        _warn(f"{AUTO_UPLOAD_ENV} is enabled but {REPO_ENV} is unset")
        return False

    try:
        plan = sync_completed_runs(
            final_paths,
            repo_id=repo_id,
            label=label,
            allow_public=(
                env.get(ALLOW_PUBLIC_ENV, "").strip().lower() in TRUE_VALUES
            ),
        )
    except Exception as exc:
        _warn(str(exc))
        return False

    if plan.artifact_paths:
        print(
            "Hugging Face dataset sync: "
            f"uploaded {len(plan.artifact_paths)} artifact(s) from "
            f"{len(plan.final_paths)} completed run(s) to {repo_id}"
        )
    else:
        print("Hugging Face dataset sync: no completed run artifacts to upload")
    return True


def _print_dry_run(plan: UploadPlan) -> None:
    print(f"repo={plan.repo_id}")
    print(f"completed_runs={len(plan.final_paths)} artifacts={len(plan.artifact_paths)}")
    for path in plan.remote_paths:
        print(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run or backfill completed simulation artifacts to Hugging Face."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Final JSON files or directories to scan (default: data/json)",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get(REPO_ENV, ""),
        help=f"Dataset repo (default: ${REPO_ENV})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact remote artifact paths without importing or contacting Hugging Face",
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=None,
        help="Optional local upload lock path",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_JSON_ROOT,
        help="Local data/json root (mainly useful for testing or alternate clones)",
    )
    args = parser.parse_args(argv)

    scan_paths = args.paths or [args.data_root]
    finals = discover_completed_final_jsons(scan_paths, data_root=args.data_root)
    repo_id = args.repo_id.strip() or "dry-run/unspecified-dataset"

    try:
        plan = build_upload_plan(
            finals,
            repo_id=repo_id,
            data_root=args.data_root,
        )
    except ValueError as exc:
        _warn(str(exc))
        return 2

    if args.dry_run:
        _print_dry_run(plan)
        return 0

    if not auto_upload_enabled():
        print(
            f"Hugging Face dataset sync disabled; set {AUTO_UPLOAD_ENV}=1 to upload."
        )
        return 0
    if not args.repo_id.strip():
        _warn(f"{REPO_ENV} is unset")
        return 2

    try:
        plan = sync_completed_runs(
            finals,
            repo_id=args.repo_id,
            data_root=args.data_root,
            lock_path=args.lock_file,
            label="backfill",
            allow_public=(
                os.environ.get(ALLOW_PUBLIC_ENV, "").strip().lower()
                in TRUE_VALUES
            ),
        )
    except Exception as exc:
        _warn(str(exc))
        return 1

    print(
        f"Uploaded {len(plan.artifact_paths)} artifact(s) from "
        f"{len(plan.final_paths)} completed run(s) to {plan.repo_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

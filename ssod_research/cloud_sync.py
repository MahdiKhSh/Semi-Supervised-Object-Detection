"""Cross-environment persistence, GPU transfer validation, and safe GitHub sync.

Designed for seamless multi-machine workflows such as:
  Kaggle (T4 / P100) -> Linux L40s -> Windows RTX 5090

Guarantees:
- Detects ephemeral cloud runtimes (Kaggle, Colab) and warns loudly when no
  persistent storage target is configured.
- Snapshots SQLite safely via ``sqlite3.Connection.backup()`` (never raw-copies
  a live WAL database).
- Copies checkpoints atomically and verifies SHA-256 sidecars.
- Prevents two machines from writing to the same persistent experiment store
  concurrently via a heartbeat lease file.
- Validates checkpoint compatibility before cross-GPU transfers.
- Pushes ONLY lightweight results/progress/figures to GitHub (never weights or
  datasets) and degrades gracefully on network interruptions.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import tarfile
import time
from typing import Any

from ssod_research.recovery import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_copy,
    atomic_write_json,
    host_id,
    read_json,
    seconds_since,
    sha256_file,
    utc_now,
    verify_file,
)


def detect_runtime_environment(output_dir: str | Path | None = None) -> dict[str, Any]:
    """Identify local vs cloud environment and whether storage is ephemeral."""
    env = "local"
    is_cloud = False
    is_ephemeral = False
    warning = None
    suggested_sync_dir = None

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        env = "kaggle"
        is_cloud = True
        is_ephemeral = True
        warning = (
            "Kaggle notebook storage (/kaggle/working) is ephemeral after session termination "
            "unless saved as a Kaggle Output/Dataset or synced to external persistent storage "
            "(--sync-dir) and lightweight results pushed via --sync-github."
        )
        suggested_sync_dir = "/kaggle/working/ssod_persistent_export"
    elif os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") or Path("/content").is_dir():
        env = "colab"
        is_cloud = True
        drive_root = Path("/content/drive/MyDrive")
        out_str = str(Path(output_dir).resolve()) if output_dir else ""
        if drive_root.is_dir() and out_str.startswith(str(drive_root)):
            is_ephemeral = False
        else:
            is_ephemeral = True
            suggested_sync_dir = "/content/drive/MyDrive/ssod_research_state"
            warning = (
                "Google Colab local disk (/content) is wiped on runtime disconnect. "
                "Mount Google Drive and pass --sync-dir /content/drive/MyDrive/ssod_research_state "
                "(or set --output inside Drive) before long runs."
            )
    elif platform.system() == "Windows":
        env = "windows_local"
    else:
        env = "linux_server"

    return {
        "environment": env,
        "is_cloud": is_cloud,
        "is_ephemeral": is_ephemeral,
        "warning": warning,
        "suggested_sync_dir": suggested_sync_dir,
        "host": host_id(),
        "os": platform.platform(),
    }


def backup_sqlite(src_db: str | Path, dst_db: str | Path) -> Path:
    """Create a crash-consistent snapshot of a live WAL SQLite database."""
    src_db = Path(src_db)
    dst_db = Path(dst_db)
    dst_db.parent.mkdir(parents=True, exist_ok=True)
    tmp_db = dst_db.with_name(f".{dst_db.name}.{os.getpid()}.tmp")
    with contextlib.suppress(OSError):
        if tmp_db.exists():
            tmp_db.unlink()
    src_conn = sqlite3.connect(str(src_db), timeout=30.0)
    try:
        dst_conn = sqlite3.connect(str(tmp_db), timeout=30.0)
        try:
            src_conn.backup(dst_conn)
            dst_conn.commit()
        finally:
            dst_conn.close()
    finally:
        src_conn.close()
    from ssod_research.recovery import replace_file
    replace_file(tmp_db, dst_db)
    return dst_db


class MachineLease:
    """Advisory cross-machine lease to prevent two hosts mutating one sync store."""

    def __init__(self, directory: str | Path, *, ttl_sec: float = 1800.0, preset: str = ""):
        self.directory = Path(directory)
        self.lease_path = self.directory / ".active_machine.json"
        self.ttl_sec = float(ttl_sec)
        self.preset = preset
        self.me = host_id()
        self.pid = os.getpid()
        self._acquired = False

    def inspect(self) -> dict[str, Any] | None:
        return read_json(self.lease_path)

    def acquire(self, *, gpu_name: str | None = None, force: bool = False) -> dict[str, Any]:
        self.directory.mkdir(parents=True, exist_ok=True)
        existing = self.inspect()
        if existing and not force:
            other_host = existing.get("host")
            other_pid = existing.get("pid")
            age = seconds_since(existing.get("heartbeat_utc") or existing.get("acquired_utc"))
            same_process = (other_host == self.me and int(other_pid or -1) == self.pid)
            if not same_process and other_host != self.me:
                if age is None or age < self.ttl_sec:
                    raise RuntimeError(
                        f"Persistent store {self.directory} is actively leased by host={other_host} "
                        f"(gpu={existing.get('gpu')}, heartbeat {round(age or 0, 1)}s ago). "
                        f"Stop that session first or pass --force-lease after verifying it is dead."
                    )
        payload = {
            "host": self.me,
            "pid": self.pid,
            "preset": self.preset,
            "gpu": gpu_name,
            "environment": detect_runtime_environment(self.directory)["environment"],
            "acquired_utc": utc_now(),
            "heartbeat_utc": utc_now(),
        }
        atomic_write_json(self.lease_path, payload)
        self._acquired = True
        return payload

    def heartbeat(self, gpu_name: str | None = None) -> None:
        if not self._acquired:
            return
        cur = self.inspect() or {}
        cur.update({
            "host": self.me,
            "pid": self.pid,
            "gpu": gpu_name or cur.get("gpu"),
            "heartbeat_utc": utc_now(),
        })
        atomic_write_json(self.lease_path, cur)

    def release(self) -> None:
        if not self._acquired:
            return
        self._acquired = False
        cur = self.inspect()
        if cur and cur.get("host") == self.me and int(cur.get("pid") or -1) == self.pid:
            with contextlib.suppress(OSError):
                self.lease_path.unlink()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def validate_transfer_compatibility(
    checkpoint_sidecar: dict[str, Any],
    *,
    expected_fingerprint: str | None = None,
    expected_dataset_fingerprint: str | None = None,
    target_device_type: str = "cpu",
    target_amp: str = "off",
    target_gpu_name: str | None = None,
) -> dict[str, Any]:
    """Validate checkpoint compatibility before transferring across GPU hosts."""
    errors: list[str] = []
    warnings: list[str] = []

    fmt = int(checkpoint_sidecar.get("format_version") or 1)
    if fmt > CHECKPOINT_FORMAT_VERSION:
        errors.append(f"Unsupported checkpoint format_version={fmt} > {CHECKPOINT_FORMAT_VERSION}")

    fp = checkpoint_sidecar.get("fingerprint")
    if expected_fingerprint and fp and fp != expected_fingerprint:
        errors.append(f"Config/dataset fingerprint mismatch ({str(fp)[:12]} != {expected_fingerprint[:12]})")

    dfp = checkpoint_sidecar.get("dataset_fingerprint")
    if expected_dataset_fingerprint and dfp and dfp != expected_dataset_fingerprint:
        errors.append(f"Dataset fingerprint mismatch ({str(dfp)[:12]} != {expected_dataset_fingerprint[:12]})")

    src_hw = checkpoint_sidecar.get("hardware") or {}
    if src_hw:
        if src_hw.get("device_type") != target_device_type or src_hw.get("amp") != target_amp:
            warnings.append(
                f"Cross-device/precision transfer: {src_hw.get('device_type')}/{src_hw.get('amp')} "
                f"({src_hw.get('gpu_name')}) -> {target_device_type}/{target_amp} ({target_gpu_name}). "
                f"Weights, optimizer and scheduler resume cleanly; RNG continuation is approximate."
            )
        elif src_hw.get("gpu_name") != target_gpu_name:
            warnings.append(
                f"Cross-GPU transfer: {src_hw.get('gpu_name')} -> {target_gpu_name}. "
                f"Resume mode will be marked approximate due to differing CUDA architectures."
            )
    else:
        warnings.append("Source checkpoint lacks hardware profile metadata; resume will be approximate.")

    return {
        "compatible": len(errors) == 0,
        "resume_mode": "exact" if (not errors and not warnings) else ("approximate" if not errors else "incompatible"),
        "errors": errors,
        "warnings": warnings,
    }


def _should_sync_file(rel: Path) -> bool:
    name = rel.name
    if name.startswith(".") and name != ".active_machine.json":
        return False
    if name.endswith((".tmp", ".lock", ".info", "-wal", "-shm", "-journal")):
        return False
    if ".corrupt-" in name:
        return False
    return True


def sync_directory(
    src_dir: str | Path,
    dst_dir: str | Path,
    *,
    verify_sidecars: bool = True,
) -> dict[str, Any]:
    """Atomically sync experiment state from ``src_dir`` to ``dst_dir``."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    errors: list[str] = []

    if not src_dir.is_dir():
        return {"copied": 0, "skipped": 0, "errors": [f"source missing: {src_dir}"]}

    # 1. SQLite files first via crash-consistent backup API.
    for db_file in sorted(src_dir.rglob("*.sqlite")):
        rel = db_file.relative_to(src_dir)
        if not _should_sync_file(rel):
            continue
        try:
            backup_sqlite(db_file, dst_dir / rel)
            copied += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"sqlite backup failed for {rel}: {exc}")

    # 2. All other non-temp files (checkpoints copied BEFORE their sidecars so
    #    destination never sees a sidecar pointing to a missing checkpoint).
    all_files = []
    for item in src_dir.rglob("*"):
        if not item.is_file():
            continue
        rel = item.relative_to(src_dir)
        if item.suffix == ".sqlite" or not _should_sync_file(rel):
            continue
        all_files.append((item, rel))

    # Copy .pth before .json sidecars.
    all_files.sort(key=lambda pair: (0 if pair[0].suffix == ".pth" else 1, str(pair[1])))

    for item, rel in all_files:
        if verify_sidecars and item.suffix == ".pth":
            side = read_json(item.with_suffix(".json"))
            if side is not None:
                problem = verify_file(item, side, deep=False)
                if problem:
                    errors.append(f"skipping invalid checkpoint {rel}: {problem}")
                    continue
        target = dst_dir / rel
        if target.is_file():
            try:
                s_stat, t_stat = item.stat(), target.stat()
                if s_stat.st_size == t_stat.st_size and abs(s_stat.st_mtime - t_stat.st_mtime) < 1e-3:
                    skipped += 1
                    continue
            except OSError:
                pass
        try:
            atomic_copy(item, target)
            copied += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"copy failed for {rel}: {exc}")

    return {"copied": copied, "skipped": skipped, "errors": errors, "utc": utc_now()}


def _rewrite_registry_paths(db_path: Path, target_output_dir: Path) -> int:
    """Rewrite machine-local output/checkpoint paths inside an imported SQLite DB."""
    if not db_path.is_file():
        return 0
    target_output_dir = Path(target_output_dir).resolve()
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    updated = 0
    try:
        rows = conn.execute("SELECT experiment_id, stage_name, output_dir, checkpoint_path FROM experiments").fetchall()
        for row in rows:
            stage_name = row["stage_name"]
            new_out = None
            new_ckpt = None
            if "/" in stage_name:
                parts = stage_name.split("/")
                if len(parts) == 2 and parts[1] == "prepare":
                    new_out = str(target_output_dir / "splits" / parts[0])
                elif len(parts) == 2:
                    new_out = str(target_output_dir / "runs" / parts[0] / parts[1])
                elif len(parts) == 3 and parts[2] in ("eval", "analysis"):
                    new_out = str(target_output_dir / "runs" / parts[0] / parts[1])
            elif row["output_dir"]:
                new_out = str(target_output_dir)

            old_ckpt = row["checkpoint_path"]
            if old_ckpt and new_out:
                ckpt_name = Path(old_ckpt).name
                candidate = Path(new_out) / "checkpoints" / ckpt_name
                if candidate.is_file():
                    new_ckpt = str(candidate)
                elif (Path(new_out) / ckpt_name).is_file():
                    new_ckpt = str(Path(new_out) / ckpt_name)

            if new_out or new_ckpt:
                conn.execute(
                    """
                    UPDATE experiments
                    SET output_dir = COALESCE(?, output_dir),
                        checkpoint_path = COALESCE(?, checkpoint_path)
                    WHERE experiment_id = ?
                    """,
                    (new_out, new_ckpt, row["experiment_id"]),
                )
                updated += 1
        conn.commit()
    finally:
        conn.close()
    return updated


def export_portable_bundle(
    output_dir: str | Path,
    bundle_path: str | Path,
    *,
    create_archive: bool = False,
) -> dict[str, Any]:
    """Export an experiment work_dir into a SHA-256 verified portable bundle.

    Use this to move work from Kaggle -> L40s -> RTX 5090 (or vice versa).
    """
    output_dir = Path(output_dir).resolve()
    bundle_path = Path(bundle_path).resolve()
    staging = bundle_path.with_suffix("") if create_archive else bundle_path
    staging.mkdir(parents=True, exist_ok=True)

    sync_res = sync_directory(output_dir, staging, verify_sidecars=True)
    files_manifest = {}
    checkpoints_summary = []
    for item in sorted(staging.rglob("*")):
        if not item.is_file() or item.name == "bundle_manifest.json":
            continue
        rel = str(item.relative_to(staging)).replace("\\", "/")
        files_manifest[rel] = {
            "size": item.stat().st_size,
            "sha256": sha256_file(item, use_cache=False),
        }
        if item.suffix == ".json" and item.stem.startswith("ckpt_"):
            side = read_json(item)
            if side:
                checkpoints_summary.append({"rel": rel, **side})

    manifest = {
        "bundle_version": 1,
        "created_utc": utc_now(),
        "source_host": host_id(),
        "source_environment": detect_runtime_environment(output_dir)["environment"],
        "file_count": len(files_manifest),
        "files": files_manifest,
        "checkpoints": checkpoints_summary,
        "sync_errors": sync_res["errors"],
    }
    atomic_write_json(staging / "bundle_manifest.json", manifest)

    if create_archive:
        archive = bundle_path if str(bundle_path).endswith(".tar.gz") else Path(str(bundle_path) + ".tar.gz")
        tmp_archive = archive.with_name(f".{archive.name}.tmp")
        with tarfile.open(tmp_archive, "w:gz") as tar:
            tar.add(staging, arcname=".")
        from ssod_research.recovery import replace_file
        replace_file(tmp_archive, archive)
        manifest["archive_path"] = str(archive)
        manifest["archive_sha256"] = sha256_file(archive, use_cache=False)

    return manifest


def import_portable_bundle(
    bundle_path: str | Path,
    target_output_dir: str | Path,
    *,
    target_device_type: str = "cpu",
    target_amp: str = "off",
    target_gpu_name: str | None = None,
) -> dict[str, Any]:
    """Verify checksums & compatibility, then import a bundle onto a new machine."""
    bundle_path = Path(bundle_path).resolve()
    target_output_dir = Path(target_output_dir).resolve()
    target_output_dir.mkdir(parents=True, exist_ok=True)

    extracted_tmp = None
    source_dir = bundle_path
    if bundle_path.is_file() and bundle_path.name.endswith(".tar.gz"):
        extracted_tmp = target_output_dir / f".bundle_extract_{os.getpid()}"
        extracted_tmp.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle_path, "r:gz") as tar:
            # Safe extraction: refuse path traversal entries.
            for member in tar.getmembers():
                member_path = (extracted_tmp / member.name).resolve()
                if not str(member_path).startswith(str(extracted_tmp.resolve())):
                    raise ValueError(f"Unsafe tar member path: {member.name}")
            tar.extractall(extracted_tmp)
        source_dir = extracted_tmp

    try:
        manifest = read_json(source_dir / "bundle_manifest.json")
        if not manifest or "files" not in manifest:
            raise ValueError(f"Missing or invalid bundle_manifest.json in {source_dir}")

        # Verify every file's SHA-256 before copying anything.
        for rel, meta in manifest["files"].items():
            src_file = source_dir / rel
            if not src_file.is_file():
                raise FileNotFoundError(f"Bundle missing file: {rel}")
            if sha256_file(src_file, use_cache=False) != meta["sha256"]:
                raise ValueError(f"Bundle file corrupted (sha256 mismatch): {rel}")

        # Validate cross-GPU checkpoint compatibility.
        compat_reports = []
        for ckpt in manifest.get("checkpoints", []):
            rep = validate_transfer_compatibility(
                ckpt,
                target_device_type=target_device_type,
                target_amp=target_amp,
                target_gpu_name=target_gpu_name,
            )
            compat_reports.append({"checkpoint": ckpt.get("rel"), **rep})
            if not rep["compatible"]:
                raise RuntimeError(f"Incompatible checkpoint {ckpt.get('rel')}: {rep['errors']}")

        # Copy verified files into target_output_dir atomically.
        restored = 0
        for rel in sorted(manifest["files"]):
            atomic_copy(source_dir / rel, target_output_dir / rel)
            restored += 1

        # Rewrite paths in imported SQLite DBs so resume works seamlessly on the new OS/machine.
        rewritten = 0
        for db_file in target_output_dir.rglob("*.sqlite"):
            rewritten += _rewrite_registry_paths(db_file, target_output_dir)

        return {
            "restored_files": restored,
            "rewritten_registry_rows": rewritten,
            "source_host": manifest.get("source_host"),
            "source_environment": manifest.get("source_environment"),
            "compatibility": compat_reports,
        }
    finally:
        if extracted_tmp and extracted_tmp.exists():
            import shutil
            shutil.rmtree(extracted_tmp, ignore_errors=True)


def snapshot_lightweight_results(
    output_dir: str | Path,
    repo_root: str | Path,
    preset: str,
    *,
    max_figure_bytes: int = 5 * 1024 * 1024,
) -> list[Path]:
    """Copy ONLY lightweight results/metrics/progress/figures into ``results/<preset>/``.

    Never copies checkpoints (.pth), raw images, raw labels, or SQLite DBs.
    """
    output_dir = Path(output_dir)
    repo_root = Path(repo_root)
    dest = repo_root / "results" / preset
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []

    for filename in (
        "results.csv",
        "results.json",
        "statistics.json",
        "PAPER_RESULTS.md",
        "progress.json",
        "runtime_estimates.json",
        "PROJECT_PROGRESS.md",
        "experiment_plan.json",
    ):
        src = output_dir / filename
        if src.is_file():
            target = dest / filename
            atomic_copy(src, target)
            copied.append(target)

    # Copy lightweight per-class and condition JSON summaries (no large detections.json).
    runs_dir = output_dir / "runs"
    if runs_dir.is_dir():
        for name in ("per_class.json", "conditions.json"):
            for src in runs_dir.rglob(f"analysis/{name}"):
                rel = src.relative_to(runs_dir)
                target = dest / "analysis" / rel
                atomic_copy(src, target)
                copied.append(target)

    # Copy figures under size guard.
    fig_dir = output_dir / "figures"
    if fig_dir.is_dir():
        for fig in sorted(fig_dir.glob("*.png")):
            if fig.stat().st_size <= max_figure_bytes:
                target = dest / "figures" / fig.name
                atomic_copy(fig, target)
                copied.append(target)

    return copied


def sync_results_to_github(
    repo_root: str | Path,
    *,
    preset: str,
    output_dir: str | Path | None = None,
    message: str | None = None,
    push: bool = True,
    retries: int = 3,
    branch: str | None = None,
) -> dict[str, Any]:
    """Commit and push lightweight results & progress docs safely (non-fatal on network errors)."""
    repo_root = Path(repo_root).resolve()
    if output_dir is not None:
        snapshot_lightweight_results(output_dir, repo_root, preset)

    def _git(*cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *cmd],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf8",
            errors="replace",
        )

    check = _git("rev-parse", "--is-inside-work-tree")
    if check.returncode != 0:
        return {"synced": False, "reason": "not a git repository"}

    paths_to_add = []
    for candidate in (
        Path("docs/PROJECT_PROGRESS.md"),
        Path("docs/CODEX_HANDOFF.md"),
        Path("docs/DECISIONS.md"),
        Path("results") / preset,
    ):
        if (repo_root / candidate).exists():
            paths_to_add.append(str(candidate))
    if not paths_to_add:
        return {"synced": False, "reason": "no lightweight result files to stage"}

    add = _git("add", "--", *paths_to_add)
    if add.returncode != 0:
        return {"synced": False, "reason": f"git add failed: {add.stderr.strip()}"}

    diff = _git("diff", "--cached", "--quiet")
    if diff.returncode == 0:
        return {"synced": True, "committed": False, "pushed": False, "reason": "no new result changes"}

    commit_msg = message or f"results({preset}): update progress and measured metrics [{host_id()}]"
    commit = _git("commit", "-m", commit_msg)
    if commit.returncode != 0:
        return {"synced": False, "reason": f"git commit failed: {commit.stderr.strip()}"}

    if not push:
        return {"synced": True, "committed": True, "pushed": False, "reason": "committed locally"}

    cur_branch = branch
    if not cur_branch:
        br = _git("rev-parse", "--abbrev-ref", "HEAD")
        cur_branch = br.stdout.strip() if br.returncode == 0 else "HEAD"

    last_err = ""
    for attempt in range(1, max(1, retries) + 1):
        res = _git("push", "origin", cur_branch)
        if res.returncode == 0:
            return {"synced": True, "committed": True, "pushed": True, "branch": cur_branch, "attempts": attempt}
        last_err = res.stderr.strip()
        if attempt < retries:
            time.sleep(min(15, 2 ** attempt))

    return {
        "synced": False,
        "committed": True,
        "pushed": False,
        "branch": cur_branch,
        "reason": f"push failed after {retries} attempts (training unaffected): {last_err[:300]}",
    }

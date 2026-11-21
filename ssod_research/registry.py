"""Transactional SQLite experiment registry for crash-safe SSOD execution.

Tracks every pipeline stage with deterministic IDs, configuration/dataset/split
fingerprints, status transitions, checkpoint locations, artifacts, accumulated
runtime, live heartbeats, and retry history.

All writes run inside IMMEDIATE transactions with WAL journaling and FULL
synchronization so power loss or process death cannot corrupt the registry.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

from ssod_research.recovery import (
    host_id,
    is_locked,
    pid_alive,
    seconds_since,
    stable_hash,
    utc_now,
)

VALID_STATUSES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "INTERRUPTED")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    suite_preset TEXT NOT NULL,
    stage_name TEXT NOT NULL,
    stage_type TEXT NOT NULL,
    stage_order INTEGER NOT NULL DEFAULT 0,
    dependencies_json TEXT NOT NULL DEFAULT '[]',
    config_json TEXT NOT NULL DEFAULT '{}',
    config_fingerprint TEXT NOT NULL DEFAULT '',
    dataset_fingerprint TEXT NOT NULL DEFAULT '',
    split_fingerprint TEXT NOT NULL DEFAULT '',
    upstream_fingerprint TEXT NOT NULL DEFAULT '',
    model_variant TEXT,
    seed INTEGER,
    label_fraction REAL,
    threshold REAL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    target_steps INTEGER NOT NULL DEFAULT 0,
    last_completed_step INTEGER NOT NULL DEFAULT 0,
    checkpoint_path TEXT,
    checkpoint_sha256 TEXT,
    resume_mode TEXT DEFAULT 'fresh',
    output_dir TEXT,
    artifacts_json TEXT NOT NULL DEFAULT '{}',
    start_time TEXT,
    last_update_time TEXT NOT NULL,
    completed_time TEXT,
    accumulated_runtime_sec REAL NOT NULL DEFAULT 0.0,
    pid INTEGER,
    host TEXT,
    gpu_name TEXT,
    heartbeat_utc TEXT,
    error_info TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    retry_history_json TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_experiments_preset_stage
    ON experiments(suite_preset, stage_name);

CREATE INDEX IF NOT EXISTS idx_experiments_status
    ON experiments(status);

CREATE TABLE IF NOT EXISTS suite_meta (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmarks (
    benchmark_key TEXT PRIMARY KEY,
    gpu_name TEXT,
    device_type TEXT NOT NULL,
    amp TEXT NOT NULL,
    max_side INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    sec_per_step REAL NOT NULL,
    peak_vram_gib REAL,
    measured_steps INTEGER NOT NULL,
    updated_utc TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}'
);
"""


def make_experiment_id(preset: str, stage_name: str) -> str:
    """Deterministic human-readable + collision-safe stage identifier."""
    slug = stage_name.replace("/", "__").replace(" ", "_")
    return f"{preset}__{slug}"


class ExperimentRegistry:
    """Persistent SQLite-backed registry for all research pipeline stages."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _tx(self):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(SCHEMA_SQL)
        finally:
            conn.close()

    @staticmethod
    def _decode_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        for src, dst in (
            ("dependencies_json", "dependencies"),
            ("config_json", "config"),
            ("artifacts_json", "artifacts"),
            ("retry_history_json", "retry_history"),
        ):
            raw = d.get(src)
            try:
                d[dst] = json.loads(raw) if raw else ([] if src.endswith(("dependencies_json", "history_json")) else {})
            except ValueError:
                d[dst] = [] if src.endswith(("dependencies_json", "history_json")) else {}
        return d

    # ------------------------------------------------------------------ CRUD
    def upsert_stage(
        self,
        *,
        experiment_id: str,
        suite_preset: str,
        stage_name: str,
        stage_type: str,
        stage_order: int = 0,
        dependencies: list[str] | None = None,
        config: dict[str, Any] | None = None,
        config_fingerprint: str = "",
        dataset_fingerprint: str = "",
        split_fingerprint: str = "",
        upstream_fingerprint: str = "",
        model_variant: str | None = None,
        seed: int | None = None,
        label_fraction: float | None = None,
        threshold: float | None = None,
        target_steps: int = 0,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Insert a stage or update its planned metadata without wiping progress."""
        now = utc_now()
        deps_json = json.dumps(dependencies or [])
        cfg_json = json.dumps(config or {}, sort_keys=True, default=str)
        out_str = str(output_dir) if output_dir is not None else None
        with self._tx() as conn:
            existing = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO experiments (
                        experiment_id, suite_preset, stage_name, stage_type, stage_order,
                        dependencies_json, config_json, config_fingerprint, dataset_fingerprint,
                        split_fingerprint, upstream_fingerprint, model_variant, seed,
                        label_fraction, threshold, status, target_steps, last_completed_step,
                        output_dir, last_update_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, 0, ?, ?)
                    """,
                    (
                        experiment_id, suite_preset, stage_name, stage_type, int(stage_order),
                        deps_json, cfg_json, config_fingerprint, dataset_fingerprint,
                        split_fingerprint, upstream_fingerprint, model_variant, seed,
                        label_fraction, threshold, int(target_steps), out_str, now,
                    ),
                )
            else:
                # Preserve status, checkpoints, runtime and recorded fingerprints of completed
                # runs so dependency/config mismatch checks can detect drift cleanly.
                conn.execute(
                    """
                    UPDATE experiments SET
                        stage_order = ?,
                        stage_type = ?,
                        dependencies_json = ?,
                        target_steps = ?,
                        output_dir = COALESCE(?, output_dir),
                        last_update_time = ?
                    WHERE experiment_id = ?
                    """,
                    (
                        int(stage_order), stage_type, deps_json,
                        int(target_steps), out_str, now, experiment_id,
                    ),
                )
                if existing["status"] == "PENDING":
                    conn.execute(
                        """
                        UPDATE experiments SET
                            config_json = ?,
                            config_fingerprint = ?,
                            dataset_fingerprint = ?,
                            split_fingerprint = ?,
                            upstream_fingerprint = ?,
                            model_variant = ?,
                            seed = ?,
                            label_fraction = ?,
                            threshold = ?
                        WHERE experiment_id = ?
                        """,
                        (
                            cfg_json, config_fingerprint, dataset_fingerprint,
                            split_fingerprint, upstream_fingerprint, model_variant,
                            seed, label_fraction, threshold, experiment_id,
                        ),
                    )
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return self._decode_row(row)  # type: ignore[return-value]

    def get_stage(self, experiment_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            return self._decode_row(row)
        finally:
            conn.close()

    def list_stages(self, preset: str | None = None) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            if preset:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE suite_preset = ? ORDER BY stage_order ASC, stage_name ASC",
                    (preset,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM experiments ORDER BY suite_preset ASC, stage_order ASC, stage_name ASC"
                ).fetchall()
            return [self._decode_row(r) for r in rows if r is not None]
        finally:
            conn.close()

    # ------------------------------------------------------------ transitions
    def mark_running(
        self,
        experiment_id: str,
        *,
        pid: int | None = None,
        host: str | None = None,
        gpu_name: str | None = None,
        config_fingerprint: str | None = None,
        dataset_fingerprint: str | None = None,
        split_fingerprint: str | None = None,
        upstream_fingerprint: str | None = None,
        config: dict[str, Any] | None = None,
        resume_mode: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        proc_pid = int(pid if pid is not None else os.getpid())
        proc_host = host or host_id()
        with self._tx() as conn:
            existing = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            if existing is None:
                raise KeyError(f"Unknown experiment_id: {experiment_id}")
            prev_status = existing["status"]
            retry_count = int(existing["retry_count"] or 0)
            history = json.loads(existing["retry_history_json"] or "[]")
            if prev_status in ("FAILED", "INTERRUPTED"):
                retry_count += 1
                history.append({
                    "event": "retry_started",
                    "from_status": prev_status,
                    "attempt": retry_count,
                    "utc": now,
                    "host": proc_host,
                    "pid": proc_pid,
                })
            conn.execute(
                """
                UPDATE experiments SET
                    status = 'RUNNING',
                    start_time = COALESCE(start_time, ?),
                    last_update_time = ?,
                    heartbeat_utc = ?,
                    pid = ?,
                    host = ?,
                    gpu_name = COALESCE(?, gpu_name),
                    config_fingerprint = COALESCE(?, config_fingerprint),
                    dataset_fingerprint = COALESCE(?, dataset_fingerprint),
                    split_fingerprint = COALESCE(?, split_fingerprint),
                    upstream_fingerprint = COALESCE(?, upstream_fingerprint),
                    config_json = COALESCE(?, config_json),
                    resume_mode = COALESCE(?, resume_mode),
                    error_info = NULL,
                    retry_count = ?,
                    retry_history_json = ?
                WHERE experiment_id = ?
                """,
                (
                    now, now, now, proc_pid, proc_host, gpu_name,
                    config_fingerprint, dataset_fingerprint, split_fingerprint,
                    upstream_fingerprint,
                    json.dumps(config, sort_keys=True, default=str) if config is not None else None,
                    resume_mode,
                    retry_count, json.dumps(history), experiment_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return self._decode_row(row)  # type: ignore[return-value]

    def update_progress(
        self,
        experiment_id: str,
        *,
        last_completed_step: int | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_sha256: str | None = None,
        resume_mode: str | None = None,
        accumulated_runtime_sec: float | None = None,
        runtime_delta_sec: float = 0.0,
        artifacts: dict[str, Any] | None = None,
        gpu_name: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._tx() as conn:
            existing = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            if existing is None:
                raise KeyError(f"Unknown experiment_id: {experiment_id}")
            merged_artifacts = json.loads(existing["artifacts_json"] or "{}")
            if artifacts:
                merged_artifacts.update(artifacts)
            runtime = (
                float(accumulated_runtime_sec)
                if accumulated_runtime_sec is not None
                else float(existing["accumulated_runtime_sec"] or 0.0) + max(0.0, float(runtime_delta_sec))
            )
            step = (
                int(last_completed_step)
                if last_completed_step is not None
                else int(existing["last_completed_step"] or 0)
            )
            conn.execute(
                """
                UPDATE experiments SET
                    last_completed_step = ?,
                    checkpoint_path = COALESCE(?, checkpoint_path),
                    checkpoint_sha256 = COALESCE(?, checkpoint_sha256),
                    resume_mode = COALESCE(?, resume_mode),
                    accumulated_runtime_sec = ?,
                    artifacts_json = ?,
                    gpu_name = COALESCE(?, gpu_name),
                    last_update_time = ?,
                    heartbeat_utc = ?
                WHERE experiment_id = ?
                """,
                (
                    step,
                    str(checkpoint_path) if checkpoint_path is not None else None,
                    checkpoint_sha256,
                    resume_mode,
                    round(runtime, 3),
                    json.dumps(merged_artifacts, sort_keys=True, default=str),
                    gpu_name,
                    now,
                    now,
                    experiment_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return self._decode_row(row)  # type: ignore[return-value]

    def mark_completed(
        self,
        experiment_id: str,
        *,
        last_completed_step: int | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_sha256: str | None = None,
        artifacts: dict[str, Any] | None = None,
        accumulated_runtime_sec: float | None = None,
        runtime_delta_sec: float = 0.0,
        config_fingerprint: str | None = None,
        dataset_fingerprint: str | None = None,
        split_fingerprint: str | None = None,
        upstream_fingerprint: str | None = None,
        resume_mode: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._tx() as conn:
            existing = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            if existing is None:
                raise KeyError(f"Unknown experiment_id: {experiment_id}")
            merged_artifacts = json.loads(existing["artifacts_json"] or "{}")
            if artifacts:
                merged_artifacts.update(artifacts)
            runtime = (
                float(accumulated_runtime_sec)
                if accumulated_runtime_sec is not None
                else float(existing["accumulated_runtime_sec"] or 0.0) + max(0.0, float(runtime_delta_sec))
            )
            step = (
                int(last_completed_step)
                if last_completed_step is not None
                else max(int(existing["last_completed_step"] or 0), int(existing["target_steps"] or 0))
            )
            conn.execute(
                """
                UPDATE experiments SET
                    status = 'COMPLETED',
                    last_completed_step = ?,
                    checkpoint_path = COALESCE(?, checkpoint_path),
                    checkpoint_sha256 = COALESCE(?, checkpoint_sha256),
                    artifacts_json = ?,
                    accumulated_runtime_sec = ?,
                    config_fingerprint = COALESCE(?, config_fingerprint),
                    dataset_fingerprint = COALESCE(?, dataset_fingerprint),
                    split_fingerprint = COALESCE(?, split_fingerprint),
                    upstream_fingerprint = COALESCE(?, upstream_fingerprint),
                    resume_mode = COALESCE(?, resume_mode),
                    start_time = COALESCE(start_time, ?),
                    last_update_time = ?,
                    completed_time = ?,
                    heartbeat_utc = ?,
                    error_info = NULL,
                    pid = NULL
                WHERE experiment_id = ?
                """,
                (
                    step,
                    str(checkpoint_path) if checkpoint_path is not None else None,
                    checkpoint_sha256,
                    json.dumps(merged_artifacts, sort_keys=True, default=str),
                    round(runtime, 3),
                    config_fingerprint,
                    dataset_fingerprint,
                    split_fingerprint,
                    upstream_fingerprint,
                    resume_mode,
                    now, now, now, now, experiment_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return self._decode_row(row)  # type: ignore[return-value]

    def mark_failed(
        self,
        experiment_id: str,
        error_info: str,
        *,
        status: str = "FAILED",
        last_completed_step: int | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_sha256: str | None = None,
        accumulated_runtime_sec: float | None = None,
        runtime_delta_sec: float = 0.0,
    ) -> dict[str, Any]:
        if status not in ("FAILED", "INTERRUPTED"):
            raise ValueError(f"Invalid failure status: {status}")
        now = utc_now()
        with self._tx() as conn:
            existing = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            if existing is None:
                raise KeyError(f"Unknown experiment_id: {experiment_id}")
            history = json.loads(existing["retry_history_json"] or "[]")
            history.append({
                "event": status.lower(),
                "error": str(error_info)[:1000],
                "step": int(last_completed_step) if last_completed_step is not None else int(existing["last_completed_step"] or 0),
                "utc": now,
                "host": existing["host"] or host_id(),
                "pid": existing["pid"],
            })
            runtime = (
                float(accumulated_runtime_sec)
                if accumulated_runtime_sec is not None
                else float(existing["accumulated_runtime_sec"] or 0.0) + max(0.0, float(runtime_delta_sec))
            )
            step = (
                int(last_completed_step)
                if last_completed_step is not None
                else int(existing["last_completed_step"] or 0)
            )
            conn.execute(
                """
                UPDATE experiments SET
                    status = ?,
                    last_completed_step = ?,
                    checkpoint_path = COALESCE(?, checkpoint_path),
                    checkpoint_sha256 = COALESCE(?, checkpoint_sha256),
                    accumulated_runtime_sec = ?,
                    error_info = ?,
                    retry_history_json = ?,
                    last_update_time = ?,
                    heartbeat_utc = ?,
                    pid = NULL
                WHERE experiment_id = ?
                """,
                (
                    status,
                    step,
                    str(checkpoint_path) if checkpoint_path is not None else None,
                    checkpoint_sha256,
                    round(runtime, 3),
                    str(error_info)[:2000],
                    json.dumps(history[-50:]),
                    now,
                    now,
                    experiment_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return self._decode_row(row)  # type: ignore[return-value]

    def invalidate_stage(
        self,
        experiment_id: str,
        reason: str,
        *,
        new_config: dict[str, Any] | None = None,
        new_config_fingerprint: str | None = None,
        new_dataset_fingerprint: str | None = None,
        new_split_fingerprint: str | None = None,
        new_upstream_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        """Reset a stage to PENDING after upstream/config invalidation."""
        now = utc_now()
        with self._tx() as conn:
            existing = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            if existing is None:
                raise KeyError(f"Unknown experiment_id: {experiment_id}")
            history = json.loads(existing["retry_history_json"] or "[]")
            history.append({
                "event": "invalidated",
                "from_status": existing["status"],
                "reason": reason,
                "utc": now,
            })
            conn.execute(
                """
                UPDATE experiments SET
                    status = 'PENDING',
                    last_completed_step = 0,
                    checkpoint_path = NULL,
                    checkpoint_sha256 = NULL,
                    completed_time = NULL,
                    error_info = ?,
                    config_json = COALESCE(?, config_json),
                    config_fingerprint = COALESCE(?, config_fingerprint),
                    dataset_fingerprint = COALESCE(?, dataset_fingerprint),
                    split_fingerprint = COALESCE(?, split_fingerprint),
                    upstream_fingerprint = COALESCE(?, upstream_fingerprint),
                    retry_history_json = ?,
                    last_update_time = ?,
                    pid = NULL
                WHERE experiment_id = ?
                """,
                (
                    f"invalidated: {reason}",
                    json.dumps(new_config, sort_keys=True, default=str) if new_config is not None else None,
                    new_config_fingerprint,
                    new_dataset_fingerprint,
                    new_split_fingerprint,
                    new_upstream_fingerprint,
                    json.dumps(history[-50:]),
                    now,
                    experiment_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return self._decode_row(row)  # type: ignore[return-value]

    def recover_stale_running(
        self,
        *,
        stale_after_sec: float = 600.0,
        current_host: str | None = None,
        preset: str | None = None,
    ) -> list[dict[str, Any]]:
        """Detect crashed/abandoned RUNNING jobs and transition them to INTERRUPTED.

        A RUNNING record is stale when:
        - It belongs to ``current_host`` and its PID is no longer alive (or its
          experiment lock is not held when an output_dir exists), OR
        - It belongs to another host (e.g., an interrupted Kaggle/Colab session)
          and its heartbeat is older than ``stale_after_sec``.
        """
        me = current_host or host_id()
        recovered: list[dict[str, Any]] = []
        running = [r for r in self.list_stages(preset=preset) if r["status"] == "RUNNING"]
        for row in running:
            reason = None
            row_host = row.get("host") or ""
            row_pid = row.get("pid")
            out_dir = row.get("output_dir")
            lock_file = Path(out_dir) / ".experiment.lock" if out_dir else None
            if row_host == me or not row_host:
                alive = pid_alive(row_pid) if row_pid else False
                locked = is_locked(lock_file) if lock_file and lock_file.exists() else False
                if not alive and not locked:
                    reason = f"stale RUNNING recovered: process pid={row_pid} on host={row_host or me} is no longer active"
            else:
                age = seconds_since(row.get("heartbeat_utc") or row.get("last_update_time"))
                if age is None or age > float(stale_after_sec):
                    reason = (
                        f"stale remote RUNNING recovered: host={row_host} heartbeat expired "
                        f"({round(age, 1) if age is not None else 'unknown'}s > {stale_after_sec}s)"
                    )
            if reason:
                updated = self.mark_failed(row["experiment_id"], reason, status="INTERRUPTED")
                recovered.append(updated)
        return recovered

    # ------------------------------------------------------------ suite meta
    def set_meta(self, key: str, value: Any) -> None:
        now = utc_now()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO suite_meta (key, value_json, updated_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_utc = excluded.updated_utc
                """,
                (key, json.dumps(value, default=str), now),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        conn = self._connect()
        try:
            row = conn.execute("SELECT value_json FROM suite_meta WHERE key = ?", (key,)).fetchone()
            if row is None:
                return default
            return json.loads(row["value_json"])
        finally:
            conn.close()

    def save_suite_state(self, preset: str, output: str | Path, command: str, args_dict: dict[str, Any]) -> None:
        self.set_meta("last_suite", {
            "preset": preset,
            "output": str(output),
            "command": command,
            "args": args_dict,
            "host": host_id(),
            "updated_utc": utc_now(),
        })

    def load_suite_state(self) -> dict[str, Any] | None:
        return self.get_meta("last_suite", None)

    # ------------------------------------------------------------ benchmarks
    def save_benchmark(
        self,
        *,
        gpu_name: str | None,
        device_type: str,
        amp: str,
        max_side: int,
        batch_size: int,
        sec_per_step: float,
        peak_vram_gib: float | None,
        measured_steps: int,
        details: dict[str, Any] | None = None,
    ) -> str:
        key = stable_hash({
            "gpu_name": gpu_name or "cpu",
            "device_type": device_type,
            "amp": amp,
            "max_side": int(max_side),
            "batch_size": int(batch_size),
        })[:24]
        now = utc_now()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO benchmarks (
                    benchmark_key, gpu_name, device_type, amp, max_side, batch_size,
                    sec_per_step, peak_vram_gib, measured_steps, updated_utc, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(benchmark_key) DO UPDATE SET
                    sec_per_step = excluded.sec_per_step,
                    peak_vram_gib = excluded.peak_vram_gib,
                    measured_steps = excluded.measured_steps,
                    updated_utc = excluded.updated_utc,
                    details_json = excluded.details_json
                """,
                (
                    key, gpu_name, device_type, amp, int(max_side), int(batch_size),
                    float(sec_per_step), peak_vram_gib, int(measured_steps), now,
                    json.dumps(details or {}, default=str),
                ),
            )
        return key

    def get_benchmark(
        self,
        *,
        gpu_name: str | None,
        device_type: str,
        amp: str,
        max_side: int,
        batch_size: int,
    ) -> dict[str, Any] | None:
        key = stable_hash({
            "gpu_name": gpu_name or "cpu",
            "device_type": device_type,
            "amp": amp,
            "max_side": int(max_side),
            "batch_size": int(batch_size),
        })[:24]
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM benchmarks WHERE benchmark_key = ?", (key,)).fetchone()
            if row is None:
                return None
            out = dict(row)
            out["details"] = json.loads(out.get("details_json") or "{}")
            return out
        finally:
            conn.close()

    def list_benchmarks(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM benchmarks ORDER BY updated_utc DESC").fetchall()
            out = []
            for row in rows:
                d = dict(row)
                d["details"] = json.loads(d.get("details_json") or "{}")
                out.append(d)
            return out
        finally:
            conn.close()

    # ------------------------------------------------------------ summary
    def summary_stats(self, preset: str | None = None) -> dict[str, Any]:
        stages = self.list_stages(preset=preset)
        counts = {s: 0 for s in VALID_STATUSES}
        total_steps = 0
        completed_steps = 0
        accumulated_runtime = 0.0
        current_stage = None
        last_ckpt = None
        last_ckpt_time = ""

        for st in stages:
            status = st["status"]
            if status in counts:
                counts[status] += 1
            target = int(st.get("target_steps") or 0)
            done = int(st.get("last_completed_step") or 0)
            total_steps += target
            completed_steps += min(done, target) if target > 0 else done
            accumulated_runtime += float(st.get("accumulated_runtime_sec") or 0.0)
            if status == "RUNNING" and current_stage is None:
                current_stage = st
            if st.get("checkpoint_path"):
                ts = str(st.get("last_update_time") or "")
                if ts >= last_ckpt_time:
                    last_ckpt_time = ts
                    last_ckpt = {
                        "experiment_id": st["experiment_id"],
                        "stage_name": st["stage_name"],
                        "step": done,
                        "path": st["checkpoint_path"],
                        "sha256": st.get("checkpoint_sha256"),
                        "updated_utc": ts,
                    }

        if current_stage is None:
            for preferred in ("INTERRUPTED", "FAILED", "PENDING"):
                current_stage = next((s for s in stages if s["status"] == preferred), None)
                if current_stage is not None:
                    break

        return {
            "preset": preset,
            "total_stages": len(stages),
            "counts": counts,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "remaining_steps": max(0, total_steps - completed_steps),
            "accumulated_runtime_sec": round(accumulated_runtime, 2),
            "current_stage": {
                "experiment_id": current_stage["experiment_id"],
                "stage_name": current_stage["stage_name"],
                "stage_type": current_stage["stage_type"],
                "status": current_stage["status"],
                "step": int(current_stage.get("last_completed_step") or 0),
                "target_steps": int(current_stage.get("target_steps") or 0),
            } if current_stage else None,
            "last_checkpoint": last_ckpt,
        }

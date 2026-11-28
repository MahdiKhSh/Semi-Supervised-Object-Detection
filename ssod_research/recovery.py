"""Crash-safe primitives shared by training, evaluation and orchestration.

Design rules (see docs/DECISIONS.md):

* Every durable file is written to a temporary file in the destination
  directory, flushed, fsynced and atomically renamed into place.
* Checkpoints are never trusted blindly: each one has a JSON sidecar with
  its SHA-256, size, step and experiment fingerprint.  The sidecar is
  written *after* the checkpoint, so a checkpoint without a sidecar is an
  incomplete write and is ignored.
* Corrupted checkpoints are quarantined (renamed), never deleted, and the
  loader falls back to the previous valid checkpoint.
* Exclusive OS-level locks (released automatically by the OS when a process
  dies) prevent two processes from modifying the same experiment.

This module must stay importable without torch (torch is imported lazily).
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import time

CHECKPOINT_FORMAT_VERSION = 2
_HASH_CACHE: dict = {}


# --------------------------------------------------------------------------- time / host
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_utc(text):
    if not text:
        return None
    try:
        value = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def seconds_since(text):
    value = parse_utc(text)
    if value is None:
        return None
    return (datetime.now(timezone.utc) - value).total_seconds()


def host_id() -> str:
    return os.environ.get("SSOD_HOST_ID") or socket.gethostname() or "unknown-host"


def pid_alive(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --------------------------------------------------------------------------- atomic IO
def _fsync_dir(directory) -> None:
    if os.name == "nt":
        return  # Windows has no directory fsync; NTFS rename is journaled.
    with contextlib.suppress(OSError):
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def replace_file(src, dst, attempts: int = 40) -> None:
    """os.replace with retries: Windows refuses while a reader/AV holds dst."""
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(0.5, 0.02 * (attempt + 1)))


def atomic_write_bytes(path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        replace_file(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _fsync_dir(path.parent)
    return path


def atomic_write_text(path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf8"))


def atomic_write_json(path, obj, indent=2) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=indent, default=str) + "\n")


def read_json(path, default=None):
    """Return parsed JSON, or ``default`` when missing *or unreadable*."""
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return default


def atomic_copy(src, dst) -> Path:
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
    try:
        with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
            for block in iter(lambda: inp.read(8 << 20), b""):
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        replace_file(tmp, dst)
        with contextlib.suppress(OSError):
            stat = src.stat()
            os.utime(dst, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return dst


# --------------------------------------------------------------------------- JSON lines
def read_jsonl(path):
    """Read JSON lines, skipping a torn last line left by a power loss."""
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def append_jsonl(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(obj, default=str) + "\n")
        stream.flush()


def append_jsonl_batch(path, objs) -> None:
    """Append a batch of JSONL records in a single open/flush (16x-32x faster on NTFS/NFS)."""
    items = list(objs)
    if not items:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(obj, default=str) + "\n" for obj in items)
    with path.open("a", encoding="utf8") as stream:
        stream.write(payload)
        stream.flush()


def truncate_jsonl(path, max_step: int) -> int:
    """Keep only valid rows with step <= max_step (rollback to a checkpoint)."""
    path = Path(path)
    if not path.is_file():
        return 0
    kept = [row for row in read_jsonl(path) if int(row.get("step", -1)) <= int(max_step)]
    atomic_write_text(path, "".join(json.dumps(row) + "\n" for row in kept))
    return len(kept)


# --------------------------------------------------------------------------- hashing
def sha256_file(path, use_cache: bool = True) -> str:
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if use_cache and key in _HASH_CACHE:
        return _HASH_CACHE[key]
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    value = digest.hexdigest()
    _HASH_CACHE[key] = value
    return value


def stable_hash(obj) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf8")).hexdigest()


# --------------------------------------------------------------------------- locking
class LockBusy(RuntimeError):
    """Another live process owns the lock."""


class FileLock:
    """Exclusive, non-blocking OS lock; released by the OS if the owner dies.

    A sibling ``<lock>.info`` file stores the owner (pid, host, purpose) for
    diagnostics only.  Correctness relies on the OS lock, not on the info file.
    """

    def __init__(self, path, purpose: str = ""):
        self.path = Path(path)
        self.info_path = self.path.with_name(self.path.name + ".info")
        self.purpose = purpose
        self._fd = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def owner(self):
        return read_json(self.info_path, {}) or {}

    def acquire(self, timeout: float = 0.0):
        if self._fd is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                if os.name == "nt":
                    import msvcrt
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                if time.monotonic() < deadline:
                    time.sleep(0.2)
                    continue
                owner = self.owner()
                raise LockBusy(
                    f"{self.path} is locked by pid={owner.get('pid')} host={owner.get('host')} "
                    f"since {owner.get('since')} ({owner.get('purpose', '')})") from None
            self._fd = fd
            break
        with contextlib.suppress(OSError):
            atomic_write_json(self.info_path, {
                "pid": os.getpid(), "host": host_id(), "since": utc_now(), "purpose": self.purpose})
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        with contextlib.suppress(OSError):
            os.unlink(self.info_path)
        try:
            if os.name == "nt":
                import msvcrt
                with contextlib.suppress(OSError):
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


def is_locked(path) -> bool:
    """True when a *live* process holds the lock (probe without keeping it)."""
    if not Path(path).exists():
        return False
    probe = FileLock(path)
    try:
        probe.acquire()
    except LockBusy:
        return True
    probe.release()
    return False


# --------------------------------------------------------------------------- RNG state
def capture_rng_state() -> dict:
    """Python, NumPy, torch CPU and CUDA RNG, in a weights_only-safe format."""
    import random
    import numpy as np
    import torch
    name, keys, pos, has_gauss, cached = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {"name": str(name), "keys": [int(k) for k in keys], "pos": int(pos),
                  "has_gauss": int(has_gauss), "cached_gaussian": float(cached)},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict, device_type: str) -> list:
    """Restore RNG state; return human-readable reasons when not exact."""
    import random
    import numpy as np
    import torch
    reasons = []
    if not state:
        return ["checkpoint has no RNG state"]
    python_state = state.get("python")
    if python_state is not None:
        version, internal, gauss = python_state
        random.setstate((int(version), tuple(int(v) for v in internal), gauss))
    else:
        reasons.append("python RNG missing")
    numpy_state = state.get("numpy")
    if numpy_state:
        np.random.set_state((numpy_state["name"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                             int(numpy_state["pos"]), int(numpy_state["has_gauss"]),
                             float(numpy_state["cached_gaussian"])))
    else:
        reasons.append("numpy RNG missing")
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"].cpu())
    else:
        reasons.append("torch CPU RNG missing")
    if device_type == "cuda":
        saved = state.get("torch_cuda")
        if saved and len(saved) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in saved])
        else:
            reasons.append("CUDA RNG state unavailable for this device set")
    return reasons


# --------------------------------------------------------------------------- checkpoints
class CheckpointError(RuntimeError):
    pass


class IncompatibleCheckpoint(CheckpointError):
    """A checkpoint belongs to a different configuration or dataset."""


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def verify_file(path, sidecar: dict, deep: bool = True):
    """Return None when valid, otherwise the reason."""
    path = Path(path)
    if not path.is_file():
        return "file missing"
    size = path.stat().st_size
    if int(sidecar.get("size", -1)) != size:
        return f"size {size} != recorded {sidecar.get('size')}"
    if deep and sidecar.get("sha256") != sha256_file(path, use_cache=False):
        return "sha256 mismatch (corrupted file)"
    return None


def torch_save_atomic(state: dict, path, extra_sidecar: dict | None = None) -> dict:
    """Serialize, fsync, hash, atomically rename, then write the sidecar."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(tmp, use_cache=False)
        size = os.path.getsize(tmp)
        replace_file(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _fsync_dir(path.parent)
    sidecar = {"format_version": CHECKPOINT_FORMAT_VERSION, "file": path.name,
               "sha256": digest, "size": size, "created_utc": utc_now(), "host": host_id()}
    sidecar.update(extra_sidecar or {})
    atomic_write_json(_sidecar(path), sidecar)
    return sidecar


def safe_torch_load(path):
    """Load only tensors / primitive containers (no arbitrary pickled code)."""
    import torch
    return torch.load(str(path), map_location="cpu", weights_only=True)


def quarantine(path, reason: str = "") -> Path | None:
    """Rename a bad file (and sidecar) aside; never delete evidence."""
    path = Path(path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    moved = None
    for item in (path, _sidecar(path)):
        if item.exists():
            target = item.with_name(f"{item.name}.corrupt-{stamp}")
            with contextlib.suppress(OSError):
                replace_file(item, target)
                moved = moved or target
    if reason:
        with contextlib.suppress(OSError):
            append_jsonl(path.parent / "quarantine.jsonl",
                         {"file": path.name, "reason": reason, "utc": utc_now()})
    return moved


class CheckpointManager:
    """Rolling verified checkpoints ``ckpt_<step>.pth`` + ``ckpt_<step>.json``."""

    def __init__(self, directory, keep: int = 2):
        self.directory = Path(directory)
        self.keep = max(2, int(keep))  # always latest + at least one previous

    def path_for(self, step: int) -> Path:
        return self.directory / f"ckpt_{int(step):09d}.pth"

    def save(self, state: dict, step: int, sidecar: dict) -> Path:
        path = self.path_for(step)
        torch_save_atomic(state, path, {"step": int(step), **sidecar})
        self.prune()
        return path

    def entries(self):
        """Newest first: (step, path, sidecar dict or None)."""
        if not self.directory.is_dir():
            return []
        found = []
        for item in self.directory.glob("ckpt_*.pth"):
            try:
                step = int(item.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            found.append((step, item, read_json(_sidecar(item))))
        found.sort(key=lambda x: x[0], reverse=True)
        return found

    def latest_info(self):
        """Fast (no hashing) view of the newest checkpoint with a sidecar."""
        for step, path, sidecar in self.entries():
            if sidecar and path.is_file():
                return {"step": step, "path": str(path), **sidecar}
        return None

    def cleanup_temporary(self) -> None:
        if self.directory.is_dir():
            for tmp in self.directory.glob(".ckpt_*.tmp"):
                with contextlib.suppress(OSError):
                    tmp.unlink()

    def prune(self, keep: int | None = None) -> None:
        keep = self.keep if keep is None else max(1, int(keep))
        valid = [(s, p) for s, p, side in self.entries() if side and p.is_file()]
        for _, path in valid[keep:]:
            for item in (path, _sidecar(path)):
                with contextlib.suppress(OSError):
                    item.unlink()

    def load_latest_valid(self, fingerprint: str, deep: bool = True):
        """Return (state, info, report).  Falls back past corrupted files.

        Raises IncompatibleCheckpoint if a checkpoint belongs to another
        configuration: mixing runs inside one directory is never allowed.
        """
        report = []
        for step, path, sidecar in self.entries():
            if sidecar is None:
                report.append(f"step {step}: no sidecar (incomplete write) -> ignored")
                quarantine(path, "missing sidecar")
                continue
            if sidecar.get("fingerprint") != fingerprint:
                raise IncompatibleCheckpoint(
                    f"{path} was produced by a different configuration/dataset "
                    f"(fingerprint {str(sidecar.get('fingerprint'))[:12]} != {fingerprint[:12]})")
            problem = verify_file(path, sidecar, deep=deep)
            if problem:
                report.append(f"step {step}: {problem} -> quarantined, falling back")
                quarantine(path, problem)
                continue
            try:
                state = safe_torch_load(path)
            except Exception as exc:  # noqa: BLE001 - any unreadable file is corrupt
                report.append(f"step {step}: unreadable ({type(exc).__name__}) -> quarantined")
                quarantine(path, f"load failed: {exc!r}")
                continue
            if state.get("fingerprint") != fingerprint or int(state.get("step", -1)) != step:
                report.append(f"step {step}: internal metadata mismatch -> quarantined")
                quarantine(path, "internal metadata mismatch")
                continue
            return state, {"step": step, "path": str(path), **sidecar}, report
        return None, None, report

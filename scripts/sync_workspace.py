#!/usr/bin/env python3
"""Portable checkpoint bundle transfer and lightweight GitHub results sync.

Examples:
  # On Kaggle (end of session): export verified checkpoints + SQLite state
  python scripts/sync_workspace.py export --output work_dirs/core_runs --bundle /kaggle/working/ssod_bundle.tar.gz --archive

  # On RTX 5090 or L40s (next day): verify checksums + GPU compatibility and import
  python scripts/sync_workspace.py import --bundle ssod_bundle.tar.gz --output work_dirs/core_runs

  # Push ONLY lightweight progress, tables, statistics, and figures to GitHub
  python scripts/sync_workspace.py github --preset core --output work_dirs/core_runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssod_research.cloud_sync import (
    detect_runtime_environment,
    export_portable_bundle,
    import_portable_bundle,
    sync_directory,
    sync_results_to_github,
)


def _target_hardware():
    try:
        import torch
        from ssod_research.native_training import resolve_amp
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amp_mode, _, _ = resolve_amp("auto", dev)
        gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else None
        return dev.type, amp_mode, gpu
    except Exception:  # noqa: BLE001
        return "cpu", "off", None


def main(argv=None):
    p = argparse.ArgumentParser(description="Cross-machine SSOD state transfer and GitHub sync")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_env = sub.add_parser("env", help="Detect local/cloud environment and storage persistence")
    p_env.add_argument("--output", type=Path, default=ROOT / "work_dirs")

    p_exp = sub.add_parser("export", help="Export verified checkpoints + SQLite DB into a portable bundle")
    p_exp.add_argument("--output", type=Path, required=True)
    p_exp.add_argument("--bundle", type=Path, required=True)
    p_exp.add_argument("--archive", action="store_true", help="Also pack as a .tar.gz archive")

    p_imp = sub.add_parser("import", help="Verify and import a portable bundle on a new GPU/machine")
    p_imp.add_argument("--bundle", type=Path, required=True)
    p_imp.add_argument("--output", type=Path, required=True)

    p_sync = sub.add_parser("sync-dir", help="Sync experiment state atomically between two directories")
    p_sync.add_argument("--src", type=Path, required=True)
    p_sync.add_argument("--dst", type=Path, required=True)

    p_gh = sub.add_parser("github", help="Commit and push ONLY lightweight results & progress docs to GitHub")
    p_gh.add_argument("--preset", required=True)
    p_gh.add_argument("--output", type=Path, required=True)
    p_gh.add_argument("--message", default=None)
    p_gh.add_argument("--no-push", action="store_true")
    p_gh.add_argument("--branch", default=None)

    args = p.parse_args(argv)
    if args.cmd == "env":
        print(json.dumps(detect_runtime_environment(args.output), indent=2))
        return
    if args.cmd == "export":
        res = export_portable_bundle(args.output, args.bundle, create_archive=args.archive)
        print(json.dumps(res, indent=2))
        return
    if args.cmd == "import":
        dev_type, amp_mode, gpu = _target_hardware()
        res = import_portable_bundle(
            args.bundle,
            args.output,
            target_device_type=dev_type,
            target_amp=amp_mode,
            target_gpu_name=gpu,
        )
        print(json.dumps(res, indent=2))
        return
    if args.cmd == "sync-dir":
        res = sync_directory(args.src, args.dst)
        print(json.dumps(res, indent=2))
        return
    if args.cmd == "github":
        res = sync_results_to_github(
            ROOT,
            preset=args.preset,
            output_dir=args.output,
            message=args.message,
            push=not args.no_push,
            branch=args.branch,
        )
        print(json.dumps(res, indent=2))
        return


if __name__ == "__main__":
    main()

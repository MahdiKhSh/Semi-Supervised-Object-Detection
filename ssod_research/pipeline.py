"""Dependency-aware pipeline DAG, runtime estimation, and progress reporting.

Models all 15 research pipeline stages explicitly:
 1. environment_verification
 2. dataset_validation
 3. gpu_benchmarking
 4. dataset_splitting
 5. supervised_warmup
 6. baseline_training
 7. mean_teacher_training
 8. proposed_training
 9. threshold_experiment
10. ablation_experiment
11. model_evaluation
12. condition_per_class_evaluation
13. statistical_analysis
14. plot_generation
15. final_report
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ssod_research.recovery import atomic_write_json, atomic_write_text, stable_hash, utc_now
from ssod_research.registry import ExperimentRegistry, make_experiment_id

STAGE_TYPES = (
    "environment_verification",
    "dataset_validation",
    "gpu_benchmarking",
    "dataset_splitting",
    "supervised_warmup",
    "baseline_training",
    "mean_teacher_training",
    "proposed_training",
    "threshold_experiment",
    "ablation_experiment",
    "model_evaluation",
    "condition_per_class_evaluation",
    "statistical_analysis",
    "plot_generation",
    "final_report",
)


def classify_training_stage(variant: str, tag: str) -> str:
    if tag == "warmup":
        return "supervised_warmup"
    if tag.startswith("threshold"):
        return "threshold_experiment"
    if tag.startswith("no_"):
        return "ablation_experiment"
    if variant == "supervised":
        return "baseline_training"
    if variant == "mean_teacher":
        return "mean_teacher_training"
    return "proposed_training"


@dataclass
class StageNode:
    experiment_id: str
    suite_preset: str
    stage_name: str
    stage_type: str
    order: int
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    config_fingerprint: str = ""
    dataset_fingerprint: str = ""
    split_fingerprint: str = ""
    model_variant: str | None = None
    seed: int | None = None
    label_fraction: float | None = None
    threshold: float | None = None
    target_steps: int = 0
    output_dir: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


class PipelineDAG:
    """Directed acyclic graph of independently resumable pipeline stages."""

    def __init__(self, nodes: list[StageNode]):
        self.nodes: dict[str, StageNode] = {}
        self.order_ids: list[str] = []
        for node in nodes:
            if node.experiment_id in self.nodes:
                raise ValueError(f"Duplicate DAG stage id: {node.experiment_id}")
            self.nodes[node.experiment_id] = node
            self.order_ids.append(node.experiment_id)
        self._validate()

    def _validate(self) -> None:
        in_degree = {nid: 0 for nid in self.nodes}
        children: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Stage {nid} depends on unknown stage {dep}")
                children[dep].append(nid)
                in_degree[nid] += 1
        q = deque(nid for nid in self.order_ids if in_degree[nid] == 0)
        visited = []
        while q:
            cur = q.popleft()
            visited.append(cur)
            for child in children[cur]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    q.append(child)
        if len(visited) != len(self.nodes):
            raise ValueError("Cycle detected in pipeline DAG")
        self._topo = visited
        self._children = children

    def topological_order(self) -> list[StageNode]:
        return [self.nodes[nid] for nid in self._topo]

    def children_of(self, experiment_id: str) -> list[str]:
        return list(self._children.get(experiment_id, []))

    def downstream_of(self, experiment_id: str) -> list[str]:
        seen = set()
        ordered = []
        q = deque(self.children_of(experiment_id))
        while q:
            cur = q.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            ordered.append(cur)
            for nxt in self.children_of(cur):
                if nxt not in seen:
                    q.append(nxt)
        return ordered

    def effective_fingerprints(self) -> dict[str, str]:
        """Compute transitive fingerprint of each node from its own config/data + upstreams."""
        eff: dict[str, str] = {}
        for node in self.topological_order():
            up = {dep: eff[dep] for dep in sorted(node.dependencies)}
            eff[node.experiment_id] = stable_hash({
                "config": node.config_fingerprint,
                "dataset": node.dataset_fingerprint,
                "split": node.split_fingerprint,
                "upstream": up,
            })
        return eff

    def upstream_fingerprint(self, experiment_id: str, eff: dict[str, str]) -> str:
        node = self.nodes[experiment_id]
        if not node.dependencies:
            return ""
        return stable_hash({dep: eff[dep] for dep in sorted(node.dependencies)})

    def reconcile_with_registry(
        self,
        registry: ExperimentRegistry,
        *,
        stale_after_sec: float = 600.0,
    ) -> dict[str, Any]:
        """Sync DAG into SQLite, recover stale RUNNING rows, invalidate only changed branches."""
        recovered = registry.recover_stale_running(stale_after_sec=stale_after_sec)
        eff = self.effective_fingerprints()
        invalidated: list[dict[str, str]] = []
        invalidated_set: set[str] = set()

        for node in self.topological_order():
            up_fp = self.upstream_fingerprint(node.experiment_id, eff)
            existing = registry.get_stage(node.experiment_id)
            if existing is None:
                registry.upsert_stage(
                    experiment_id=node.experiment_id,
                    suite_preset=node.suite_preset,
                    stage_name=node.stage_name,
                    stage_type=node.stage_type,
                    stage_order=node.order,
                    dependencies=node.dependencies,
                    config=node.config,
                    config_fingerprint=node.config_fingerprint,
                    dataset_fingerprint=node.dataset_fingerprint,
                    split_fingerprint=node.split_fingerprint,
                    upstream_fingerprint=up_fp,
                    model_variant=node.model_variant,
                    seed=node.seed,
                    label_fraction=node.label_fraction,
                    threshold=node.threshold,
                    target_steps=node.target_steps,
                    output_dir=node.output_dir,
                )
                continue

            # Check whether this node or any upstream changed since it was run.
            reasons = []
            if any(dep in invalidated_set for dep in node.dependencies):
                changed_deps = [d for d in node.dependencies if d in invalidated_set]
                reasons.append(f"upstream invalidated ({', '.join(changed_deps)})")
            if existing["status"] != "PENDING":
                if node.config_fingerprint and existing.get("config_fingerprint") and existing["config_fingerprint"] != node.config_fingerprint:
                    reasons.append("config_fingerprint changed")
                if node.dataset_fingerprint and existing.get("dataset_fingerprint") and existing["dataset_fingerprint"] != node.dataset_fingerprint:
                    reasons.append("dataset_fingerprint changed")
                if node.split_fingerprint and existing.get("split_fingerprint") and existing["split_fingerprint"] != node.split_fingerprint:
                    reasons.append("split_fingerprint changed")
                if up_fp and existing.get("upstream_fingerprint") and existing["upstream_fingerprint"] != up_fp:
                    reasons.append("upstream_fingerprint changed")

            if reasons:
                reason_text = "; ".join(reasons)
                registry.invalidate_stage(
                    node.experiment_id,
                    reason_text,
                    new_config=node.config,
                    new_config_fingerprint=node.config_fingerprint,
                    new_dataset_fingerprint=node.dataset_fingerprint,
                    new_split_fingerprint=node.split_fingerprint,
                    new_upstream_fingerprint=up_fp,
                )
                invalidated_set.add(node.experiment_id)
                invalidated.append({"experiment_id": node.experiment_id, "reason": reason_text})
            else:
                registry.upsert_stage(
                    experiment_id=node.experiment_id,
                    suite_preset=node.suite_preset,
                    stage_name=node.stage_name,
                    stage_type=node.stage_type,
                    stage_order=node.order,
                    dependencies=node.dependencies,
                    config=node.config,
                    config_fingerprint=node.config_fingerprint,
                    dataset_fingerprint=node.dataset_fingerprint,
                    split_fingerprint=node.split_fingerprint,
                    upstream_fingerprint=up_fp,
                    model_variant=node.model_variant,
                    seed=node.seed,
                    label_fraction=node.label_fraction,
                    threshold=node.threshold,
                    target_steps=node.target_steps,
                    output_dir=node.output_dir,
                )

        return {
            "recovered_stale": [r["experiment_id"] for r in recovered],
            "invalidated": invalidated,
        }


# ------------------------------------------------------------------ formatting & estimates
def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unmeasured (run RUN_BENCHMARK.cmd or complete steps)"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(round(seconds)), 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    if days > 0:
        return f"{days}d {hours}h {mins}m"
    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    return f"{mins}m {secs}s"


def compute_runtime_estimates(
    registry: ExperimentRegistry,
    *,
    preset: str | None = None,
) -> dict[str, Any]:
    """Compute honest remaining-runtime estimates from observed runs or GPU benchmark."""
    stages = registry.list_stages(preset=preset)
    train_types = {
        "supervised_warmup", "baseline_training", "mean_teacher_training",
        "proposed_training", "threshold_experiment", "ablation_experiment",
    }
    observed_steps = 0
    observed_sec = 0.0
    remaining_steps = 0
    per_type_rates: dict[str, list[float]] = {}

    for st in stages:
        if st["stage_type"] not in train_types:
            continue
        target = int(st.get("target_steps") or 0)
        done = min(int(st.get("last_completed_step") or 0), target) if target > 0 else 0
        runtime = float(st.get("accumulated_runtime_sec") or 0.0)
        if st["status"] != "COMPLETED":
            remaining_steps += max(0, target - done)
        if done > 0 and runtime > 0:
            observed_steps += done
            observed_sec += runtime
            per_type_rates.setdefault(st["stage_type"], []).append(runtime / done)

    sec_per_step = None
    source = "unmeasured"
    benchmark_used = None
    if observed_steps > 0 and observed_sec > 0:
        sec_per_step = observed_sec / observed_steps
        source = "observed_training_steps"
    else:
        benchmarks = registry.list_benchmarks()
        if benchmarks:
            benchmark_used = benchmarks[0]
            sec_per_step = float(benchmark_used["sec_per_step"])
            source = f"gpu_benchmark ({benchmark_used.get('gpu_name') or benchmark_used.get('device_type')})"

    estimated_remaining_sec = (
        round(remaining_steps * sec_per_step, 2)
        if sec_per_step is not None
        else None
    )
    return {
        "generated_utc": utc_now(),
        "preset": preset,
        "measurement_source": source,
        "measured_sec_per_step": round(sec_per_step, 4) if sec_per_step is not None else None,
        "observed_steps": observed_steps,
        "observed_runtime_sec": round(observed_sec, 2),
        "remaining_training_steps": remaining_steps,
        "estimated_remaining_sec": estimated_remaining_sec,
        "estimated_remaining_human": format_duration(estimated_remaining_sec),
        "per_stage_type_sec_per_step": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(per_type_rates.items())
        },
        "latest_benchmark": benchmark_used,
    }


def write_progress_reports(
    registry: ExperimentRegistry,
    *,
    preset: str | None,
    output_dir: Path,
    resume_command: str,
    workspace_root: Path | None = None,
    update_workspace_docs: bool = False,
) -> dict[str, Any]:
    """Write machine-readable and human-readable progress reports atomically."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = registry.summary_stats(preset=preset)
    estimates = compute_runtime_estimates(registry, preset=preset)
    stages = registry.list_stages(preset=preset)

    progress_payload = {
        "generated_utc": utc_now(),
        "preset": preset,
        "output_dir": str(output_dir),
        "registry_db": str(registry.db_path),
        "resume_command": resume_command,
        "summary": summary,
        "runtime_estimates": estimates,
        "stages": [
            {
                "experiment_id": s["experiment_id"],
                "stage_name": s["stage_name"],
                "stage_type": s["stage_type"],
                "status": s["status"],
                "variant": s.get("model_variant"),
                "seed": s.get("seed"),
                "label_fraction": s.get("label_fraction"),
                "threshold": s.get("threshold"),
                "last_completed_step": int(s.get("last_completed_step") or 0),
                "target_steps": int(s.get("target_steps") or 0),
                "accumulated_runtime_sec": float(s.get("accumulated_runtime_sec") or 0.0),
                "checkpoint_path": s.get("checkpoint_path"),
                "checkpoint_sha256": s.get("checkpoint_sha256"),
                "resume_mode": s.get("resume_mode"),
                "retry_count": int(s.get("retry_count") or 0),
                "error_info": s.get("error_info"),
                "last_update_time": s.get("last_update_time"),
            }
            for s in stages
        ],
    }

    atomic_write_json(output_dir / "progress.json", progress_payload)
    atomic_write_json(output_dir / "runtime_estimates.json", estimates)

    counts = summary["counts"]
    cur = summary.get("current_stage")
    last_ckpt = summary.get("last_checkpoint")
    pct_steps = (
        round(100.0 * summary["completed_steps"] / summary["total_steps"], 2)
        if summary["total_steps"] > 0
        else (100.0 if summary["total_stages"] > 0 and counts["COMPLETED"] == summary["total_stages"] else 0.0)
    )

    lines = [
        "# Project Progress",
        "",
        f"_Last updated (UTC): `{progress_payload['generated_utc']}`_",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Active Preset | `{preset or 'all'}` |",
        f"| Total Stages / Experiments | **{summary['total_stages']}** |",
        f"| Completed | **{counts['COMPLETED']}** |",
        f"| Running | **{counts['RUNNING']}** |",
        f"| Interrupted (Resumable) | **{counts['INTERRUPTED']}** |",
        f"| Failed (Retryable) | **{counts['FAILED']}** |",
        f"| Pending | **{counts['PENDING']}** |",
        f"| Completed Training Steps | **{summary['completed_steps']} / {summary['total_steps']}** ({pct_steps:.2f}%) |",
        f"| Accumulated Runtime | `{format_duration(summary['accumulated_runtime_sec'])}` ({summary['accumulated_runtime_sec']:.1f}s) |",
        f"| Measured Step Speed | `{estimates['measured_sec_per_step']} s/step` ({estimates['measurement_source']}) |",
        f"| Estimated Remaining Runtime | `{estimates['estimated_remaining_human']}` |",
    ]
    if cur:
        lines.append(
            f"| Current / Next Stage | `{cur['stage_name']}` ({cur['status']}, step {cur['step']}/{cur['target_steps']}) |"
        )
    else:
        lines.append("| Current / Next Stage | All stages complete |")

    if last_ckpt:
        lines.append(
            f"| Last Successful Checkpoint | `{last_ckpt['stage_name']}` step {last_ckpt['step']} (`{last_ckpt['path']}`, sha256=`{str(last_ckpt.get('sha256') or '')[:12]}`) |"
        )
    else:
        lines.append("| Last Successful Checkpoint | None yet |")

    lines.extend([
        "",
        "## Resume Command",
        "",
        "```powershell",
        resume_command,
        "```",
        "",
        "## Stage Status",
        "",
        "| # | Stage | Type | Status | Step | Runtime | Resume Mode | Retries | Notes |",
        "|---:|---|---|---|---:|---:|---|---:|---|",
    ])

    for idx, s in enumerate(stages, 1):
        step_text = (
            f"{int(s.get('last_completed_step') or 0)}/{int(s.get('target_steps') or 0)}"
            if int(s.get("target_steps") or 0) > 0
            else "-"
        )
        note = str(s.get("error_info") or "").replace("\n", " ").replace("|", "/")[:80]
        lines.append(
            f"| {idx} | `{s['stage_name']}` | `{s['stage_type']}` | **{s['status']}** | "
            f"{step_text} | {format_duration(float(s.get('accumulated_runtime_sec') or 0.0))} | "
            f"`{s.get('resume_mode') or 'fresh'}` | {int(s.get('retry_count') or 0)} | {note} |"
        )

    lines.append("")
    md_text = "\n".join(lines)
    atomic_write_text(output_dir / "PROJECT_PROGRESS.md", md_text)

    if update_workspace_docs and workspace_root is not None:
        ws = Path(workspace_root)
        work_dirs = ws / "work_dirs"
        work_dirs.mkdir(parents=True, exist_ok=True)
        atomic_write_json(work_dirs / "progress.json", progress_payload)
        atomic_write_json(work_dirs / "runtime_estimates.json", estimates)
        atomic_write_text(ws / "docs" / "PROJECT_PROGRESS.md", md_text)

    return progress_payload


def inspect_codex_session(workspace_root: Path, db_path: Path | None = None, preset: str | None = None) -> dict[str, Any]:
    """Session-startup helper implementing Section 8 (Codex session continuity).

    Verifies handoff documents, recovers stale RUNNING states in SQLite,
    and identifies completed, incomplete, and blocked tasks with the exact
    continuation command.
    """
    ws = Path(workspace_root)
    docs_status = {}
    for name in ("CODEX_HANDOFF.md", "PROJECT_PROGRESS.md", "DECISIONS.md"):
        path = ws / "docs" / name
        docs_status[name] = {"exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}

    resolved_db = Path(db_path) if db_path else (ws / "work_dirs" / "experiment_state.sqlite")
    if not resolved_db.is_file():
        return {
            "docs": docs_status,
            "registry_exists": False,
            "registry_path": str(resolved_db),
            "recovered_stale": [],
            "completed": [],
            "incomplete": [],
            "blocked": [],
            "recommended_command": "python scripts/run_suite.py --preset core --resume",
        }

    reg = ExperimentRegistry(resolved_db)
    recovered = reg.recover_stale_running()
    suite = reg.load_suite_state() or {}
    active_preset = preset or suite.get("preset")
    stages = reg.list_stages(preset=active_preset)
    completed = [s["stage_name"] for s in stages if s["status"] == "COMPLETED"]
    failed = [s["stage_name"] for s in stages if s["status"] == "FAILED"]
    incomplete = [s["stage_name"] for s in stages if s["status"] in ("PENDING", "INTERRUPTED", "RUNNING")]
    next_stage = next((s for s in stages if s["status"] != "COMPLETED"), None)
    cmd = suite.get("command") or f"python scripts/run_suite.py --preset {active_preset or 'core'} --resume"
    return {
        "docs": docs_status,
        "registry_exists": True,
        "registry_path": str(resolved_db),
        "active_preset": active_preset,
        "recovered_stale": [r["experiment_id"] for r in recovered],
        "completed": completed,
        "incomplete": incomplete,
        "blocked": failed,
        "next_stage": next_stage["stage_name"] if next_stage else None,
        "recommended_command": cmd,
    }

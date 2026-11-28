#!/usr/bin/env python3
"""Generate manuscript-style figures from *observed* training/results/calibration files."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def render(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall COCO AP bar charts by label fraction
    path = root / "statistics.json"
    if path.is_file():
        stat_obj = json.loads(path.read_text(encoding="utf8"))
        summary = stat_obj.get("summary", [])
        is_diag_only = False
        if not summary and stat_obj.get("diagnostic_partial_summary"):
            summary = stat_obj.get("diagnostic_partial_summary", [])
            is_diag_only = True
        by_fraction = defaultdict(list)
        for row in summary:
            by_fraction[row["fraction"]].append(row)
        for fraction, entries in by_fraction.items():
            entries.sort(key=lambda x: x["variant"])
            labels = [e["variant"] for e in entries]
            means = [100 * e["mean_AP"] for e in entries]
            deviations = [100 * (e["std_AP"] or 0.0) for e in entries]
            fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(labels)), 5))
            ax.bar(range(len(labels)), means, yerr=deviations, capsize=3)
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
            ax.set_ylabel("COCO AP @[0.50:0.95] (%)")
            suffix = " (DIAGNOSTIC PARTIAL VAL — NOT FOR PAPER)" if is_diag_only else " (mean +/- std)"
            ax.set_title(f"BDD100K {fraction}: observed results{suffix}")
            ax.set_ylim(bottom=0)
            fig.tight_layout()
            fname = f"ap_diagnostic_{fraction}.png" if is_diag_only else f"ap_{fraction}.png"
            fig.savefig(figure_dir / fname, dpi=180)
            plt.close(fig)

    run_root = root / "runs"
    if run_root.exists():
        # 2. Training loss curves
        for step_file in run_root.rglob("steps.jsonl"):
            try:
                lines = [json.loads(s) for s in step_file.read_text(encoding="utf8").splitlines() if s.strip()]
                if not lines:
                    continue
                steps = [r["step"] for r in lines]
                vals = [r["loss"] for r in lines]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(steps, vals, linewidth=0.8)
                ax.set_xlabel("Optimizer step")
                ax.set_ylabel("Training loss (not AP)")
                ax.set_title(str(step_file.parent.relative_to(run_root)))
                fig.tight_layout()
                key = str(step_file.parent.relative_to(run_root)).replace("\\", "_").replace("/", "_")
                fig.savefig(figure_dir / f"loss_{key}.png", dpi=150)
                plt.close(fig)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                print(f"Skipping malformed training log {step_file}: {exc}")

        # 3. Conformal Risk Control verification plots (Risk vs Target alpha, Thresholds, Pooled Recall & FDR)
        for conf_file in run_root.rglob("conformal_calibration.jsonl"):
            try:
                rows = [json.loads(s) for s in conf_file.read_text(encoding="utf8").splitlines() if s.strip()]
                if not rows:
                    continue
                key = str(conf_file.parent.relative_to(run_root)).replace("\\", "_").replace("/", "_")
                steps = [r["step"] for r in rows]
                alpha_s = float(rows[0].get("alpha_strict", 0.08))
                risk_metric = str(rows[0].get("risk_metric", "fdr"))
                has_recall = any(
                    r.get("regimes", {}).get("clear", {}).get("pooled_empirical_recall") is not None
                    for r in rows
                )

                ncols = 3 if has_recall else 2
                fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5))
                ax1, ax2 = axes[0], axes[1]
                ax3 = axes[2] if has_recall else None
                obs_risks = [alpha_s]

                for regime, color in (("clear", "#1f77b4"), ("degraded", "#d62728")):
                    emp_r = [r.get("regimes", {}).get(regime, {}).get("empirical_weighted_risk") for r in rows]
                    ub_r = [r.get("regimes", {}).get(regime, {}).get("conformal_upper_bound") for r in rows]
                    tau_h = [r.get("regimes", {}).get(regime, {}).get("tau_high") for r in rows]
                    tau_l = [r.get("regimes", {}).get(regime, {}).get("tau_low") for r in rows]
                    for v in emp_r + ub_r:
                        if v is not None:
                            obs_risks.append(float(v))
                    if any(v is not None for v in emp_r):
                        ax1.plot(steps, [v if v is not None else float("nan") for v in emp_r],
                                 marker="o", markersize=3, color=color, label=f"{regime} empirical risk")
                    if any(v is not None for v in ub_r):
                        ax1.plot(steps, [v if v is not None else float("nan") for v in ub_r],
                                 linestyle="--", linewidth=1.0, color=color, alpha=0.6, label=f"{regime} conformal UB")
                    if any(v is not None for v in tau_h):
                        ax2.plot(steps, [v if v is not None else float("nan") for v in tau_h],
                                 marker="o", markersize=3, color=color, label=f"{regime} tau_high (Zone 1)")
                    if any(v is not None for v in tau_l):
                        ax2.plot(steps, [v if v is not None else float("nan") for v in tau_l],
                                 linestyle=":", color=color, label=f"{regime} tau_low (Zone 2)")
                    if ax3 is not None:
                        rec = [r.get("regimes", {}).get(regime, {}).get("pooled_empirical_recall") for r in rows]
                        fdr = [r.get("regimes", {}).get(regime, {}).get("pooled_empirical_fdr") for r in rows]
                        if any(v is not None for v in rec):
                            ax3.plot(steps, [v if v is not None else float("nan") for v in rec],
                                     marker="o", markersize=3, color=color, label=f"{regime} recall")
                        if any(v is not None for v in fdr):
                            ax3.plot(steps, [v if v is not None else float("nan") for v in fdr],
                                     linestyle="--", color=color, alpha=0.65, label=f"{regime} pooled FDR")

                ax1.axhline(alpha_s, color="black", linestyle="-.", linewidth=1.2, label=f"Target alpha={alpha_s}")
                ax1.set_xlabel("Optimizer step")
                ylabel = "Weighted FP-per-GT Risk" if risk_metric == "monotone_fp_per_gt" else "Weighted Pseudo-Label FDR Risk"
                ax1.set_ylabel(ylabel)
                ax1.set_title("Conformal Risk Control Verification")
                ymax = max(0.25, min(1.0, max(alpha_s * 2.5, max(obs_risks) * 1.25)))
                ax1.set_ylim(0, ymax)
                ax1.legend(fontsize=8)

                ax2.set_xlabel("Optimizer step")
                ax2.set_ylabel("Reliability Threshold")
                ax2.set_title("Adaptive Conformal Tri-Zone Thresholds")
                ax2.set_ylim(0, 1.0)
                ax2.legend(fontsize=8)

                if ax3 is not None:
                    ax3.set_xlabel("Optimizer step")
                    ax3.set_ylabel("Rate")
                    ax3.set_title("Calibration Recall & Pooled FDR")
                    ax3.set_ylim(0, 1.0)
                    ax3.legend(fontsize=8)

                fig.suptitle(f"CoRe-SSOD Calibration: {conf_file.parent.relative_to(run_root)}")
                fig.tight_layout()
                fig.savefig(figure_dir / f"conformal_{key}.png", dpi=180)
                plt.close(fig)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                print(f"Skipping malformed conformal log {conf_file}: {exc}")

        # 4. Stratified Adverse-Condition AP (daytime vs night vs rainy/snowy/foggy)
        cond_by_frac = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        key_conditions = ("timeofday/daytime", "timeofday/night", "weather/rainy", "weather/snowy", "weather/foggy")
        for cond_file in run_root.rglob("analysis/conditions.json"):
            try:
                rel_parts = cond_file.parent.parent.relative_to(run_root).parts
                if len(rel_parts) != 2:
                    continue
                split_tag, variant = rel_parts
                if variant == "warmup" or "_s" not in split_tag:
                    continue
                fraction = split_tag.split("_s", 1)[0]
                data = json.loads(cond_file.read_text(encoding="utf8")).get("metrics", {})
                for ck in key_conditions:
                    ap = (data.get(ck) or {}).get("AP")
                    if ap is not None:
                        cond_by_frac[fraction][variant][ck].append(100.0 * float(ap))
            except (ValueError, KeyError, json.JSONDecodeError):
                continue

        for fraction, by_var in cond_by_frac.items():
            variants = sorted(by_var.keys())
            present_conds = [c for c in key_conditions if any(c in by_var[v] for v in variants)]
            if not variants or not present_conds:
                continue
            fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(variants)), 5))
            x = list(range(len(variants)))
            width = 0.8 / max(1, len(present_conds))
            for idx, ck in enumerate(present_conds):
                vals = [
                    (sum(by_var[v][ck]) / len(by_var[v][ck])) if by_var[v].get(ck) else 0.0
                    for v in variants
                ]
                offsets = [xi + (idx - 0.5 * (len(present_conds) - 1)) * width for xi in x]
                ax.bar(offsets, vals, width=width, label=ck.split("/")[-1])
            ax.set_xticks(x, variants, rotation=40, ha="right")
            ax.set_ylabel("Stratified COCO AP (%)")
            ax.set_title(f"BDD100K {fraction}: Condition-Stratified Robustness")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figure_dir / f"conditions_{fraction}.png", dpi=180)
            plt.close(fig)

    return figure_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    args = p.parse_args()
    print("Figures:", render(args.root))

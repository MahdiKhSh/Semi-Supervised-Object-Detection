#!/usr/bin/env python3
"""Generate manuscript-style figures from *observed* training/results files."""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def render(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = root / "statistics.json"
    if path.is_file():
        summary = json.loads(path.read_text(encoding="utf8")).get("summary", [])
        by_fraction = defaultdict(list)
        for row in summary:
            by_fraction[row["fraction"]].append(row)
        for fraction, entries in by_fraction.items():
            entries.sort(key=lambda x: x["variant"])
            labels = [e["variant"] for e in entries]
            means = [100 * e["mean_AP"] for e in entries]
            deviations = [100 * (e["std_AP"] or 0.) for e in entries]
            fig, ax = plt.subplots(figsize=(max(8, .65 * len(labels)), 5))
            ax.bar(range(len(labels)), means, yerr=deviations, capsize=3)
            ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
            ax.set_ylabel("COCO AP @[0.50:0.95] (%)")
            ax.set_title(f"BDD100K {fraction}: observed results; check number of seeds")
            ax.set_ylim(bottom=0)
            fig.tight_layout()
            fig.savefig(figure_dir / f"ap_{fraction}.png", dpi=180)
            plt.close(fig)
    run_root = root / "runs"
    if run_root.exists():
        for path in run_root.rglob("steps.jsonl"):
            try:
                lines = [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s.strip()]
                if not lines:
                    continue
                steps = [r["step"] for r in lines]
                vals = [r["loss"] for r in lines]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(steps, vals, linewidth=.8)
                ax.set_xlabel("Optimizer step")
                ax.set_ylabel("Training loss (not AP)")
                ax.set_title(str(path.parent.relative_to(run_root)))
                fig.tight_layout()
                key = str(path.parent.relative_to(run_root)).replace("\\", "_").replace("/", "_")
                fig.savefig(figure_dir / f"loss_{key}.png", dpi=150)
                plt.close(fig)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                print(f"Skipping malformed training log {path}: {exc}")
    return figure_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    args = p.parse_args()
    print("Figures:", render(args.root))

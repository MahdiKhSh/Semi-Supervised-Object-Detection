#!/usr/bin/env python3
"""Analyze genuine multi-seed results; never fill unobserved AP.

Reports mean±sample std per variant/label fraction and paired differences
for matched seeds. Bootstrap intervals are exploratory with only 3 seeds.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
from statistics import mean, stdev


def bootstrap_ci(values, seed=42, n_samples=4000):
    if not values:
        return None
    rng = random.Random(seed)
    draws = sorted(mean(rng.choice(values) for _ in values) for _ in range(n_samples))
    return [draws[int(.025 * (n_samples - 1))], draws[int(.975 * (n_samples - 1))]]


def load_rows(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("AP") not in (None, "", "None")
            and str(r.get("partial_validation", "")).lower() not in ("true", "1")]


def stats_for_rows(rows):
    groups = defaultdict(list)
    paired = defaultdict(dict)
    for item in rows:
        parts = item["experiment"].split("/")
        if len(parts) != 2:
            continue
        split, variant_name = parts
        if variant_name == "warmup":
            continue
        prefix, seed_text = split.split("_s", 1)
        score = float(item["AP"])
        seed = int(seed_text)
        groups[(prefix, variant_name)].append((seed, score))
        paired[(prefix, seed)][variant_name] = score
    summary = []
    for (fraction, variant), vals in sorted(groups.items()):
        scores = [v for _, v in vals]
        summary.append({"fraction": fraction, "variant": variant,
                        "n_seeds": len(scores),
                        "mean_AP": mean(scores),
                        "std_AP": stdev(scores) if len(scores) > 1 else None,
                        "seeds": sorted(int(s) for s, _ in vals)})
    comparisons = []
    for fraction in sorted({p for p, _ in groups}):
        for baseline in ("supervised", "mean_teacher"):
            diffs = []
            for (fr, seed), by_name in paired.items():
                if fr == fraction and "full" in by_name and baseline in by_name:
                    diffs.append((seed, by_name["full"] - by_name[baseline]))
            if not diffs:
                continue
            delta = [d for _, d in diffs]
            comparisons.append({
                "fraction": fraction, "comparison": "full_minus_" + baseline,
                "n_pairs": len(delta), "paired_mean_delta_AP": mean(delta),
                "paired_std_delta_AP": stdev(delta) if len(delta) > 1 else None,
                "bootstrap_95_ci": bootstrap_ci(delta),
                "paired_seeds": [s for s, _ in diffs],
                "warning": "Three paired seeds give weak uncertainty estimates."
            })
    return summary, comparisons


def write_report(source, output):
    source = Path(source)
    output = Path(output)
    rows = load_rows(source)
    summary, comparisons = stats_for_rows(rows)
    output.mkdir(parents=True, exist_ok=True)
    package = {"completed_run_rows": len(rows), "summary": summary,
               "paired_comparisons": comparisons,
               "note": "Synthetic or partial validation results are diagnostics, not paper outcomes."}
    (output / "statistics.json").write_text(json.dumps(package, indent=2), encoding="utf8")
    md = ["# SSOD experiment results", "",
          f"Recorded completed metrics: {len(rows)}", "",
          "**Do not report synthetic or partial-validation AP as paper results.**", "",
          "| Fraction | Variant | Seeds | Mean AP | Std AP |",
          "|---|---|---:|---:|---:|"]
    for r in summary:
        std_text = f"{r['std_AP']:.4f}" if r['std_AP'] is not None else "N/A"
        md.append(f"| {r['fraction']} | {r['variant']} | {r['n_seeds']} | "
                  f"{r['mean_AP']:.4f} | {std_text} |")
    md.extend(["", "## Paired full-model differences", "",
               "| Fraction | Comparison | Pairs | Mean delta AP | 95% bootstrap interval |",
               "|---|---|---:|---:|---|"])
    for p in comparisons:
        ci = p["bootstrap_95_ci"]
        interval = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "N/A"
        md.append(f"| {p['fraction']} | {p['comparison']} | {p['n_pairs']} | "
                  f"{p['paired_mean_delta_AP']:.4f} | {interval} |")
    md.extend(["", "Interpret intervals conservatively: few seeds and many exploratory "
               "comparisons increase false-discovery risk.", ""])
    (output / "PAPER_RESULTS.md").write_text("\n".join(md), encoding="utf8")
    return package


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = write_report(args.results, args.output)
    print(f"Wrote statistical report for {report['completed_run_rows']} measured rows")


if __name__ == "__main__":
    main()

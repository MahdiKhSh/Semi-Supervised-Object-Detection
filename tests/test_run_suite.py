from pathlib import Path
from scripts.run_suite import experiment_plan, generate_jobs, parse_args


def test_plan_matrices_have_warmup_and_unique_names(tmp_path):
    for preset in ("smoke", "pilot", "paper"):
        args = parse_args(["--preset", preset, "--output", str(tmp_path)])
        jobs = generate_jobs(args)
        names = [job["name"] for job in jobs]
        assert len(names) == len(set(names))
        assert sum(x["type"] == "prepare" for x in jobs) == (6 if preset == "paper" else 1)
        assert sum("warmup" in x["name"] for x in jobs) == (6 if preset == "paper" else 1)
        assert any("--variant" in x["command"] and "full" in x["command"] for x in jobs)
        assert any("/supervised" in x["name"] for x in jobs)
        for job in jobs:
            if job["type"] == "train" and "warmup" not in job["name"]:
                assert "--init-checkpoint" in job["command"]


def test_threshold_ablation_is_explicit(tmp_path):
    args = parse_args(["--preset", "pilot", "--output", str(tmp_path)])
    jobs = generate_jobs(args)
    variant_jobs = {j["name"].split("/")[-1]: j["command"] for j in jobs if j["type"] == "train"}
    assert "--pseudo-threshold" in variant_jobs["threshold05"]
    assert "0.5" in variant_jobs["threshold05"]
    assert "--box-weight" in variant_jobs["no_box"]
    assert "0.0" in variant_jobs["no_box"]


def test_paper_has_all_seeds_and_fractions(tmp_path):
    args = parse_args(["--preset", "paper", "--output", str(tmp_path)])
    jobs = generate_jobs(args)
    names = [j["name"] for j in jobs]
    for pct in ("05pct", "10pct"):
        for seed in (42, 123, 2026):
            assert f"{pct}_s{seed}/warmup" in names
            assert f"{pct}_s{seed}/no_localization" in names
            assert f"{pct}_s{seed}/no_extra_consistency" in names

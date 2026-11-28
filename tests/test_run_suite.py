from pathlib import Path
from scripts.run_suite import experiment_plan, generate_jobs, parse_args


def test_plan_matrices_have_warmup_and_unique_names(tmp_path):
    for preset in ("smoke", "pilot", "core", "paper"):
        args = parse_args(["--preset", preset, "--output", str(tmp_path)])
        jobs = generate_jobs(args)
        names = [job["name"] for job in jobs]
        assert len(names) == len(set(names))
        assert sum(x["type"] == "prepare" for x in jobs) == (6 if preset in ("core", "paper") else 1)
        assert sum("warmup" in x["name"] for x in jobs) == (6 if preset in ("core", "paper") else 1)
        assert any("--variant" in x["command"] and "full" in x["command"] for x in jobs)
        assert any("--variant" in x["command"] and "supervised" in x["command"] for x in jobs)
        for job in jobs:
            if job["type"] == "train" and "warmup" not in job["name"]:
                assert "--init-checkpoint" in job["command"]


def test_a100_reuses_input_paths_with_a_fresh_training_schedule(tmp_path):
    import json

    old = parse_args(["--preset", "q1_pilot", "--batch-size", "40", "--train-steps", "300",
                      "--train-images", str(tmp_path / "server images")])
    path = tmp_path / "experiment_plan.json"
    path.write_text(json.dumps({"jobs": generate_jobs(old)}), encoding="utf8")
    args = parse_args(["--preset", "q1_a100", "--dataset-from-plan", str(path)])
    jobs = generate_jobs(args)
    assert args.train_images == old.train_images and args.batch_size * args.grad_accum_steps == 8
    moved = parse_args(["--preset", "q1_a100", "--dataset-from-plan", str(path),
                        "--val-images", str(tmp_path / "moved")])
    assert moved.val_images == tmp_path / "moved" and moved.train_images == old.train_images
    assert [job["steps"] for job in jobs[1:]] == [1500] + [3000] * 5
    for job in jobs[1:]:
        assert "--reserve-calibration" in job["command"]
        assert "--eval-max-images" not in job["command"]
    assert {job["variant"] for job in jobs[2:]} == {
        "supervised", "mean_teacher", "soft_teacher", "consistent_teacher", "core_ssod"
    }
    # Legacy experiment_plan.json where --train-images/--val-images lived only on train jobs
    legacy_jobs = json.loads(path.read_text(encoding="utf8"))["jobs"]
    prep_cmd = legacy_jobs[0]["command"]
    for drop_flag in ("--train-images", "--val-images"):
        if drop_flag in prep_cmd:
            idx = prep_cmd.index(drop_flag)
            del prep_cmd[idx:idx + 2]
    legacy_path = tmp_path / "legacy_plan.json"
    legacy_path.write_text(json.dumps({"jobs": legacy_jobs}), encoding="utf8")
    legacy_args = parse_args(["--preset", "q1_a100", "--dataset-from-plan", str(legacy_path)])
    assert legacy_args.train_images == old.train_images and legacy_args.val_images == old.val_images


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

def test_core_matrix_size_and_seed_scoped_ablations(tmp_path):
    args = parse_args(["--preset", "core", "--output", str(tmp_path)])
    jobs = generate_jobs(args)
    assert len(jobs) == 52
    assert any(j["name"] == "05pct_s42/no_box" for j in jobs)
    assert not any(j["name"] == "05pct_s123/no_box" for j in jobs)
    assert any(j["name"] == "10pct_s2026/full" for j in jobs)

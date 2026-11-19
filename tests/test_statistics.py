from scripts.report_results import bootstrap_ci, stats_for_rows


def test_paired_seeds_and_std():
    rows = []
    for seed, base, full in [(42, .2, .25), (123, .22, .27), (2026, .21, .28)]:
        for name, score in [("supervised", base), ("full", full)]:
            rows.append({"experiment": f"05pct_s{seed}/{name}", "AP": score})
    summaries, comparison = stats_for_rows(rows)
    assert len(summaries) == 2
    assert all(r["n_seeds"] == 3 for r in summaries)
    assert len(comparison) == 1
    assert comparison[0]["comparison"] == "full_minus_supervised"
    assert comparison[0]["paired_mean_delta_AP"] > 0


def test_bootstrap_is_deterministic():
    assert bootstrap_ci([.01, .02, .03]) == bootstrap_ci([.01, .02, .03])
    assert bootstrap_ci([]) is None

import pandas as pd

from mp024.data_checks import (check_duplicate_rows, check_stuck_sensor, run_all_checks, typo_mask)


def test_all_injected_problems_are_found(dirty_train, clean_frame, holdout_like, params):
    issues = run_all_checks(dirty_train, holdout_like, clean_frame, params)["data_issues"]
    assert issues["leaking_column"] == {"present": True, "column": "kg_weighed_at_factory"}
    assert issues["unit_mixup"]["column"] == "rain_mm_week"
    assert issues["unit_mixup"]["group_value"] == "River Side"
    assert issues["unit_mixup"]["factor"] == 0.1          # wrong value = right value x 0.1
    assert issues["sentinel_value"] == {"present": True, "column": "worker_exp_yr", "value": 9999, "count": 15}
    assert issues["target_typos"] == {"present": True, "count": 6}
    assert issues["spelling_variants"] == {"present": True, "column": "flush", "true_levels": 3}


def test_absent_problems_are_reported_absent(dirty_train, clean_frame, holdout_like, params):
    issues = run_all_checks(dirty_train, holdout_like, clean_frame, params)["data_issues"]
    assert issues["duplicate_rows"]["present"] is False and issues["duplicate_rows"]["count"] is None
    assert issues["stuck_sensor"]["present"] is False


def test_a_clean_file_has_no_problems(clean_frame, holdout_like, params):
    # nothing injected -> only the structural leak (target-like column absent in holdout) may be reported
    issues = run_all_checks(clean_frame, holdout_like, clean_frame, params)["data_issues"]
    for name in ["unit_mixup", "sentinel_value", "duplicate_rows", "stuck_sensor", "target_typos", "spelling_variants"]:
        assert issues[name]["present"] is False, name


def test_duplicates_count_extra_copies_only(clean_frame):
    doubled = pd.concat([clean_frame, clean_frame.head(7)], ignore_index=True)
    finding, _ = check_duplicate_rows(doubled, "record_id")
    assert finding == {"present": True, "count": 7}


def test_stuck_sensor_reports_first_and_last_date(clean_frame):
    stuck = clean_frame.copy()
    window = (stuck.date >= "2026-02-01") & (stuck.date <= "2026-02-10")
    stuck.loc[window, "temp_c"] = 21.5
    finding, _ = check_stuck_sensor(stuck, "date", "record_id", ["kg_per_worker", "record_id"])
    assert finding == {"present": True, "column": "temp_c", "start_date": "2026-02-01", "end_date": "2026-02-10"}


def test_typo_mask_flags_only_the_ten_times_values():
    y = pd.Series([20.0 + i * 0.3 for i in range(100)] + [250.0, 310.0])
    mask = typo_mask(y)
    assert mask.sum() == 2 and mask.iloc[-1] and mask.iloc[-2] and not mask.iloc[:100].any()

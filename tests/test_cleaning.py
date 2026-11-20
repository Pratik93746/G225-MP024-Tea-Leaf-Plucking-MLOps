import numpy as np
import pandas as pd

from mp024.cleaning import canonical_level, clean_training_frame, map_to_levels
from mp024.data_checks import run_all_checks

LEVELS = ["first", "second", "monsoon"]


def test_spelling_variants_map_to_real_levels():
    raw = pd.Series(["First", " first", "FIRST", "fir.", "sec.", "SECOND", " monsoon", "mon.", "Monsoon"])
    assert list(map_to_levels(raw, LEVELS)) == ["first", "first", "first", "first", "second", "second",
                                                 "monsoon", "monsoon", "monsoon"]


def test_unknown_spelling_becomes_missing():
    assert pd.isna(canonical_level("winter", LEVELS))
    assert pd.isna(canonical_level("", LEVELS))


def test_cleaning_repairs_every_detected_problem(dirty_train, clean_frame, holdout_like, params):
    issues = run_all_checks(dirty_train, holdout_like, clean_frame, params)["data_issues"]
    cleaned = clean_training_frame(dirty_train, issues, {"flush": LEVELS}, params)

    assert "kg_weighed_at_factory" not in cleaned.columns                       # leak dropped
    assert set(cleaned.flush.unique()) == set(LEVELS)                            # spellings fixed
    assert cleaned.worker_exp_yr.isna().sum() == 15 and (cleaned.worker_exp_yr.dropna() <= 30).all()  # sentinel -> NaN
    assert cleaned.kg_per_worker.max() < 60                                      # typos divided by 10
    river = cleaned[cleaned.section == "River Side"].rain_mm_week
    assert river.max() > 200                                                     # cm converted back to mm
    assert len(cleaned) == len(dirty_train)                                      # nothing was thrown away
    assert cleaned.date.is_monotonic_increasing

"""Shared test data. Tests build small frames in memory - the real CSVs are not in git."""
import numpy as np
import pandas as pd
import pytest

SECTIONS = ["Upper Garden", "Lower Garden", "River Side"]


def make_clean_frame(n: int = 600, seed: int = 0, start: str = "2026-01-01") -> pd.DataFrame:
    """A clean frame shaped like later.csv, with a simple known relationship for the target."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n // 5, freq="D").repeat(5)[:n]
    df = pd.DataFrame({
        "record_id": [f"R{i:05d}" for i in range(n)],
        "date": days.strftime("%Y-%m-%d"),
        "worker_exp_yr": rng.integers(0, 31, n),
        "rain_mm_week": rng.integers(0, 301, n).astype(float),
        "slope_deg": rng.integers(0, 36, n),
        "temp_c": rng.uniform(10, 35, n).round(1),
        "bush_age_yr": rng.integers(3, 81, n),
        "hours_worked": rng.uniform(4, 10, n).round(1),
        "flush": rng.choice(["first", "second", "monsoon"], n),
        "section": rng.choice(SECTIONS, n),
    })
    df["kg_per_worker"] = (3.5 * df.hours_worked + 0.25 * df.worker_exp_yr - 0.2 * df.slope_deg
                           + rng.normal(0, 1, n)).round(1)
    df["kg_weighed_at_factory"] = (df.kg_per_worker + rng.normal(0, 0.3, n)).round(1)  # the leak
    return df


def make_dirty_train(n: int = 600, seed: int = 1) -> pd.DataFrame:
    """A train-like frame with five of the seven problems injected (no duplicates, no stuck sensor)."""
    df = make_clean_frame(n, seed)
    df.loc[df.section == "River Side", "rain_mm_week"] /= 10            # unit mix-up (cm instead of mm)
    df.loc[df.index[:15], "worker_exp_yr"] = 9999                        # sentinel
    df.loc[df.index[100:106], "kg_per_worker"] *= 10                     # six target typos
    variants = {"first": ["First", " first", "fir."], "second": ["SECOND", "sec.", " second"],
                "monsoon": ["Monsoon", "mon.", "MONSOON"]}
    rng = np.random.default_rng(seed)
    swap = rng.random(n) < 0.4
    df.loc[swap, "flush"] = [rng.choice(variants[v]) for v in df.loc[swap, "flush"]]
    return df


@pytest.fixture
def clean_frame():
    return make_clean_frame()


@pytest.fixture
def dirty_train():
    return make_dirty_train()


@pytest.fixture
def holdout_like(clean_frame):
    return clean_frame.drop(columns=["kg_per_worker", "kg_weighed_at_factory"]).head(50)


@pytest.fixture
def params():
    from mp024.config import load_params
    return load_params()

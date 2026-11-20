"""Cleaning: undo each data problem that data_checks.py found.

The cleaning is driven by the *findings* (``issues``), so it fixes exactly what was detected:
  * leaking column     -> dropped (the model must never see it)
  * unit mix-up        -> the affected group's values are converted back (divide by the factor)
  * sentinel value     -> replaced by NaN (the model pipeline imputes it later)
  * duplicate rows     -> extra copies dropped
  * stuck sensor       -> readings inside the stuck window set to NaN
  * target typos       -> divided by 10 (they were typed with one zero too many)
  * spelling variants  -> mapped to the real categories
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def normalise_text(value) -> str:
    """lower-case, remove everything that is not a letter or digit: ' First.' -> 'first'."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def canonical_level(value, levels: list[str]):
    """Map one raw spelling to a real category, or NaN if it cannot be matched.

    Exact match after normalising (``SECOND``, `` second``) first, then abbreviations
    (``sec.`` -> ``second``) but only when the abbreviation matches exactly one category.
    """
    key = normalise_text(value)
    if not key:
        return np.nan
    by_key = {normalise_text(level): level for level in levels}
    if key in by_key:
        return by_key[key]
    matches = [level for k, level in by_key.items() if k.startswith(key)]
    return matches[0] if len(matches) == 1 else np.nan


def map_to_levels(values: pd.Series, levels: list[str]) -> pd.Series:
    return values.map(lambda v: canonical_level(v, levels))


def clean_training_frame(df: pd.DataFrame, issues: dict, clean_levels: dict[str, list[str]], params: dict) -> pd.DataFrame:
    """Return a cleaned copy of train.csv, sorted by date."""
    from mp024.data_checks import typo_mask  # local import: data_checks imports this module

    d, c = params["data"], params["data_checks"]
    out = df.copy()

    # 4. duplicates first, so later steps never count a row twice
    if issues["duplicate_rows"]["present"]:
        out = out.drop_duplicates().drop_duplicates(subset=[col for col in out.columns if col != d["id_column"]])

    # 7. spelling variants
    spelling = issues["spelling_variants"]
    if spelling["present"]:
        col = spelling["column"]
        out[col] = map_to_levels(out[col], clean_levels[col])

    # 2. unit mix-up: wrong = right x factor  ->  right = wrong / factor
    unit = issues["unit_mixup"]
    if unit["present"]:
        rows = out[d["group_column"]] == unit["group_value"]
        out.loc[rows, unit["column"]] = out.loc[rows, unit["column"]] / unit["factor"]

    # 3. sentinel -> missing
    sentinel = issues["sentinel_value"]
    if sentinel["present"]:
        out[sentinel["column"]] = out[sentinel["column"]].replace(sentinel["value"], np.nan)

    # 5. stuck sensor -> missing inside the stuck window
    stuck = issues["stuck_sensor"]
    if stuck["present"]:
        window = (out[d["date_column"]] >= stuck["start_date"]) & (out[d["date_column"]] <= stuck["end_date"])
        out.loc[window, stuck["column"]] = np.nan

    # 6. target typos -> /10
    if issues["target_typos"]["present"]:
        bad = typo_mask(out[d["target"]], c["typo_gap_ratio"], c["typo_factor"])
        out.loc[bad, d["target"]] = out.loc[bad, d["target"]] / c["typo_factor"]

    # 1. the leaking column is dropped
    leak = issues["leaking_column"]
    if leak["present"] and leak["column"] in out.columns:
        out = out.drop(columns=[leak["column"]])

    return out.sort_values([d["date_column"], d["id_column"]], kind="stable").reset_index(drop=True)


def clean_reference_frame(df: pd.DataFrame, issues: dict, clean_levels: dict[str, list[str]], params: dict) -> pd.DataFrame:
    """later.csv / holdout_features.csv are clean already: only normalise spellings, drop the leak column, sort by date."""
    d = params["data"]
    out = df.copy()
    spelling = issues["spelling_variants"]
    if spelling["present"] and spelling["column"] in out.columns:
        out[spelling["column"]] = map_to_levels(out[spelling["column"]], clean_levels[spelling["column"]])
    leak = issues["leaking_column"]
    if leak["present"] and leak["column"] in out.columns:
        out = out.drop(columns=[leak["column"]])
    return out.sort_values([d["date_column"], d["id_column"]], kind="stable").reset_index(drop=True)

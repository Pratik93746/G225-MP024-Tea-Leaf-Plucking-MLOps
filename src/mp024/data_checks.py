"""The seven data-problem detectors for train.csv.

Every answer in findings.yaml comes from these functions looking at the actual file.
Each detector returns ``(finding, evidence)``:
  * ``finding``  - exactly the keys that findings.yaml asks for
  * ``evidence`` - extra numbers that justify the finding (printed and saved in reports/data_issues.json)

Run it:  PYTHONPATH=src python -m mp024.data_checks
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mp024.cleaning import normalise_text
from mp024.config import load_params, resolve

# Typical "I do not know the value" numbers people type into a spreadsheet.
CANDIDATE_SENTINELS = [-1, -9, -99, -999, -9999, 9, 99, 999, 9999, 99999]
# Typical unit conversions (mm<->cm 10, years<->months 12, hours<->minutes 60, ...).
KNOWN_FACTORS = [10, 12, 60, 100, 1000]


def _numeric_columns(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]


# --------------------------------------------------------------------------- 1. leaking column
def check_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame, target: str, min_spearman: float = 0.95):
    """A column that is missing from holdout_features.csv AND agrees almost exactly with the target."""
    missing_in_holdout = [c for c in train.columns if c not in holdout.columns and c != target]
    best, evidence = None, {"missing_from_holdout": missing_in_holdout, "spearman_with_target": {}}
    for col in missing_in_holdout:
        if not pd.api.types.is_numeric_dtype(train[col]):
            continue
        # Spearman = rank correlation, so a few typo'd targets (10x too big) cannot hide the leak
        rho = float(train[col].corr(train[target], method="spearman"))
        evidence["spearman_with_target"][col] = round(rho, 4)
        if rho >= min_spearman and (best is None or rho > best[1]):
            best = (col, rho)
    return {"present": best is not None, "column": best[0] if best else None}, evidence


# --------------------------------------------------------------------------- 2. unit mix-up
def _snap_factor(ratio: float) -> float:
    """Snap a measured ratio (wrong / right) to the nearest exact unit conversion."""
    options = KNOWN_FACTORS + [1.0 / k for k in KNOWN_FACTORS]
    best = min(options, key=lambda o: abs(np.log(ratio) - np.log(o)))
    if abs(np.log(ratio) - np.log(best)) > np.log(1.5):  # nothing close -> report the measured ratio
        return float(round(ratio, 4))
    return float(round(best, 6))


def check_unit_mixup(df: pd.DataFrame, group_col: str, exclude: list[str], ratio_trigger: float = 5.0):
    """One value of the group column has a column that is ~10/12/60/100/1000x bigger or smaller."""
    best, evidence = None, {"group_median_ratio_vs_other_groups": {}}
    for col in _numeric_columns(df, exclude + [group_col]):
        medians = df.groupby(group_col)[col].median()
        baseline = medians.median()
        if not baseline or baseline <= 0:
            continue
        for group, med in medians.items():
            if med <= 0:
                continue
            ratio = float(med / baseline)
            deviation = abs(np.log(ratio))
            evidence["group_median_ratio_vs_other_groups"][f"{col} | {group}"] = round(ratio, 3)
            if deviation >= np.log(ratio_trigger) and (best is None or deviation > best[0]):
                best = (deviation, col, group, ratio)
    if best is None:
        return {"present": False, "column": None, "group_value": None, "factor": None}, evidence
    _, col, group, ratio = best
    # wrong value = right value x factor
    return {"present": True, "column": col, "group_value": str(group), "factor": _snap_factor(ratio)}, evidence


# --------------------------------------------------------------------------- 3. sentinel value
def check_sentinel_value(df: pd.DataFrame, exclude: list[str], min_count: int = 3):
    """A special number (-1, -999, 9999 ...) typed instead of a blank: it repeats and lies outside the normal range."""
    best, evidence = None, {"candidates_found": {}}
    for col in _numeric_columns(df, exclude):
        series = df[col]
        for sentinel in CANDIDATE_SENTINELS:
            n = int((series == sentinel).sum())
            if n < min_count:
                continue
            rest = series[series != sentinel]
            low, high = rest.quantile(0.01), rest.quantile(0.99)
            outside = sentinel < low or sentinel > high  # impossible next to the other values
            evidence["candidates_found"][f"{col}={sentinel}"] = {"count": n, "outside_normal_range": bool(outside)}
            if outside and (best is None or n > best[2]):
                best = (col, sentinel, n)
    if best is None:
        return {"present": False, "column": None, "value": None, "count": None}, evidence
    return {"present": True, "column": best[0], "value": best[1], "count": best[2]}, evidence


# --------------------------------------------------------------------------- 4. duplicate rows
def check_duplicate_rows(df: pd.DataFrame, id_col: str):
    """Number of EXTRA copies (rows you would delete)."""
    as_given = int(df.duplicated().sum())
    ignoring_id = int(df.drop(columns=[id_col]).duplicated().sum()) if id_col in df.columns else as_given
    count = max(as_given, ignoring_id)
    evidence = {"duplicated_rows_as_given": as_given, "duplicated_rows_ignoring_record_id": ignoring_id,
                "duplicated_record_ids": int(df[id_col].duplicated().sum()) if id_col in df.columns else None}
    return {"present": count > 0, "count": count if count > 0 else None}, evidence


# --------------------------------------------------------------------------- 5. stuck sensor
def check_stuck_sensor(df: pd.DataFrame, date_col: str, id_col: str, exclude: list[str],
                       min_rows: int = 10, min_days: int = 3):
    """A column that keeps the very same value for a long run of rows spanning several days."""
    ordered = df.sort_values([date_col, id_col], kind="stable").reset_index(drop=True)
    best, evidence = None, {"longest_run_per_column": {}}
    for col in _numeric_columns(ordered, exclude):
        series = ordered[col]
        run_id = (series != series.shift()).cumsum()
        runs = ordered.groupby(run_id).agg(value=(col, "first"), rows=(col, "size"),
                                           days=(date_col, "nunique"),
                                           start=(date_col, "min"), end=(date_col, "max"))
        longest = runs.sort_values("rows", ascending=False).iloc[0]
        evidence["longest_run_per_column"][col] = {"value": float(longest["value"]), "rows": int(longest["rows"]),
                                                   "days": int(longest["days"])}
        if longest["rows"] >= min_rows and longest["days"] >= min_days:
            if best is None or longest["rows"] > best[1]:
                best = (col, int(longest["rows"]), str(longest["start"]), str(longest["end"]))
    if best is None:
        return {"present": False, "column": None, "start_date": None, "end_date": None}, evidence
    return {"present": True, "column": best[0], "start_date": best[2], "end_date": best[3]}, evidence


# --------------------------------------------------------------------------- 6. target typos
def typo_mask(target: pd.Series, gap_ratio: float = 2.0, factor: float = 10.0, max_share: float = 0.10) -> pd.Series:
    """True for rows whose target was typed ``factor`` times too big.

    Sort the target: typos sit far above everybody else with a big empty gap below them.
    We look for the largest jump between neighbouring sorted values in the upper half; if it is
    at least ``gap_ratio``, everything above the jump is a typo - provided that dividing them by
    ``factor`` brings them back into the normal range (so it really is "an extra zero").
    """
    mask = pd.Series(False, index=target.index)
    values = target.dropna().sort_values().to_numpy()
    n = len(values)
    if n < 20:
        return mask
    half = n // 2
    jumps = values[half + 1:] / np.where(values[half:-1] == 0, np.nan, values[half:-1])
    if len(jumps) == 0 or np.all(np.isnan(jumps)):
        return mask
    i = half + int(np.nanargmax(jumps))          # last "normal" value sits at index i
    if jumps[i - half] < gap_ratio:
        return mask
    threshold = values[i]
    n_above = n - (i + 1)
    if n_above == 0 or n_above / n > max_share:
        return mask
    if (values[i + 1:] / factor).max() > values[: i + 1].max() * 1.25:  # /10 must land in the normal range
        return mask
    return target > threshold


def check_target_typos(df: pd.DataFrame, target: str, gap_ratio: float = 2.0, factor: float = 10.0):
    mask = typo_mask(df[target], gap_ratio, factor)
    count = int(mask.sum())
    evidence = {"largest_normal_value": float(df.loc[~mask, target].max()),
                "typo_values": sorted(float(v) for v in df.loc[mask, target]),
                "typo_values_divided_by_factor": sorted(round(float(v) / factor, 2) for v in df.loc[mask, target])}
    return {"present": count > 0, "count": count if count > 0 else None}, evidence


# --------------------------------------------------------------------------- 7. spelling variants
def derive_clean_levels(values: pd.Series, reference: pd.Series | None = None) -> list[str]:
    """The real categories of a text column.

    If a clean reference column exists (later.csv has the clean spellings) use its categories.
    Otherwise merge spellings that are the same after lower-casing / stripping / dropping punctuation,
    or where one is an abbreviation (prefix) of another (``fir.`` -> ``first``).
    """
    if reference is not None:
        return sorted(reference.dropna().astype(str).str.strip().unique())
    counts = values.dropna().astype(str).map(normalise_text).value_counts()
    levels: list[str] = []
    for key in counts.index:                         # most frequent spelling first
        if not any(lv.startswith(key) or key.startswith(lv) for lv in levels):
            levels.append(key)
    return sorted(levels)


def check_spelling_variants(df: pd.DataFrame, id_cols: list[str], reference: pd.DataFrame | None = None):
    """A text column that writes the same category in several ways (``First``, `` first``, ``fir.``)."""
    best, evidence = None, {"raw_vs_clean_levels": {}}
    for col in df.columns:
        if col in id_cols or pd.api.types.is_numeric_dtype(df[col]):
            continue
        ref = reference[col] if reference is not None and col in reference.columns else None
        levels = derive_clean_levels(df[col], ref)
        raw = sorted(df[col].dropna().astype(str).unique())
        evidence["raw_vs_clean_levels"][col] = {"raw_distinct": len(raw), "clean_levels": levels}
        extra = len(raw) - len(levels)
        if extra > 0 and (best is None or extra > best[2]):
            best = (col, len(levels), extra)
    if best is None:
        return {"present": False, "column": None, "true_levels": None}, evidence
    return {"present": True, "column": best[0], "true_levels": best[1]}, evidence


# --------------------------------------------------------------------------- everything together
def run_all_checks(train: pd.DataFrame, holdout: pd.DataFrame, later: pd.DataFrame | None, params: dict) -> dict:
    """Run the seven detectors. Returns {"data_issues": {...findings...}, "evidence": {...}}."""
    d, c = params["data"], params["data_checks"]
    target, id_col, date_col, group_col = d["target"], d["id_column"], d["date_column"], d["group_column"]

    leak, leak_ev = check_leaking_column(train, holdout, target, c["leak_min_spearman"])
    # leak column and target are not "features": keep them out of the feature-level checks
    skip = [target, id_col] + ([leak["column"]] if leak["present"] else [])

    unit, unit_ev = check_unit_mixup(train, group_col, skip, c["unit_ratio_trigger"])
    sentinel, sent_ev = check_sentinel_value(train, [id_col], c["sentinel_min_count"])
    dup, dup_ev = check_duplicate_rows(train, id_col)
    # a sentinel (e.g. 9999) repeated in neighbouring rows is a sentinel, not a stuck sensor: blank it out first
    scan = train.copy()
    if sentinel["present"]:
        scan[sentinel["column"]] = scan[sentinel["column"]].replace(sentinel["value"], np.nan)
    stuck, stuck_ev = check_stuck_sensor(scan, date_col, id_col, skip, c["stuck_min_rows"], c["stuck_min_days"])
    typos, typo_ev = check_target_typos(train, target, c["typo_gap_ratio"], c["typo_factor"])
    spelling, spell_ev = check_spelling_variants(train, [id_col, date_col], later)

    return {
        "data_issues": {
            "leaking_column": leak, "unit_mixup": unit, "sentinel_value": sentinel,
            "duplicate_rows": dup, "stuck_sensor": stuck, "target_typos": typos, "spelling_variants": spelling,
        },
        "evidence": {
            "leaking_column": leak_ev, "unit_mixup": unit_ev, "sentinel_value": sent_ev,
            "duplicate_rows": dup_ev, "stuck_sensor": stuck_ev, "target_typos": typo_ev,
            "spelling_variants": spell_ev,
        },
    }


def main() -> dict:
    params = load_params()
    paths = params["paths"]
    train = pd.read_csv(resolve(paths["train_csv"]))
    holdout = pd.read_csv(resolve(paths["holdout_csv"]))
    later = pd.read_csv(resolve(paths["later_csv"]))
    result = run_all_checks(train, holdout, later, params)

    out_path = resolve(paths["data_issues_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    for name, finding in result["data_issues"].items():
        print(f"{name:18s} {finding}")
    print(f"saved -> {out_path}")
    return result


if __name__ == "__main__":
    main()

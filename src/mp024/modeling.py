"""Model building and the shared "load + clean" step used by training, monitoring and prediction."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from mp024.cleaning import clean_reference_frame, clean_training_frame
from mp024.config import load_params, resolve
from mp024.data_checks import derive_clean_levels, run_all_checks


def feature_columns(params: dict) -> list[str]:
    d = params["data"]
    return list(d["numeric_features"]) + list(d["categorical_features"])


def build_pipeline(cfg: dict, params: dict) -> Pipeline:
    """impute + scale numbers, one-hot the text columns, (optional) pairwise interactions, then the regressor."""
    d, seed = params["data"], params["training"]["seed"]
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    preprocess = ColumnTransformer([
        ("num", numeric, list(d["numeric_features"])),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(d["categorical_features"])),
    ])
    steps: list[tuple] = [("preprocess", preprocess)]
    if cfg.get("poly_degree", 1) > 1:
        # degree 2 = every feature x every other feature (e.g. section x hours), plus squares
        steps.append(("interactions", PolynomialFeatures(degree=cfg["poly_degree"], include_bias=False)))

    kind = cfg["kind"]
    if kind == "ridge":
        model = Ridge(alpha=cfg["alpha"])
    elif kind == "gbr":
        model = GradientBoostingRegressor(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                                          learning_rate=cfg["learning_rate"], random_state=seed)
    elif kind == "hgb":
        model = HistGradientBoostingRegressor(max_iter=cfg["max_iter"], learning_rate=cfg["learning_rate"],
                                              random_state=seed)
    elif kind == "rf":
        model = RandomForestRegressor(n_estimators=cfg["n_estimators"], min_samples_leaf=cfg["min_samples_leaf"],
                                      random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    steps.append(("model", model))
    return Pipeline(steps)


def cv_mae(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, folds: int, seed: int) -> tuple[float, float]:
    """K-fold cross-validated MAE: (mean, std over folds)."""
    scores = -cross_val_score(pipe, X, y, cv=KFold(folds, shuffle=True, random_state=seed),
                              scoring="neg_mean_absolute_error")
    return float(scores.mean()), float(scores.std())


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def load_and_clean(params: dict | None = None) -> dict:
    """Read the three CSVs, detect the data problems, and return cleaned frames.

    Returns a dict with: params, issues, train_raw, train, later, holdout (all cleaned / sorted by date).
    """
    params = params or load_params()
    paths = params["paths"]
    train_raw = pd.read_csv(resolve(paths["train_csv"]))
    later_raw = pd.read_csv(resolve(paths["later_csv"]))
    holdout_raw = pd.read_csv(resolve(paths["holdout_csv"]))

    result = run_all_checks(train_raw, holdout_raw, later_raw, params)
    issues = result["data_issues"]

    clean_levels: dict[str, list[str]] = {}
    spelling = issues["spelling_variants"]
    if spelling["present"]:
        col = spelling["column"]
        clean_levels[col] = derive_clean_levels(train_raw[col], later_raw[col] if col in later_raw else None)

    return {
        "params": params,
        "issues": issues,
        "train_raw": train_raw,
        "train": clean_training_frame(train_raw, issues, clean_levels, params),
        "later": clean_reference_frame(later_raw, issues, clean_levels, params),
        "holdout": clean_reference_frame(holdout_raw, issues, clean_levels, params),
    }


def save_champion(pipe: Pipeline, info: dict, params: dict) -> None:
    """Write the model file the API loads, plus a small JSON saying which registry version it is."""
    model_path = resolve(params["paths"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    resolve(params["paths"]["model_info_path"]).write_text(json.dumps(info, indent=2), encoding="utf-8")


def split_later_by_date(later: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """later.csv sorted by date: the first ``fraction`` of rows and the rest."""
    ordered = later.sort_values(["date", "record_id"], kind="stable").reset_index(drop=True)
    cut = int(round(len(ordered) * fraction))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


__all__ = ["build_pipeline", "cv_mae", "mae", "feature_columns", "load_and_clean", "save_champion",
           "split_later_by_date", "np"]

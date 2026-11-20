"""Train v1: one MLflow run per candidate model, register the best one, give it the `champion` alias.

Run it:  PYTHONPATH=src python -m mp024.train
"""
from __future__ import annotations

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from mp024.config import load_params, resolve, sha256_of_file
from mp024.modeling import build_pipeline, cv_mae, feature_columns, load_and_clean, mae, save_champion


def start_tracking(params: dict) -> None:
    """Point MLflow at the SQLite file in the repository root and select the experiment."""
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])


def train_v1(params: dict | None = None) -> dict:
    params = params or load_params()
    start_tracking(params)
    data = load_and_clean(params)
    train = data["train"]
    features, target = feature_columns(params), params["data"]["target"]
    X, y = train[features], train[target]
    data_sha = sha256_of_file(resolve(params["paths"]["train_csv"]))  # hash of the ORIGINAL train.csv
    seed, folds = params["training"]["seed"], params["training"]["cv_folds"]

    results = []
    for cfg in params["training"]["candidates"]:
        with mlflow.start_run(run_name=cfg["name"]) as run:
            pipe = build_pipeline(cfg, params)
            mlflow.log_param("data_sha256", data_sha)
            mlflow.log_param("stage", "v1_candidate")
            mlflow.log_param("train_rows", len(train))
            mlflow.log_params({f"cfg_{k}": v for k, v in cfg.items()})

            mae_cv, mae_cv_std = cv_mae(pipe, X, y, folds, seed)   # honest estimate: out-of-fold
            pipe.fit(X, y)
            mae_train = mae(y, pipe.predict(X))                      # optimistic: in-sample
            mlflow.log_metrics({"mae_cv": mae_cv, "mae_cv_std": mae_cv_std, "mae_train": mae_train})
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")

            results.append({"cfg": cfg, "run_id": run.info.run_id, "mae_cv": mae_cv, "pipe": pipe})
            print(f"{cfg['name']:16s} mae_cv={mae_cv:.4f}  mae_train={mae_train:.4f}  run={run.info.run_id}")

    best = min(results, key=lambda r: r["mae_cv"])
    reg_name, alias = params["mlflow"]["registered_model_name"], params["mlflow"]["champion_alias"]
    client = MlflowClient()
    version = mlflow.register_model(f"runs:/{best['run_id']}/model", reg_name)   # first registration = version 1
    client.set_registered_model_alias(reg_name, alias, version.version)
    client.set_tag(best["run_id"], "role", "v1_best")
    client.set_model_version_tag(reg_name, version.version, "candidate", best["cfg"]["name"])

    save_champion(best["pipe"], {"registered_model": reg_name, "version": int(version.version),
                                 "run_id": best["run_id"], "candidate": best["cfg"]["name"]}, params)
    print(f"\nbest v1 = {best['cfg']['name']} (mae_cv={best['mae_cv']:.4f}) -> {reg_name} version {version.version} "
          f"@{alias}, run {best['run_id']}")
    return {"best_run_id": best["run_id"], "version": int(version.version), "candidate": best["cfg"]["name"],
            "mae_cv": best["mae_cv"]}


if __name__ == "__main__":
    train_v1()

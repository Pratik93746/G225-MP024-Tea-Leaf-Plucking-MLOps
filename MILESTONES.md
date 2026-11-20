# Milestones (the four stages in STUDENT_HANDBOOK.md, section 6)

The repository is complete. This file says which files belong to which stage, so
each member can commit "their" stage from their **own GitHub account** (the
marking tool counts commits per person). Commit in this order.

| Stage | Role | Files | Done when |
|---|---|---|---|
| 1. Repo, DVC, data problems | Data lead | `README.md`, `.gitignore`, `.dvc/config`, `.dvc/.gitignore`, `.dvcignore`, `data/raw/*.dvc`, `data/raw/.gitignore`, `data/raw/holdout_features.csv`, `params.yaml`, `src/mp024/__init__.py`, `config.py`, `data_checks.py`, `reports/data_issues.json` | `.dvc` files are on GitHub and the data part of `findings.yaml` is filled |
| 2. Cleaning, MLflow, v1 | Model lead | `src/mp024/cleaning.py`, `modeling.py`, `train.py`, `requirements.txt`, `pytest.ini`, `mlflow.db`, `models/`, `tests/conftest.py`, `tests/test_data_checks.py`, `tests/test_cleaning.py` | `mlflow.db` has 3+ runs, a registered model and the `champion` alias |
| 3. Service, Docker, CI, k8s | Service lead | `app/`, `src/mp024/logging_utils.py`, `Dockerfile`, `.dockerignore`, `tests/test_api.py`, `tests/test_logging.py`, `.github/workflows/ci.yml`, `k8s/`, `tests/test_k8s_files.py` | green tick on the GitHub Actions tab |
| 4. Monitoring, v2, predictions | Ops lead | `src/mp024/monitoring.py`, `predict_holdout.py`, `findings.py`, `tests/test_monitoring.py`, `reports/monitoring.json`, `reports/findings.yaml`, `predictions/holdout_predictions.csv`, `run_pipeline.sh` | form submitted before the deadline |

## Committing one stage (example: stage 1)

Check `git config user.name` and `git config user.email` are **your own** GitHub
identity first, then:

```bash
git add README.md .gitignore .dvc/config .dvc/.gitignore .dvcignore \
        data/raw/*.dvc data/raw/.gitignore data/raw/holdout_features.csv \
        params.yaml src/mp024/__init__.py src/mp024/config.py \
        src/mp024/data_checks.py reports/data_issues.json
git commit -m "Stage 1: DVC, data checks, README"
git push
```

Do the same for stages 2 to 4 with the file lists above. Everything in
`data/raw/*.csv` (except `holdout_features.csv`) and `.dvc/cache` stays out of
Git; the `.gitignore` files that `dvc add` and `dvc init` created already cover them.

## Re-running everything

```bash
pip install -r requirements.txt
python -m pytest -q          # 26 tests
bash run_pipeline.sh         # checks, train v1, monitor, retrain v2, predict, findings
```

`run_pipeline.sh` deletes and rebuilds `mlflow.db`, so the run ids in
`reports/findings.yaml` are rewritten each time. Commit the fresh `mlflow.db`
and `reports/` together.

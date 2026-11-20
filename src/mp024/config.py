"""Read params.yaml and small file helpers shared by every module."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

# src/mp024/config.py -> parents[2] is the repository root
ROOT = Path(__file__).resolve().parents[2]


def load_params(path: str | Path | None = None) -> dict:
    """Load params.yaml (the repository one unless another path is given)."""
    params_path = Path(path) if path else ROOT / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(relative: str | Path) -> Path:
    """Turn a path from params.yaml into an absolute path inside the repository."""
    path = Path(relative)
    return path if path.is_absolute() else ROOT / path


def sha256_of_file(path: str | Path) -> str:
    """SHA-256 of the raw bytes of a file (used for the `data_sha256` MLflow parameter)."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()

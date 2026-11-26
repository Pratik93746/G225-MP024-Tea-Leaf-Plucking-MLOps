"""Prediction service for Lucknow Tea Growers' Estate.

    POST /predict  -> expected kilos plucked by one worker in a day
    GET  /health   -> {"status": "ok"}

The client's two rules live in the request schema, so a bad request gets status 422
and never reaches the model:
    1. hours_worked above 15 is impossible          -> Field(le=15)
    2. rain_mm_week below 0 is a typing mistake     -> Field(ge=0)
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mp024.cleaning import canonical_level
from mp024.logging_utils import get_json_logger

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "model.joblib"))
MODEL_INFO_PATH = Path(os.environ.get("MODEL_INFO_PATH", ROOT / "models" / "model_info.json"))

FLUSH_LEVELS = ["first", "second", "monsoon"]
SECTION_LEVELS = ["Upper Garden", "Lower Garden", "River Side"]
FEATURE_ORDER = ["worker_exp_yr", "rain_mm_week", "slope_deg", "temp_c", "bush_age_yr", "hours_worked", "flush", "section"]

logger = get_json_logger("mp024.api")  # one JSON line per prediction, to stdout



@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model before the server starts accepting traffic, so the first prediction is not slow."""
    get_model()
    yield


app = FastAPI(title="MP024 Tea Leaf Plucking - kg per worker", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    """Exactly these eight fields - nothing more, nothing less."""

    model_config = ConfigDict(extra="forbid")

    worker_exp_yr: float = Field(..., ge=0, le=60, description="worker's experience, years")
    rain_mm_week: float = Field(..., ge=0, description="rain in the past week, mm (client rule: never below 0)")
    slope_deg: float = Field(..., ge=0, le=90, description="slope of the section, degrees")
    temp_c: float = Field(..., ge=-20, le=60, description="average temperature, deg C")
    bush_age_yr: float = Field(..., ge=0, le=150, description="age of the tea bushes, years")
    hours_worked: float = Field(..., ge=0, le=15, description="hours plucked (client rule: never above 15)")
    flush: str = Field(..., description="picking season: first, second or monsoon")
    section: str = Field(..., description="Upper Garden, Lower Garden or River Side")

    @field_validator("flush")
    @classmethod
    def _flush(cls, value: str) -> str:
        level = canonical_level(value, FLUSH_LEVELS)  # also accepts ' First', 'SECOND', 'mon.'
        if not isinstance(level, str):
            raise ValueError(f"flush must be one of {FLUSH_LEVELS}")
        return level

    @field_validator("section")
    @classmethod
    def _section(cls, value: str) -> str:
        level = canonical_level(value, SECTION_LEVELS)
        if not isinstance(level, str):
            raise ValueError(f"section must be one of {SECTION_LEVELS}")
        return level


class PredictResponse(BaseModel):
    prediction: float = Field(..., description="expected kg plucked by the worker in the day")
    model_version: str


@lru_cache(maxsize=1)
def get_model():
    """Load the champion model once, on the first request (works with or without a lifespan)."""
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_model_version() -> str:
    try:
        return str(json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))["version"])
    except (OSError, KeyError, ValueError):
        return "unknown"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    started = time.perf_counter()
    row = pd.DataFrame([request.model_dump()])[FEATURE_ORDER]
    value = max(0.0, float(get_model().predict(row)[0]))   # a negative harvest is impossible
    logger.info(
        "prediction served",
        extra={  # agronomic inputs only - no personal details about any worker
            "event": "prediction", "prediction_kg": round(value, 3), "model_version": get_model_version(),
            "section": request.section, "flush": request.flush, "hours_worked": request.hours_worked,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return PredictResponse(prediction=round(value, 3), model_version=get_model_version())

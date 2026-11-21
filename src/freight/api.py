"""FastAPI service: the registry's champion model behind /predict, plus a drift check.

The model is loaded from the MLflow registry (``models:/freight-price@champion``) on first
use. ``FREIGHT_MODEL_URI`` overrides that, which the tests use.
"""

from __future__ import annotations

import os
from collections import deque
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from freight.cli import ALIAS, NAME, tracking_uri
from freight.data import CATEGORICAL, FEATURES, MODES
from freight.model import drift

app = FastAPI(title="Freight cost model", version="0.1.0")
RECENT: deque[dict] = deque(maxlen=1000)  # ponytail: in-memory, resets on restart


class Shipment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    country: str = Field(min_length=2, max_length=60, examples=["Nigeria"])
    shipment_mode: Literal[tuple(MODES)] = Field(examples=["Air"])  # type: ignore[valid-type]
    vendor_inco_term: str = Field(min_length=2, max_length=30, examples=["N/A - From RDC"])
    fulfill_via: Literal["Direct Drop", "From RDC"] = Field(examples=["From RDC"])
    origin: str = Field(min_length=2, max_length=80, examples=["Aurobindo Unit III, India"])
    product_group: str = Field(min_length=2, max_length=10, examples=["ARV"])
    weight_kg: float = Field(gt=0, le=1_000_000, examples=[1500])
    line_value_usd: float = Field(ge=0, le=50_000_000, examples=[120000])
    line_quantity: float = Field(ge=0, le=5_000_000, examples=[4000])
    n_lines: int = Field(default=1, ge=1, le=100)


class Quote(BaseModel):
    p10: float
    p50: float
    p90: float
    currency: str = "USD"
    model_version: str
    unseen: list[str] = Field(description="inputs the model never saw in training")


@lru_cache(maxsize=1)
def champion():
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    uri = os.environ.get("FREIGHT_MODEL_URI", f"models:/{NAME}@{ALIAS}")
    version = "local"
    if uri.startswith("models:/"):
        version = mlflow.MlflowClient().get_model_version_by_alias(NAME, ALIAS).version
    return mlflow.sklearn.load_model(uri), version


@app.get("/health")
def health() -> dict:
    _, version = champion()
    return {"status": "ok", "model_version": version}


@app.post("/predict")
def predict(s: Shipment) -> Quote:
    model, version = champion()
    row = s.model_dump()
    RECENT.append(row)
    p = model.predict(pd.DataFrame([row])[FEATURES]).iloc[0]
    unseen = [c for c in CATEGORICAL if row[c] not in model.categories_[c]]
    return Quote(
        p10=round(p["p10"], 2),
        p50=round(p["p50"], 2),
        p90=round(p["p90"], 2),
        model_version=version,
        unseen=unseen,
    )


@app.get("/drift")
def recent_drift(min_rows: int = 50) -> dict:
    """PSI of the last 1,000 requests against the training data. 'retrain' means act."""
    if len(RECENT) < min_rows:
        return {"verdict": "not enough requests", "rows": len(RECENT), "min_rows": min_rows}
    model, _ = champion()
    return drift(model.reference_, pd.DataFrame(list(RECENT)))

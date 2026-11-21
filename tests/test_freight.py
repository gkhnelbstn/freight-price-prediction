"""Each test pins one claim from the README."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight.data import FEATURES, MODES, TARGET, load, read_raw, split
from freight.model import Baseline, QuantileModel, drift, reference


@pytest.fixture(scope="session")
def shipments() -> pd.DataFrame:
    return load()


@pytest.fixture(scope="session")
def small_model(shipments) -> QuantileModel:
    train, _ = split(shipments)
    fit, cal = split(train, 2012)
    model = QuantileModel(max_iter=60).fit(fit).calibrate(cal)
    model.reference_ = reference(fit)
    return model


def test_each_shipment_has_exactly_one_freight_value() -> None:
    raw = read_raw()
    numeric = pd.to_numeric(raw["Freight Cost (USD)"], errors="coerce").notna()
    per_shipment = numeric.groupby(raw["ASN/DN #"]).sum()
    assert per_shipment.max() == 1  # so taking "the row with the number" is not a choice


def test_shipment_table_shape(shipments) -> None:
    assert len(shipments) == 5963
    assert shipments["shipment_id"].is_unique
    assert set(shipments["shipment_mode"]) == set(MODES)
    assert (shipments[TARGET] > 0).all() and (shipments["weight_kg"] > 0).all()


def test_features_are_known_at_quote_time() -> None:
    assert not {"delivered", TARGET, "insurance"} & set(FEATURES)


def test_split_is_by_time(shipments) -> None:
    train, test = split(shipments, 2013)
    assert train["delivered"].max() < test["delivered"].min()
    assert len(train) + len(test) == len(shipments)


def test_baseline_is_rate_per_kg_times_weight(shipments) -> None:
    b = Baseline().fit(shipments)
    row = shipments.iloc[[0]]
    assert b.predict(row)[0] == pytest.approx(
        b.rate_[row["shipment_mode"].iloc[0]] * row["weight_kg"].iloc[0]
    )


def test_ranges_are_ordered(small_model, shipments) -> None:
    p = small_model.predict(shipments.sample(500, random_state=1))
    assert (p["p10"] <= p["p50"]).all() and (p["p50"] <= p["p90"]).all()


def test_calibration_hits_target_on_its_own_year(shipments) -> None:
    train, _ = split(shipments)
    fit, cal = split(train, 2012)
    model = QuantileModel(max_iter=60).fit(fit)

    def coverage(m):
        p, y = m.predict(cal), cal[TARGET]
        return ((y >= p["p10"]) & (y <= p["p90"])).mean()

    raw = coverage(model)
    assert coverage(model.calibrate(cal, 0.80)) >= 0.80 > raw


def test_drift_is_stable_on_training_data_and_flags_a_shift(small_model, shipments) -> None:
    train, _ = split(shipments, 2012)
    assert drift(small_model.reference_, train)["verdict"] == "stable"
    shifted = train.assign(weight_kg=train["weight_kg"] * 20, origin="New factory")
    result = drift(small_model.reference_, shifted)
    assert result["verdict"] == "retrain"
    assert result["psi"]["origin"] > 0.25 and result["psi"]["weight_kg"] > 0.25


# --- API -------------------------------------------------------------------------------

GOOD = {
    "country": "Nigeria",
    "shipment_mode": "Air",
    "vendor_inco_term": "N/A - From RDC",
    "fulfill_via": "From RDC",
    "origin": "Aurobindo Unit III, India",
    "product_group": "ARV",
    "weight_kg": 1500,
    "line_value_usd": 120000,
    "line_quantity": 4000,
}


@pytest.fixture(scope="module")
def client(small_model, tmp_path_factory, monkeypatch_module):
    import mlflow
    from fastapi.testclient import TestClient

    from freight import api
    from freight.cli import TRUSTED

    path = tmp_path_factory.mktemp("model") / "m"
    mlflow.sklearn.save_model(small_model, str(path), skops_trusted_types=TRUSTED)
    monkeypatch_module.setenv("FREIGHT_MODEL_URI", str(path))
    api.champion.cache_clear()
    api.RECENT.clear()
    return TestClient(api.app)


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def test_predict_returns_an_ordered_range(client) -> None:
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    q = r.json()
    assert 0 < q["p10"] <= q["p50"] <= q["p90"]
    assert q["unseen"] == []


@pytest.mark.parametrize(
    "change",
    [
        {"shipment_mode": "Rail"},
        {"weight_kg": 0},
        {"weight_kg": -5},
        {"surprise": 1},
        {"country": ""},
    ],
)
def test_predict_rejects_bad_input(client, change) -> None:
    assert client.post("/predict", json=GOOD | change).status_code == 422


def test_predict_flags_values_it_never_saw(client) -> None:
    q = client.post("/predict", json=GOOD | {"country": "Atlantis"}).json()
    assert q["unseen"] == ["country"] and q["p50"] > 0


def test_drift_endpoint_waits_for_enough_requests_then_reports(client) -> None:
    from freight import api

    api.RECENT.clear()
    assert client.get("/drift").json()["verdict"] == "not enough requests"
    for w in np.linspace(5_000, 50_000, 60):  # all far heavier than typical training rows
        client.post("/predict", json=GOOD | {"weight_kg": float(w), "origin": "New factory"})
    assert client.get("/drift").json()["verdict"] == "retrain"

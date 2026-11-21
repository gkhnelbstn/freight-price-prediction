"""Baseline, quantile model, metrics and drift checks.

The model predicts log freight, so a $500 and a $50,000 shipment count the same in
training. Three gradient boosting models give P10, P50 and P90 in USD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from freight.data import CATEGORICAL, FEATURES, NUMERIC, TARGET

QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}


class Baseline:
    """Median freight per kg for each mode, times the weight. What a rate card would do."""

    def fit(self, df: pd.DataFrame) -> Baseline:
        per_kg = df[TARGET] / df["weight_kg"]
        self.rate_ = per_kg.groupby(df["shipment_mode"]).median().to_dict()
        self.default_ = float(per_kg.median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        rate = df["shipment_mode"].map(self.rate_).fillna(self.default_)
        return (rate * df["weight_kg"]).to_numpy()


class QuantileModel:
    def __init__(self, max_iter: int = 400, learning_rate: float = 0.05, seed: int = 7):
        self.params = dict(
            max_iter=max_iter,
            learning_rate=learning_rate,
            min_samples_leaf=20,
            l2_regularization=1.0,
            categorical_features="from_dtype",
            random_state=seed,
        )

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(index=df.index)
        for c in CATEGORICAL:
            # Values not seen in training become missing, which the trees handle.
            known = self.categories_[c]
            x[c] = pd.Categorical(df[c].where(df[c].isin(known)), categories=known)
        for c in NUMERIC:
            x[c] = np.log1p(df[c].astype(float))
        return x

    def fit(self, df: pd.DataFrame) -> QuantileModel:
        self.categories_ = {c: sorted(df[c].dropna().unique()) for c in CATEGORICAL}
        x, y = self._x(df), np.log(df[TARGET].to_numpy())
        self.models_ = {
            name: HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.params).fit(
                x, y
            )
            for name, q in QUANTILES.items()
        }
        return self

    def _logs(self, df: pd.DataFrame) -> np.ndarray:
        x = self._x(df[FEATURES])
        logs = np.column_stack([m.predict(x) for m in self.models_.values()])
        logs.sort(axis=1)  # separately fitted quantiles can cross; keep p10 <= p50 <= p90
        return logs

    def calibrate(self, cal: pd.DataFrame, coverage: float = 0.80) -> QuantileModel:
        """Conformal step: widen P10/P90 until they hold ``coverage`` of a held-out year.

        Raw quantile trees are overconfident (their range is fitted to the same rows they
        learned from), so the offset is measured on shipments the model has not seen.
        """
        logs, y = self._logs(cal), np.log(cal[TARGET].to_numpy())
        score = np.maximum(logs[:, 0] - y, y - logs[:, 2])
        n = len(score)
        self.offset_ = float(np.quantile(score, min(1.0, np.ceil((n + 1) * coverage) / n)))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        logs = self._logs(df)
        offset = getattr(self, "offset_", 0.0)
        logs[:, 0] -= offset
        logs[:, 2] += offset
        return pd.DataFrame(np.exp(logs), columns=list(QUANTILES), index=df.index)


def errors(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    ape = np.abs(predicted - actual) / actual
    return {
        "mae_usd": float(np.mean(np.abs(predicted - actual))),
        "median_ape": float(np.median(ape)),
        "within_25pct": float(np.mean(ape <= 0.25)),
    }


def report(test: pd.DataFrame, baseline: np.ndarray, pred: pd.DataFrame) -> pd.DataFrame:
    """Baseline vs model per mode and overall, plus P10-P90 coverage for the model."""
    rows = []
    groups = [("all", test.index)] + list(test.groupby("shipment_mode").groups.items())
    for name, idx in groups:
        y = test.loc[idx, TARGET].to_numpy()
        p = pred.loc[idx]
        covered = (y >= p["p10"].to_numpy()) & (y <= p["p90"].to_numpy())
        for model, yhat in [
            ("baseline", baseline[test.index.get_indexer(idx)]),
            ("model", p["p50"]),
        ]:
            rows.append(
                {"mode": name, "shipments": len(idx), "model": model}
                | errors(y, np.asarray(yhat))
                | {"p10_p90_coverage": float(covered.mean()) if model == "model" else np.nan}
            )
    return pd.DataFrame(rows)


def importance(model: QuantileModel, test: pd.DataFrame, repeats: int = 5, seed: int = 7):
    """Rise in P50 median APE when one feature is shuffled. Bigger = the model leans on it."""
    rng = np.random.default_rng(seed)
    y = test[TARGET].to_numpy()
    base = errors(y, model.predict(test)["p50"].to_numpy())["median_ape"]
    out = {}
    for f in FEATURES:
        rises = []
        for _ in range(repeats):
            shuffled = test.copy()
            shuffled[f] = rng.permutation(shuffled[f].to_numpy())
            rises.append(errors(y, model.predict(shuffled)["p50"].to_numpy())["median_ape"] - base)
        out[f] = float(np.mean(rises))
    return pd.Series(out).sort_values(ascending=False)


# --- drift -------------------------------------------------------------------------------

PSI_WATCH, PSI_RETRAIN = 0.10, 0.25


def reference(train: pd.DataFrame) -> dict:
    """What drift is measured against: decile edges and shares per feature, from training."""
    ref = {}
    for c in NUMERIC:
        edges = np.unique(np.quantile(train[c].astype(float), np.linspace(0, 1, 11)[1:-1]))
        ref[c] = {"edges": edges.tolist(), "shares": _shares_numeric(train[c], edges)}
    for c in CATEGORICAL:
        shares = train[c].value_counts(normalize=True)
        ref[c] = {"shares": shares.to_dict()}
    return ref


def _shares_numeric(values: pd.Series, edges) -> list[float]:
    counts = np.bincount(
        np.searchsorted(edges, values.astype(float), side="right"), minlength=len(edges) + 1
    )
    return (counts / counts.sum()).tolist()


def _psi(expected, actual, eps: float = 1e-4) -> float:
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift(ref: dict, recent: pd.DataFrame) -> dict:
    """PSI per feature for recent requests against the training reference."""
    psi = {}
    for c in NUMERIC:
        psi[c] = _psi(ref[c]["shares"], _shares_numeric(recent[c], np.asarray(ref[c]["edges"])))
    for c in CATEGORICAL:
        seen = ref[c]["shares"]
        now = recent[c].value_counts(normalize=True)
        new = float(now[~now.index.isin(list(seen))].sum())  # values training never saw
        keys = list(seen)
        psi[c] = _psi(list(seen.values()) + [0.0], [float(now.get(k, 0.0)) for k in keys] + [new])
    worst = max(psi.values())
    verdict = "retrain" if worst >= PSI_RETRAIN else "watch" if worst >= PSI_WATCH else "stable"
    return {
        "psi": dict(sorted(psi.items(), key=lambda kv: -kv[1])),
        "verdict": verdict,
        "rows": len(recent),
    }

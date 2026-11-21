"""freight train | drift | serve."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from freight.data import FEATURES, load, split
from freight.model import Baseline, QuantileModel, drift, importance, reference, report

NAME = "freight-price"
ALIAS = "champion"
REPORTS = Path("reports")
# skops refuses unknown types on save and load. These are the ones this model is made of;
# the file is our own, written by `freight train`.
TRUSTED = [
    "freight.model.QuantileModel",
    "functools.partial",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
]


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def train(test_from: int, max_iter: int) -> dict:
    import mlflow
    from mlflow import MlflowClient

    shipments = load()
    train_set, test = split(shipments, test_from)
    fit, cal = split(train_set, test_from - 1)  # last training year calibrates the ranges

    baseline = Baseline().fit(fit)
    model = QuantileModel(max_iter=max_iter).fit(fit).calibrate(cal)
    model.reference_ = reference(fit)  # drift is measured against what the model learned on

    pred = model.predict(test)
    table = report(test, baseline.predict(test), pred)
    weights = importance(model, test)
    shift = drift(model.reference_, test)

    overall = table[table["mode"] == "all"].set_index("model")
    metrics = {
        "test_mae_usd": overall.loc["model", "mae_usd"],
        "test_median_ape": overall.loc["model", "median_ape"],
        "test_within_25pct": overall.loc["model", "within_25pct"],
        "test_p10_p90_coverage": overall.loc["model", "p10_p90_coverage"],
        "baseline_mae_usd": overall.loc["baseline", "mae_usd"],
        "baseline_median_ape": overall.loc["baseline", "median_ape"],
        "interval_offset_log": model.offset_,
    }
    write_reports(table, weights, shift, len(fit), len(cal), len(test), test_from)

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(NAME)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "fit_years": f"..{test_from - 2}",
                "calibration_year": test_from - 1,
                "test_years": f"{test_from}..",
                "fit_rows": len(fit),
                "test_rows": len(test),
                **model.params,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(REPORTS), "reports")
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=NAME,
            input_example=test[FEATURES].head(3),
            skops_trusted_types=TRUSTED,
        )

    client = MlflowClient()
    version = info.registered_model_version
    client.set_model_version_tag(NAME, version, "test_mae_usd", f"{metrics['test_mae_usd']:.2f}")
    promoted = promote(client, version, metrics["test_mae_usd"])
    result = {"run_id": run.info.run_id, "version": version, "promoted": promoted, **metrics}
    print(json.dumps(result, indent=2, default=float))
    return result


def promote(client, version: str, mae: float) -> bool:
    """The new version becomes champion only if it beats the current one on test MAE."""
    try:
        current = client.get_model_version_by_alias(NAME, ALIAS)
    except Exception:  # no champion yet
        current = None
    if current is not None and float(current.tags.get("test_mae_usd", "inf")) <= mae:
        return False
    client.set_registered_model_alias(NAME, ALIAS, version)
    return True


def write_reports(table, weights, shift, n_fit, n_cal, n_test, test_from) -> None:
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / "errors_by_mode.csv", index=False)
    weights.rename("median_ape_rise").to_csv(REPORTS / "feature_importance.csv")
    (REPORTS / "drift.json").write_text(json.dumps(shift, indent=2))
    pct = {"median_ape", "within_25pct", "p10_p90_coverage"}
    shown = table.copy()
    for c in shown.columns:
        if c in pct:
            shown[c] = shown[c].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    shown["mae_usd"] = shown["mae_usd"].map(lambda v: f"{v:,.0f}")
    lines = [
        f"Fit on {n_fit:,} shipments delivered before {test_from - 1}, ranges calibrated on "
        f"{n_cal:,} from {test_from - 1}, tested on {n_test:,} from {test_from} onward.",
        "",
        shown.to_markdown(index=False),
        "",
        "Feature importance (rise in median APE when the feature is shuffled):",
        "",
        weights.map(lambda v: f"{v:+.3f}").to_frame("median_ape_rise").to_markdown(),
        "",
        f"Drift, train vs test (PSI): verdict **{shift['verdict']}**",
        "",
        pd.Series(shift["psi"]).map(lambda v: f"{v:.2f}").to_frame("psi").to_markdown(),
        "",
    ]
    (REPORTS / "results.md").write_text("\n".join(lines), encoding="utf-8")


def drift_cmd(csv: str) -> dict:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    model = mlflow.sklearn.load_model(f"models:/{NAME}@{ALIAS}")
    result = drift(model.reference_, pd.read_csv(csv))
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="freight")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="fit, evaluate, log to MLflow, promote if better")
    t.add_argument("--test-from", type=int, default=2013)
    t.add_argument("--max-iter", type=int, default=400)
    d = sub.add_parser(
        "drift", help="PSI of a CSV of requests against the champion's training data"
    )
    d.add_argument("csv")
    s = sub.add_parser("serve", help="run the API")
    s.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    if a.cmd == "train":
        train(a.test_from, a.max_iter)
    elif a.cmd == "drift":
        drift_cmd(a.csv)
    else:
        import uvicorn

        uvicorn.run("freight.api:app", host="0.0.0.0", port=a.port)


if __name__ == "__main__":
    main()

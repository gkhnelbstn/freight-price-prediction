# freight-price-prediction

[![CI](https://github.com/gkhnelbstn/freight-price-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/gkhnelbstn/freight-price-prediction/actions/workflows/ci.yml)

A freight cost model trained on real shipments. It returns a range (P10, P50, P90) instead of one
number, runs against a rate-card baseline, checks its own inputs for drift, and only replaces
the model in production when the new one scores better. MLflow tracks every run and holds the
registry. FastAPI serves the `@champion` version.

```bash
docker compose up --build          # MLflow on :5000, trains, then the API on :8000
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "country": "Nigeria", "shipment_mode": "Air", "vendor_inco_term": "N/A - From RDC",
  "fulfill_via": "From RDC", "origin": "Aurobindo Unit III, India", "product_group": "ARV",
  "weight_kg": 1500, "line_value_usd": 120000, "line_quantity": 4000}'
```
```json
{"p10": 4433.07, "p50": 9223.88, "p90": 23491.72, "currency": "USD", "model_version": "1", "unseen": []}
```

## The data

5,963 shipments from USAID's Supply Chain Management System, delivered to 38 countries between
2006 and 2015, each with its freight cost in USD. These are mostly air shipments of HIV
medicines and test kits, not truckload freight. The file is public domain;
[data/README.md](data/README.md) covers where it comes from, since USAID's portal went offline
in 2025, and how the copy is pinned.

The raw file has one row per line item, but freight and weight are recorded once per shipment.
The other lines of the shipment say `See ASN-... (ID#:...)`. So one training row is one
shipment: freight and weight from the row that has them, value and quantity summed over the
shipment. A test checks that each shipment carries exactly one freight value.

The model only sees what is known when freight gets quoted: destination country, mode, INCO
term, direct drop or regional warehouse, manufacturing site, product group, weight, value,
quantity and number of lines. Delivery dates and insurance stay out.

## How it is tested

By time, the way it would be used. The model learns from shipments delivered up to 2011.
Shipments from 2012 set the width of the ranges, and those from 2013 to 2015 (2,036) are
the test. A random split would mix 2014 prices into training and flatter every number below.

The baseline is a rate card: the median freight per kg for each mode, times the weight. The
model is three gradient boosting regressors on log freight, one each for P10, P50 and P90.

| mode | shipments | baseline MAE | model MAE | baseline median error | model median error | P10–P90 holds the actual |
|:--|--:|--:|--:|--:|--:|--:|
| all | 2,036 | $12,695 | **$7,575** | 56% | **41%** | 72% |
| Air | 1,259 | $8,113 | **$6,074** | 46% | **33%** | 75% |
| Truck | 573 | $11,965 | **$6,866** | 72% | **50%** | 76% |
| Air Charter | 152 | $55,570 | **$19,747** | 116% | **57%** | 34% |
| Ocean | 52 | **$6,351** | $16,143 | **41%** | 157% | 56% |

Full table, feature importance and drift numbers: [reports/results.md](reports/results.md).
CI regenerates that file and fails if it changes.

What the numbers say:

- The model cuts the rate card's error by 40% overall and beats it on air, truck and charter.
  The median shipment is still off by 41%, so treat P50 as a first quote and the range as the
  answer.
- On ocean the rate card wins. Training has about 190 ocean shipments and the test years 52,
  too few for the trees. A client would get the rate card for ocean until more data comes in.
- Weight does most of the work: shuffling it raises the median error by 35 points. Origin,
  quantity and INCO term add 2 to 3 points each.

## The ranges

Quantile trees come out overconfident. Fitted alone, the P10–P90 band held 57% of the test
shipments instead of 80%. A conformal step fixes most of that: it measures how far the 2012
shipments fall outside the band and widens the band by that much (in log space, so by 27%
each side). On the test years the band then holds 72%.

The remaining gap is drift, and the drift check sees it. Between training and test, the mix of
manufacturing sites (PSI 1.99), countries (1.53) and INCO terms (0.58) moved, and any PSI
above 0.25 means retrain. Air charter, the mode that moved most, has the worst coverage at 34%.

## Drift and retraining

`GET /drift` compares the last 1,000 requests with the training data, feature by feature,
and answers `stable`, `watch` (some PSI ≥ 0.10) or `retrain` (some PSI ≥ 0.25).
`freight drift requests.csv` does the same for a file. `/predict` also lists inputs the model
never saw, such as a new country, in `unseen`.

`freight train` logs each run to MLflow and registers a new version, then moves the
`@champion` alias to it only if its test MAE beats the current champion's. CI trains twice
and checks that the second, identical run is not promoted.

## Run it without Docker

```bash
uv sync
uv run freight train      # writes reports/, logs to ./mlflow.db, sets @champion
uv run freight serve      # http://localhost:8000/docs
uv run pytest             # 16 tests: data, ranges, calibration, drift, API validation
```

## Layout

```
src/freight/data.py    line items -> shipments, time split, pinned checksum
src/freight/model.py   rate-card baseline, quantile model + conformal step, metrics, PSI
src/freight/cli.py     freight train | drift | serve; MLflow logging and promotion
src/freight/api.py     /predict, /drift, /health
```

MIT licensed.

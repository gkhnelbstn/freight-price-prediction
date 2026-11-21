"""Load the SCMS file and turn line items into one row per shipment.

The source has one row per line item, but freight and weight are recorded once per
shipment (ASN/DN). The other rows of that shipment say "See ASN-... (ID#:...)". So a
training row is a shipment: freight and weight from the row that has the numbers, value
and quantity summed over the shipment, and the quote-time fields from its first row.
"""

from __future__ import annotations

import gzip
import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "scms_delivery_history.csv.gz"
SHA256 = "918b992dd3e8d4b64d2a727b2c4ea607603d0c58f19484e73f7b78528c6a8673"  # uncompressed CSV

# Known when the freight is quoted. Delivery dates and insurance are not.
CATEGORICAL = [
    "country",
    "shipment_mode",
    "vendor_inco_term",
    "fulfill_via",
    "origin",
    "product_group",
]
NUMERIC = ["weight_kg", "line_value_usd", "line_quantity", "n_lines"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "freight_usd"
MODES = ["Air", "Air Charter", "Ocean", "Truck"]


def read_raw(path: Path = DATA) -> pd.DataFrame:
    raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError(f"{path} does not match the pinned SCMS checksum")
    return pd.read_csv(BytesIO(raw), encoding="latin1", dtype=str)


def shipments(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per ASN/DN with a numeric freight cost and weight, quote-time fields only."""
    df = raw.rename(columns=lambda c: c.strip().lstrip("﻿ï»¿"))
    freight = pd.to_numeric(df["Freight Cost (USD)"], errors="coerce")
    weight = pd.to_numeric(df["Weight (Kilograms)"], errors="coerce")
    df = df.assign(
        _freight=freight,
        _weight=weight,
        _value=pd.to_numeric(df["Line Item Value"], errors="coerce").fillna(0.0),
        _qty=pd.to_numeric(df["Line Item Quantity"], errors="coerce").fillna(0.0),
        _delivered=pd.to_datetime(df["Delivered to Client Date"], format="%d-%b-%y"),
    )
    key = "ASN/DN #"
    sums = df.groupby(key).agg(
        line_value_usd=("_value", "sum"),
        line_quantity=("_qty", "sum"),
        n_lines=("_value", "size"),
        delivered=("_delivered", "min"),
    )
    # Exactly one row per shipment carries the numbers (checked in tests).
    priced = df[df["_freight"].notna()].drop_duplicates(key).set_index(key)
    out = pd.DataFrame(
        {
            "country": priced["Country"],
            "shipment_mode": priced["Shipment Mode"],
            "vendor_inco_term": priced["Vendor INCO Term"],
            "fulfill_via": priced["Fulfill Via"],
            "origin": priced["Manufacturing Site"],
            "product_group": priced["Product Group"],
            "weight_kg": priced["_weight"],
            TARGET: priced["_freight"],
        }
    ).join(sums)
    keep = (out[TARGET] > 0) & (out["weight_kg"] > 0) & out["shipment_mode"].isin(MODES)
    return out[keep].sort_values("delivered").reset_index(names="shipment_id")


def load() -> pd.DataFrame:
    return shipments(read_raw())


def split(df: pd.DataFrame, test_from: int = 2013) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on shipments delivered before ``test_from``, test on the later ones."""
    later = df["delivered"].dt.year >= test_from
    return df[~later].reset_index(drop=True), df[later].reset_index(drop=True)

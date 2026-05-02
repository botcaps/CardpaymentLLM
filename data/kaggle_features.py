"""
Derives feature-store lookup tables from the Kaggle Credit Card Fraud Detection dataset
(mlg-ulb/creditcardfraud, creditcard.csv).

Computed tables (keyed to match feature_store / fraud_store format):
  auth_rate_history  — (bin, scheme) → {rate_30d, rate_90d, volume_30d}
  velocity           — (bin, merchant_id) → {txns_1h, txns_24h}
  bin_table          — bin → (issuer_id, issuer_name, country, card_type, brand_capabilities)
  issuer_health      — issuer_id → {status, incident}
  decline_patterns   — (bin, hour_bucket) → {insufficient_funds, do_not_honor, fraud, other}

creditcard.csv has no card/merchant metadata — all card/merchant fields are derived
deterministically from the PCA components V1-V3 (same derivation used in kaggle_loader.py).

Auth success rate is approximated as (1 - fraud_rate); creditcard.csv records all
authorised transactions so there are no issuer-declined rows in the dataset.
"""
from __future__ import annotations

import os
import pathlib
import pickle

import pandas as pd

_HOUR = 3600
_DAY  = 86400

_SCHEME_MAP = {0: "visa", 1: "mastercard", 2: "amex", 3: "discover"}

_WANT_COLS = ["Time", "Amount", "Class", "V1", "V2", "V3"]


def _hour_bucket(hour: int) -> str:
    return "business" if 8 <= hour <= 20 else "offhours"


def _derive_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived card/merchant columns so all builders can share a single prep step."""
    df = df.copy()

    def _bin(v1, v2):
        try:
            return str(abs(int(float(v1) * 10_000 + float(v2) * 1_000))).zfill(6)[:6]
        except (ValueError, TypeError, OverflowError):
            return None

    df["bin"]         = df.apply(lambda r: _bin(r["V1"], r["V2"]), axis=1)
    df["bin_int"]     = df["bin"].apply(lambda b: int(b) if b else None)
    df["scheme"]      = df["bin_int"].apply(lambda i: _SCHEME_MAP[int(i) % 4] if i is not None else None)
    df["card_type"]   = df["V3"].apply(lambda v: "debit" if float(v) < 0 else "credit")
    df["merchant_id"] = df["bin_int"].apply(lambda i: f"mer_{int(i) % 100}" if i is not None else "mer_42")
    # Alias columns to the names the builder functions expect
    df["isFraud"]       = df["Class"]
    df["TransactionDT"] = df["Time"]

    return df.dropna(subset=["bin", "scheme"])


# ── table builders ───────────────────────────────────────────────────────────

def _build_auth_rate_history(df: pd.DataFrame) -> dict:
    result: dict = {}
    for (bin_str, scheme), grp in df.groupby(["bin", "scheme"]):
        fraud_rate = float(grp["isFraud"].mean())
        auth_rate  = round(max(0.50, 1.0 - fraud_rate), 4)
        result[(bin_str, scheme)] = {
            "rate_30d":   auth_rate,
            "rate_90d":   round(max(0.50, auth_rate - 0.003), 4),
            "volume_30d": len(grp),
        }
    return result


def _build_velocity(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["TransactionDT"])
    result: dict = {}
    for (bin_str, merchant_id), grp in df.groupby(["bin", "merchant_id"]):
        if len(grp) < 2:
            continue
        dts = grp["TransactionDT"].values.astype(float)
        ref = dts.max()
        result[(bin_str, merchant_id)] = {
            "txns_1h":  int(((ref - dts) <= _HOUR).sum()),
            "txns_24h": int(((ref - dts) <= _DAY).sum()),
        }
    return result


def _build_bin_table(df: pd.DataFrame) -> dict:
    """Returns tuples matching the BIN_TABLE format: (issuer_id, name, country, card_type, schemes)."""
    result: dict = {}
    for bin_str, grp in df.groupby("bin"):
        schemes   = sorted(grp["scheme"].unique().tolist())
        card_type = grp["card_type"].mode()[0]
        issuer_id = f"iss_{bin_str}"
        result[bin_str] = (
            issuer_id,
            f"EU Issuer {bin_str}",
            "FR",        # dataset is European (French origin)
            card_type,
            schemes,
        )
    return result


def _build_issuer_health(df: pd.DataFrame) -> dict:
    result: dict = {}
    for bin_str, grp in df.groupby("bin"):
        fraud_rate = float(grp["isFraud"].mean())
        issuer_id  = f"iss_{bin_str}"
        if fraud_rate > 0.10:
            result[issuer_id] = {
                "status":   "elevated_declines",
                "incident": f"Fraud rate {fraud_rate:.1%} exceeds 10% threshold",
            }
        else:
            result[issuer_id] = {"status": "healthy", "incident": None}
    return result


def _build_decline_patterns(df: pd.DataFrame) -> dict:
    """Approximates decline-reason distribution from the fraud rate.
    Non-fraud splits use industry-baseline ratios (42% insufficient_funds,
    28% do_not_honor, 30% other)."""
    df = df.copy()
    df["hour_of_day"] = (df["TransactionDT"].astype(float).astype(int) % _DAY) // _HOUR
    df["hour_bucket"] = df["hour_of_day"].apply(_hour_bucket)

    result: dict = {}
    for (bin_str, bucket), grp in df.groupby(["bin", "hour_bucket"]):
        fraud     = float(grp["isFraud"].mean())
        non_fraud = 1.0 - fraud
        result[(bin_str, bucket)] = {
            "insufficient_funds": round(non_fraud * 0.42, 4),
            "do_not_honor":       round(non_fraud * 0.28, 4),
            "fraud":              round(fraud,            4),
            "other":              round(non_fraud * 0.30, 4),
        }
    return result


# ── public entry point ───────────────────────────────────────────────────────

def load_kaggle_feature_tables(csv_path: str, nrows: int = 100_000) -> dict:
    """
    Read creditcard.csv and compute all feature-store tables.

    Returns a dict with keys:
      auth_rate_history, velocity, bin_table, issuer_health, decline_patterns
    """
    available = set(pd.read_csv(csv_path, nrows=0).columns)
    usecols   = [c for c in _WANT_COLS if c in available]

    df = pd.read_csv(csv_path, usecols=usecols, nrows=nrows)

    if "Class" not in df.columns:
        df["Class"] = 0.0

    df = _derive_fields(df)

    return {
        "auth_rate_history": _build_auth_rate_history(df),
        "velocity":          _build_velocity(df),
        "bin_table":         _build_bin_table(df),
        "issuer_health":     _build_issuer_health(df),
        "decline_patterns":  _build_decline_patterns(df),
    }


# ── disk-backed cache ────────────────────────────────────────────────────────

_CACHE_DIR  = pathlib.Path.home() / ".cache" / "cso_pipeline"
_CACHE_FILE = _CACHE_DIR / "features.pkl"


def load_kaggle_feature_tables_cached(csv_path: str, nrows: int = 100_000) -> dict:
    """
    load_kaggle_feature_tables() with a disk pickle cache keyed on CSV mtime.
    Survives Streamlit server restarts; recomputes only when the CSV changes.
    """
    csv_mtime = os.path.getmtime(csv_path)

    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "rb") as fh:
                cached = pickle.load(fh)
            if cached.get("mtime") == csv_mtime:
                return cached["tables"]
        except Exception:
            pass  # corrupt cache — fall through to recompute

    tables = load_kaggle_feature_tables(csv_path, nrows=nrows)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(_CACHE_FILE, "wb") as fh:
            pickle.dump({"mtime": csv_mtime, "tables": tables}, fh)
    except OSError:
        pass  # non-fatal: proceed without persisting

    return tables

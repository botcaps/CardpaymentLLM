"""
Kaggle Credit Card Fraud Detection dataset → list[Transaction].

Dataset: mlg-ulb/creditcardfraud  (public, no competition rule acceptance needed)
File:    creditcard.csv

Runtime fetch via `kagglehub`:
  Calling get_or_download_csv() triggers kagglehub.dataset_download(),
  which streams the dataset into the user's kagglehub cache
  (default: ~/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/<n>/).
  Subsequent calls hit the cache and return immediately.

  The repo never contains the CSV. Only the *trained* fraud model
  (ml/fraud_model.pkl, ~5-10 MB) is the persistent artifact.

Authentication:
  kagglehub uses the same ~/.kaggle/kaggle.json credentials as the
  legacy `kaggle` package. If the file is missing, kagglehub raises
  a clear setup error; we re-raise it with our own setup guide.

Column mapping (synthetic, for demo only):
  V1, V2    → bin           (deterministic 6-digit derived key)
  V3        → card_type     (negative → debit, positive → credit)
  bin % 4   → scheme        (0=visa, 1=mastercard, 2=amex, 3=discover)
  bin % 100 → merchant_id
  Amount    → amount_minor  (EUR cents)
  Amount    → mcc           (range-based category)
  Time      → hour_of_day   (seconds → hour of day)
  Class     → isFraud proxy (used for ML training only)
  Everything else fixed: EUR / EU / FR / ecommerce / none
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from contracts.models import Transaction

# ── runtime fetch via kagglehub ──────────────────────────────────────────────

_DATASET = "mlg-ulb/creditcardfraud"
_CSV_NAME = "creditcard.csv"

_SETUP_GUIDE = """
Kaggle credentials not found.

To set up your API key:
  1. Go to https://www.kaggle.com/settings/account
  2. Scroll to *API* → click 'Create New Token' (downloads kaggle.json)
  3. Move the file:  mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
  4. Set permissions: chmod 600 ~/.kaggle/kaggle.json
  5. Reload.

Note: credentials are only required to fetch the dataset. Once downloaded
to the kagglehub cache, subsequent runs do not need them.
"""


def get_or_download_csv() -> str:
    """
    Return local path to creditcard.csv.

    Lazily fetches the dataset via kagglehub on first call; subsequent calls
    hit the kagglehub cache (~/.cache/kagglehub/...). Returns the absolute
    path to the CSV inside the cache directory.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "The 'kagglehub' package is required. Install with: pip install kagglehub"
        ) from exc

    try:
        # Returns the local directory containing the dataset's files.
        # On first call this triggers the download; on later calls it returns
        # the cached path immediately.
        cache_dir = kagglehub.dataset_download(_DATASET)
    except Exception as exc:  # kagglehub raises generic exceptions on auth failure
        msg = str(exc).lower()
        if "credential" in msg or "auth" in msg or "401" in msg or "kaggle.json" in msg:
            raise RuntimeError(_SETUP_GUIDE) from exc
        raise RuntimeError(f"kagglehub download failed: {exc}") from exc

    csv_path = Path(cache_dir) / _CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{_CSV_NAME} not found in kagglehub cache at {cache_dir}.\n"
            f"Expected file: {csv_path}\n"
            "Manual fallback: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    return str(csv_path)


# ── column derivation ────────────────────────────────────────────────────────

_SCHEME_MAP = {0: "visa", 1: "mastercard", 2: "amex", 3: "discover"}

_AMOUNT_MCC = [
    (10_00,    "5812"),  # < €10    → restaurant
    (100_00,   "5999"),  # €10-100  → misc retail
    (1000_00,  "5411"),  # €100-1k  → grocery / retail
]
_DEFAULT_MCC = "4511"    # > €1000  → airline / travel


def _derive_bin(v1: float, v2: float) -> str | None:
    try:
        return str(abs(int(v1 * 10_000 + v2 * 1_000))).zfill(6)[:6]
    except (ValueError, TypeError, OverflowError):
        return None


def _amount_to_mcc(amount_minor: int) -> str:
    for threshold, mcc in _AMOUNT_MCC:
        if amount_minor < threshold:
            return mcc
    return _DEFAULT_MCC


_USECOLS = ["Time", "Amount", "Class", "V1", "V2", "V3"]


def load_kaggle_transactions(csv_path: str, n: int = 20) -> list[Transaction]:
    """Load up to *n* rows from creditcard.csv and map to Transaction objects."""
    df = pd.read_csv(csv_path, usecols=_USECOLS, nrows=max(n * 5, 1000))
    df = df.dropna(subset=["V1", "V2", "Amount"])
    df = df[df["Amount"] > 0]
    df = df.head(n)

    txns: list[Transaction] = []
    for idx, row in df.iterrows():
        bin_str = _derive_bin(float(row["V1"]), float(row["V2"]))
        if not bin_str:
            continue

        bin_int = int(bin_str)
        scheme = _SCHEME_MAP[bin_int % 4]
        card_type = "debit" if float(row["V3"]) < 0 else "credit"
        merchant_id = f"mer_{bin_int % 100}"

        try:
            amount_minor = max(1, int(round(float(row["Amount"]) * 100)))
        except (ValueError, TypeError):
            continue

        mcc = _amount_to_mcc(amount_minor)

        try:
            hour_of_day = int(float(row["Time"])) % 86400 // 3600
        except (ValueError, TypeError):
            hour_of_day = 14

        txns.append(Transaction(
            txn_id=f"txn_{idx}",
            bin=bin_str,
            card_type=card_type,
            card_brand_capabilities=[scheme],
            merchant_id=merchant_id,
            mcc=mcc,
            amount_minor=amount_minor,
            currency="EUR",
            channel="ecommerce",
            three_ds_status="none",
            region="EU",
            issuer_country="FR",
            acquirer_country="FR",
            hour_of_day=hour_of_day,
            is_network_token=False,
            token_network=None,
        ))

    return txns

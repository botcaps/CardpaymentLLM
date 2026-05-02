"""
Sample transactions used by the demo runner.

Each one is designed to hit a specific branch of the pipeline:
  1. Vanilla EU dual-brand card        -> cost tie-breaker
  2. Issuer with elevated declines     -> auth score penalty
  3. US travel transaction             -> high-interchange territory
  4. Debit card under Durbin           -> cap enforcement
  5. Amex at a non-OptBlue merchant    -> compliance rejection
  6. Network-token-locked PAN          -> only token network eligible
  7. Single-brand card (no choice)     -> degraded mode
  8. High-risk MCC (gambling)          -> cost + merchant eligibility
"""
from __future__ import annotations
from typing import Optional
from contracts.models import Transaction

SAMPLES: list[Transaction] = [
    Transaction(
        txn_id="txn_0001",
        bin="424242", card_type="credit",
        card_brand_capabilities=["visa", "mastercard"],
        merchant_id="mer_42", mcc="5411",
        amount_minor=4999, currency="EUR",
        channel="ecommerce", three_ds_status="authenticated_frictionless",
        region="EU", issuer_country="FR", acquirer_country="DE",
        hour_of_day=14,
    ),
    Transaction(
        txn_id="txn_0002",
        bin="424242", card_type="credit",
        card_brand_capabilities=["visa", "mastercard"],
        merchant_id="mer_101", mcc="5812",
        amount_minor=8750, currency="EUR",
        channel="pos", three_ds_status="none",
        region="EU", issuer_country="FR", acquirer_country="FR",
        hour_of_day=22,  # offhours — elevated declines + offhour fraud
    ),
    Transaction(
        txn_id="txn_0003",
        bin="520000", card_type="credit",
        card_brand_capabilities=["visa", "mastercard"],
        merchant_id="mer_77", mcc="4511",
        amount_minor=125_000, currency="USD",
        channel="ecommerce", three_ds_status="authenticated_frictionless",
        region="US", issuer_country="US", acquirer_country="US",
        hour_of_day=11,
    ),
    Transaction(
        txn_id="txn_0004",
        bin="540001", card_type="debit",
        card_brand_capabilities=["mastercard"],
        merchant_id="mer_42", mcc="5411",
        amount_minor=3200, currency="EUR",
        channel="pos", three_ds_status="none",
        region="EU", issuer_country="NL", acquirer_country="DE",
        hour_of_day=18,
    ),
    Transaction(
        txn_id="txn_0005",
        bin="378282", card_type="credit",
        card_brand_capabilities=["amex"],
        merchant_id="mer_9", mcc="5812",  # SmallTown Diner — NOT on OptBlue
        amount_minor=6200, currency="USD",
        channel="pos", three_ds_status="none",
        region="US", issuer_country="US", acquirer_country="US",
        hour_of_day=19,
    ),
    Transaction(
        txn_id="txn_0006",
        bin="520000", card_type="credit",
        card_brand_capabilities=["visa", "mastercard"],
        merchant_id="mer_77", mcc="7011",
        amount_minor=45_000, currency="USD",
        channel="ecommerce", three_ds_status="authenticated_frictionless",
        region="US", issuer_country="US", acquirer_country="US",
        hour_of_day=10,
        is_network_token=True,
        token_network="mastercard",  # token locks us to MC
    ),
    Transaction(
        txn_id="txn_0007",
        bin="601100", card_type="credit",
        card_brand_capabilities=["discover"],  # only one brand on the card
        merchant_id="mer_77", mcc="5999",
        amount_minor=7500, currency="USD",
        channel="ecommerce", three_ds_status="none",
        region="US", issuer_country="US", acquirer_country="US",
        hour_of_day=15,
    ),
    Transaction(
        txn_id="txn_0008",
        bin="450000", card_type="debit",
        card_brand_capabilities=["visa"],
        merchant_id="mer_9", mcc="5812",
        amount_minor=2499, currency="USD",
        channel="pos", three_ds_status="none",
        region="US", issuer_country="US", acquirer_country="US",
        hour_of_day=12,
    ),
]


def get_samples(
    source: str = "mock",
    csv_path: Optional[str] = None,
    n: int = 20,
) -> list[Transaction]:
    """
    Return a list of Transaction objects.

    source='mock'   → returns the hardcoded SAMPLES list (default)
    source='kaggle' → loads n rows from an IEEE-CIS train_transaction.csv
    """
    if source == "kaggle":
        if not csv_path:
            raise ValueError("csv_path is required when source='kaggle'")
        from data.kaggle_loader import load_kaggle_transactions
        return load_kaggle_transactions(csv_path, n=n)
    return list(SAMPLES)

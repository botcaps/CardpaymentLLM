"""
Synthetic interchange and scheme-fee tables.

Real schedules are hundreds of pages per network. These are simplified
demo figures — structurally faithful, numerically illustrative.
"""
from __future__ import annotations

# (scheme, region, card_type, mcc_tier) -> interchange in basis points
# mcc_tier is "supermarket" | "travel" | "standard" | "high_risk"
INTERCHANGE_TABLE = {
    # EU IFR caps dominate EU consumer card rates
    ("visa",       "EU", "credit", "standard"):   30.0,   # 0.30% IFR cap
    ("visa",       "EU", "debit",  "standard"):   20.0,   # 0.20% IFR cap
    ("mastercard", "EU", "credit", "standard"):   30.0,
    ("mastercard", "EU", "debit",  "standard"):   20.0,
    ("visa",       "EU", "credit", "supermarket"):28.0,
    ("mastercard", "EU", "credit", "supermarket"):28.0,

    # US rates are category-driven
    ("visa",       "US", "credit", "standard"):   155.0,
    ("visa",       "US", "credit", "supermarket"):110.0,
    ("visa",       "US", "credit", "travel"):     175.0,
    ("visa",       "US", "debit",  "standard"):   80.0,    # regulated vs exempt complicates this
    ("mastercard", "US", "credit", "standard"):   160.0,
    ("mastercard", "US", "debit",  "standard"):   85.0,
    ("amex",       "US", "credit", "standard"):   240.0,
    ("amex",       "US", "credit", "travel"):     265.0,
    ("discover",   "US", "credit", "standard"):   170.0,
}

# (scheme, region) -> fixed scheme/assessment fees in bps
SCHEME_FEES = {
    ("visa",       "EU"): {"assessment_bps": 11.0, "acquirer_bps": 2.0},
    ("mastercard", "EU"): {"assessment_bps": 12.0, "acquirer_bps": 2.0},
    ("visa",       "US"): {"assessment_bps": 14.0, "acquirer_bps": 3.0},
    ("mastercard", "US"): {"assessment_bps": 13.5, "acquirer_bps": 3.0},
    ("amex",       "US"): {"assessment_bps": 15.0, "acquirer_bps": 3.5},
    ("discover",   "US"): {"assessment_bps": 13.0, "acquirer_bps": 3.0},
}

# MCC -> tier mapping
MCC_TIERS = {
    "5411": "supermarket",   # grocery
    "5812": "standard",      # restaurants
    "5814": "standard",      # fast food
    "4511": "travel",        # airlines
    "7011": "travel",        # lodging
    "5999": "standard",      # misc retail
    "7995": "high_risk",     # gambling
}

def mcc_to_tier(mcc: str) -> str:
    return MCC_TIERS.get(mcc, "standard")


# Merchant contracts — which schemes can we even route through?
MERCHANT_CONTRACTS = {
    "mer_42":   {"name": "EuroGrocer GmbH",    "region": "EU", "accepts": ["visa", "mastercard"], "optblue": False},
    "mer_77":   {"name": "Transatlantic Air",  "region": "US", "accepts": ["visa", "mastercard", "amex", "discover"], "optblue": True},
    "mer_101":  {"name": "Cafe Parisien",      "region": "EU", "accepts": ["visa", "mastercard"], "optblue": False},
    "mer_9":    {"name": "SmallTown Diner",    "region": "US", "accepts": ["visa", "mastercard"], "optblue": False},  # Amex NOT enrolled
}

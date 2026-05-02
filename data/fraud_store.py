"""
Mock fraud intelligence data store.

In production these would be backed by a real-time fraud scoring service,
a velocity ledger, and a geo-risk model trained on historical chargebacks.
"""
from __future__ import annotations

# (bin, merchant_id) → recent transaction velocity
VELOCITY: dict[tuple[str, str], dict] = {
    ("424242", "mer_42"):  {"txns_1h": 1,  "txns_24h": 4},
    ("424242", "mer_101"): {"txns_1h": 9,  "txns_24h": 52},   # high-velocity — suspicious
    ("520000", "mer_77"):  {"txns_1h": 1,  "txns_24h": 2},
    ("540001", "mer_42"):  {"txns_1h": 2,  "txns_24h": 7},
    ("378282", "mer_9"):   {"txns_1h": 0,  "txns_24h": 1},
    ("601100", "mer_77"):  {"txns_1h": 3,  "txns_24h": 14},
    ("450000", "mer_9"):   {"txns_1h": 1,  "txns_24h": 3},
}

# (issuer_country, region) → additional p_fraud increment for cross-border usage
GEO_RISK: dict[tuple[str, str], float] = {
    ("FR", "EU"): 0.00,
    ("FR", "US"): 0.18,   # French card used in US — elevated cross-border risk
    ("US", "US"): 0.00,
    ("US", "EU"): 0.14,   # US card in EU — moderate
    ("NL", "EU"): 0.00,
    ("DE", "EU"): 0.00,
}

# Scheme fraud-detection network strength [0-1]
SCHEME_FRAUD_DEFENSE: dict[str, dict] = {
    "visa":       {"network_strength": 0.92, "notes": "VisaNet AI fraud scoring"},
    "mastercard": {"network_strength": 0.91, "notes": "Decision Intelligence"},
    "amex":       {"network_strength": 0.89, "notes": "Enhanced Authorisation"},
    "discover":   {"network_strength": 0.85, "notes": "Fraud Protection Guarantee"},
}

def inject_kaggle_fraud_features(tables: dict) -> None:
    """Update VELOCITY in-place from Kaggle-derived velocity table."""
    if "velocity" in tables:
        VELOCITY.update(tables["velocity"])


# (channel, three_ds_status) → base fraud probability
CHANNEL_3DS_RISK: dict[tuple[str, str], float] = {
    ("ecommerce", "authenticated_frictionless"): 0.02,
    ("ecommerce", "challenged"):                 0.04,
    ("ecommerce", "none"):                       0.12,   # CNP with no 3DS = high risk
    ("pos",       "authenticated_frictionless"): 0.01,
    ("pos",       "challenged"):                 0.02,
    ("pos",       "none"):                       0.03,
}

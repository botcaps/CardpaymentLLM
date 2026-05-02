"""
Synthetic feature store used by the Auth Score Agent tools.

All data here is fabricated for demo purposes. In production this would
be a real feature store (Feast, Tecton, or a read-replica of your auth log).
"""
from __future__ import annotations

# BIN -> (issuer_id, issuer_name, country, card_type, brand_capabilities)
BIN_TABLE = {
    "411111": ("iss_db_001", "Deutsche Bank",    "DE", "credit", ["visa"]),
    "424242": ("iss_bnp_002", "BNP Paribas",     "FR", "credit", ["visa", "mastercard"]),
    "555555": ("iss_db_001", "Deutsche Bank",    "DE", "credit", ["mastercard"]),
    "540001": ("iss_ing_003", "ING",             "NL", "debit",  ["mastercard"]),
    "378282": ("iss_amex_004","American Express","US", "credit", ["amex"]),
    "601100": ("iss_disc_005","Discover Bank",   "US", "credit", ["discover"]),
    "450000": ("iss_chase_006","Chase",          "US", "debit",  ["visa"]),
    "520000": ("iss_citi_007","Citi",            "US", "credit", ["visa", "mastercard"]),
}

# (bin, scheme) -> rolling auth rates
AUTH_RATE_HISTORY = {
    ("411111", "visa"):       {"rate_30d": 0.941, "rate_90d": 0.938, "volume_30d": 184_203},
    ("424242", "visa"):       {"rate_30d": 0.922, "rate_90d": 0.919, "volume_30d":  96_541},
    ("424242", "mastercard"): {"rate_30d": 0.935, "rate_90d": 0.931, "volume_30d":  88_102},
    ("555555", "mastercard"): {"rate_30d": 0.947, "rate_90d": 0.944, "volume_30d": 201_889},
    ("540001", "mastercard"): {"rate_30d": 0.965, "rate_90d": 0.962, "volume_30d": 312_440},  # debit, high auth
    ("378282", "amex"):       {"rate_30d": 0.918, "rate_90d": 0.915, "volume_30d":  45_221},
    ("601100", "discover"):   {"rate_30d": 0.889, "rate_90d": 0.891, "volume_30d":  12_004},
    ("450000", "visa"):       {"rate_30d": 0.958, "rate_90d": 0.955, "volume_30d": 270_550},
    ("520000", "visa"):       {"rate_30d": 0.931, "rate_90d": 0.928, "volume_30d": 145_880},
    ("520000", "mastercard"): {"rate_30d": 0.940, "rate_90d": 0.937, "volume_30d": 132_115},
}

# issuer_id -> current status
ISSUER_HEALTH = {
    "iss_db_001":   {"status": "healthy",            "incident": None},
    "iss_bnp_002":  {"status": "elevated_declines",  "incident": "intermittent 3DS timeouts"},
    "iss_ing_003":  {"status": "healthy",            "incident": None},
    "iss_amex_004": {"status": "healthy",            "incident": None},
    "iss_disc_005": {"status": "healthy",            "incident": None},
    "iss_chase_006":{"status": "healthy",            "incident": None},
    "iss_citi_007": {"status": "healthy",            "incident": None},
}

# (bin, hour_bucket) -> decline code distribution
#   hour_bucket: "business" (8-20 local) or "offhours"
DECLINE_PATTERNS = {
    ("411111", "business"): {"insufficient_funds": 0.41, "do_not_honor": 0.22, "fraud": 0.15, "other": 0.22},
    ("411111", "offhours"): {"insufficient_funds": 0.33, "do_not_honor": 0.28, "fraud": 0.21, "other": 0.18},
    ("424242", "business"): {"insufficient_funds": 0.36, "do_not_honor": 0.30, "fraud": 0.11, "other": 0.23},
    ("424242", "offhours"): {"insufficient_funds": 0.29, "do_not_honor": 0.34, "fraud": 0.18, "other": 0.19},
    ("540001", "business"): {"insufficient_funds": 0.52, "do_not_honor": 0.18, "fraud": 0.08, "other": 0.22},
}

def _hour_bucket(hour: int) -> str:
    return "business" if 8 <= hour <= 20 else "offhours"


def inject_kaggle_features(tables: dict) -> None:
    """Update feature-store dicts in-place from Kaggle-derived tables.
    Called once at dashboard startup when a Kaggle CSV is selected."""
    if "auth_rate_history" in tables:
        AUTH_RATE_HISTORY.update(tables["auth_rate_history"])
    if "bin_table" in tables:
        BIN_TABLE.update(tables["bin_table"])
    if "issuer_health" in tables:
        ISSUER_HEALTH.update(tables["issuer_health"])
    if "decline_patterns" in tables:
        DECLINE_PATTERNS.update(tables["decline_patterns"])

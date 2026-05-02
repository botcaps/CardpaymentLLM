"""
Tools exposed to the Auth Score Agent.

These would be wrapped as Claude tool_use schemas in production. Here
they're plain Python functions; the mocked agent calls them directly.
"""
from __future__ import annotations
from data.feature_store import (
    AUTH_RATE_HISTORY, ISSUER_HEALTH, DECLINE_PATTERNS, BIN_TABLE, _hour_bucket,
)


def feature_store_lookup(bin: str, scheme: str) -> dict | None:
    """Rolling 30/90-day auth rates for this BIN on this scheme."""
    return AUTH_RATE_HISTORY.get((bin, scheme))


def issuer_health_check(bin: str, scheme: str) -> dict:
    """Current issuer status for the bank behind this BIN."""
    if bin not in BIN_TABLE:
        return {"status": "unknown", "incident": None}
    issuer_id = BIN_TABLE[bin][0]
    return ISSUER_HEALTH.get(issuer_id, {"status": "unknown", "incident": None})


def scheme_decline_patterns(bin: str, hour_of_day: int) -> dict | None:
    """Decline-code distribution by time-of-day bucket."""
    bucket = _hour_bucket(hour_of_day)
    return DECLINE_PATTERNS.get((bin, bucket))

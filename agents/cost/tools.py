"""
Tools exposed to the Cost Agent.

In production these wrap the interchange schedule API and a
scheme-fee configuration service. Normalise_fee stays server-side
so the LLM never does final currency arithmetic.
"""
from __future__ import annotations
from data.interchange import (
    INTERCHANGE_TABLE, SCHEME_FEES, MERCHANT_CONTRACTS, mcc_to_tier,
)


def interchange_lookup(scheme: str, region: str, card_type: str, mcc: str) -> float | None:
    """Return interchange in bps for the (scheme, region, card_type, MCC tier) combo."""
    tier = mcc_to_tier(mcc)
    # Try exact, then fall back to 'standard' tier within same region/card_type
    bps = INTERCHANGE_TABLE.get((scheme, region, card_type, tier))
    if bps is None:
        bps = INTERCHANGE_TABLE.get((scheme, region, card_type, "standard"))
    return bps


def scheme_fee_lookup(scheme: str, region: str) -> dict | None:
    """Return assessment + acquirer fees in bps for (scheme, region)."""
    return SCHEME_FEES.get((scheme, region))


def normalise_fee(interchange_bps: float, assessment_bps: float,
                  acquirer_bps: float, txn_amount_minor: int) -> dict:
    """
    Normalise fee components to a single total_fee_bps.

    Real schedules also have fixed per-transaction cents (e.g. Durbin
    21c + 5bps + 1c); we keep it to bps here for simplicity.
    """
    total = interchange_bps + assessment_bps + acquirer_bps
    fee_minor = round(txn_amount_minor * total / 10_000, 2)
    return {
        "interchange_bps": round(interchange_bps, 2),
        "assessment_bps":  round(assessment_bps, 2),
        "acquirer_bps":    round(acquirer_bps, 2),
        "total_bps":       round(total, 2),
        "fee_minor":       fee_minor,
    }


def merchant_accepts(merchant_id: str, scheme: str) -> bool:
    """Contract check — does this merchant accept this scheme at all?"""
    contract = MERCHANT_CONTRACTS.get(merchant_id)
    if not contract:
        return False
    return scheme in contract["accepts"]

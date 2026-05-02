"""
Tools exposed to the Fraud Risk Agent.

Each tool returns a dict describing one fraud signal. The agent accumulates
signals from all tools before calling emit_fraud_score with the final p_fraud.
"""
from __future__ import annotations
from data.fraud_store import VELOCITY, GEO_RISK, SCHEME_FRAUD_DEFENSE, CHANNEL_3DS_RISK


def velocity_check(bin: str, merchant_id: str) -> dict:
    """Return transaction velocity for this BIN at this merchant (last 1h and 24h)."""
    return VELOCITY.get((bin, merchant_id), {"txns_1h": 0, "txns_24h": 0})


def geo_anomaly_check(issuer_country: str, region: str) -> dict:
    """Return geo-risk score when the card issuer country doesn't match the transaction region."""
    risk = GEO_RISK.get((issuer_country, region), 0.05)   # unknown combo → mild risk
    return {
        "issuer_country":       issuer_country,
        "transaction_region":   region,
        "geo_risk_increment":   risk,
        "cross_border":         risk > 0,
    }


def device_risk_score(channel: str, three_ds_status: str) -> dict:
    """Return fraud risk contribution from the channel + 3DS status combination."""
    risk = CHANNEL_3DS_RISK.get((channel, three_ds_status), 0.05)
    return {
        "channel":         channel,
        "three_ds_status": three_ds_status,
        "base_fraud_risk": risk,
    }


def scheme_fraud_defense(scheme: str) -> dict:
    """Return the fraud-detection network strength for this payment scheme."""
    return SCHEME_FRAUD_DEFENSE.get(
        scheme, {"network_strength": 0.80, "notes": "unknown scheme"}
    )

"""
Score aggregation.

Implements: scheme* = argmax [ w1*p_auth - w2*norm(fee) - w3*p_fraud ]
with w1 >> w2, w3 as specified in the implementation guide.
"""
from __future__ import annotations
from contracts.models import AuthScore, CostScore, FraudScore

W1 = 1.0     # auth weight
W2 = 0.15    # cost weight
W3 = 0.30    # fraud weight
FEE_CLAMP_BPS = 200.0


def aggregate(
    auth_scores:  list[AuthScore],
    cost_scores:  list[CostScore],
    fraud_scores: list[FraudScore] | None = None,
) -> list[dict]:
    """Return list of ranked scheme entries, highest weighted score first."""
    auth_by_scheme  = {a.scheme: a for a in auth_scores}
    cost_by_scheme  = {c.scheme: c for c in cost_scores}
    fraud_by_scheme = {f.scheme: f for f in (fraud_scores or [])}

    schemes = set(auth_by_scheme) & set(cost_by_scheme)
    ranked: list[dict] = []

    for scheme in schemes:
        a = auth_by_scheme[scheme]
        c = cost_by_scheme[scheme]
        norm_fee = min(c.total_fee_bps / FEE_CLAMP_BPS, 1.0)
        score = W1 * a.p_auth - W2 * norm_fee

        p_fraud = None
        fraud_reasoning = None
        if scheme in fraud_by_scheme:
            f = fraud_by_scheme[scheme]
            p_fraud = f.p_fraud
            fraud_reasoning = f.reasoning
            score -= W3 * p_fraud

        ranked.append({
            "scheme":           scheme,
            "p_auth":           a.p_auth,
            "confidence":       a.confidence,
            "total_fee_bps":    c.total_fee_bps,
            "interchange_bps":  c.breakdown["interchange_bps"],
            "p_fraud":          p_fraud,
            "weighted_score":   round(score, 4),
            "auth_reasoning":   a.reasoning,
            "cost_reasoning":   c.reasoning,
            "fraud_reasoning":  fraud_reasoning,
        })

    ranked.sort(key=lambda e: e["weighted_score"], reverse=True)
    return ranked

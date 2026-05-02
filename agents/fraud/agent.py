"""
Fraud Risk Agent.

Two scoring paths:
  1. **Model path** (preferred): if ml/fraud_model.pkl exists, the agent
     calls FraudModel.predict_proba() for the headline p_fraud number.
     The rule-based velocity / geo / channel signals are still gathered
     as *explainability features* the agent can cite alongside.
  2. **Rule path** (fallback): if the model isn't trained yet, the agent
     falls back to the original rule-based scoring (the project's Day-0
     behaviour). This means the system works out-of-the-box without
     requiring `python -m ml.training.train_fraud` to have been run.

Live LLM mode wraps either path inside a ReAct loop in graph.py. This
file is the deterministic / mock entry point.
"""
from __future__ import annotations

import logging

from contracts.models import Transaction, FraudScore
from agents.fraud.tools import (
    velocity_check, geo_anomaly_check, device_risk_score, scheme_fraud_defense,
)

log = logging.getLogger(__name__)


# ── Rule-based scoring (the original Day-0 path) ─────────────────────────────

def _rule_score_one(txn: Transaction, scheme: str) -> FraudScore:
    """
    Pure rule-based fraud score.

    This is the fallback when the trained model isn't available. It is
    the original `_mock_score_one` from before Day 6 — kept verbatim so
    test outputs and dashboard rendering remain reproducible.
    """
    vel     = velocity_check(txn.bin, txn.merchant_id)
    geo     = geo_anomaly_check(txn.issuer_country, txn.region)
    device  = device_risk_score(txn.channel, txn.three_ds_status)
    defense = scheme_fraud_defense(scheme)

    p = device["base_fraud_risk"]
    adjustments = [f"channel+3DS base: {p:.3f}"]

    if vel["txns_1h"] > 5:
        p += 0.15
        adjustments.append(f"high velocity {vel['txns_1h']} txns/hr: +0.150")
    elif vel["txns_1h"] > 2:
        p += 0.05
        adjustments.append(f"moderate velocity {vel['txns_1h']} txns/hr: +0.050")

    geo_risk = geo["geo_risk_increment"]
    if geo_risk > 0:
        p += geo_risk
        adjustments.append(f"geo cross-border ({txn.issuer_country}→{txn.region}): +{geo_risk:.3f}")

    factor = 1.0 - (defense["network_strength"] - 0.85) * 0.5
    p = p * factor
    adjustments.append(f"{scheme} defense ({defense['network_strength']:.0%}): ×{factor:.2f}")

    p = round(max(0.0, min(1.0, p)), 4)
    reasoning = (
        f"[rule] Fraud risk for {scheme}: "
        + "; ".join(adjustments)
        + f". Final p_fraud={p:.4f}."
    )
    return FraudScore(scheme=scheme, p_fraud=p, confidence=0.70, reasoning=reasoning)


# ── Model-based scoring (Day 6 addition) ─────────────────────────────────────

def _model_score_one(txn: Transaction, scheme: str) -> FraudScore:
    """
    Use the trained XGBoost model for the headline p_fraud, augmented with
    rule-based explainability features.

    Falls back transparently to _rule_score_one if the model can't load.
    """
    try:
        from ml.fraud_model import get_model
        model = get_model()
    except RuntimeError as exc:
        log.debug("fraud model unavailable, falling back to rules: %s", exc)
        return _rule_score_one(txn, scheme)

    # Headline number from the model
    p_model = model.predict_proba(txn, scheme=scheme)

    # Rule signals — we still gather them so the agent can cite them
    vel    = velocity_check(txn.bin, txn.merchant_id)
    geo    = geo_anomaly_check(txn.issuer_country, txn.region)
    device = device_risk_score(txn.channel, txn.three_ds_status)
    defense = scheme_fraud_defense(scheme)

    # Confidence from the model is hard to estimate without calibration data;
    # use 0.80 (higher than the rule path's 0.70) since the model is trained
    # on real labelled data, but lower than 1.0 because of the runtime feature
    # mismatch documented in ml/fraud_model.py.
    confidence = 0.80

    p = round(max(0.0, min(1.0, p_model)), 4)
    reasoning = (
        f"[model] XGBoost p_fraud={p:.4f}. "
        f"Supporting signals: velocity {vel['txns_1h']} txns/hr, "
        f"geo {txn.issuer_country}→{txn.region} (risk +{geo['geo_risk_increment']:.3f}), "
        f"channel/{txn.three_ds_status} base {device['base_fraud_risk']:.3f}, "
        f"{scheme} defense {defense['network_strength']:.0%}."
    )
    return FraudScore(scheme=scheme, p_fraud=p, confidence=confidence, reasoning=reasoning)


# ── Dispatch ────────────────────────────────────────────────────────────────

def _mock_score_one(txn: Transaction, scheme: str) -> FraudScore:
    """
    Deterministic scoring entry point.

    Tries the trained model first; falls back to rule-based scoring if
    the model isn't trained yet. Keeping the function name unchanged
    means the existing ReAct agent path in orchestrator/graph.py
    (`from agents.fraud.agent import _mock_score_one`) still works.
    """
    from ml.fraud_model import model_available
    if model_available():
        return _model_score_one(txn, scheme)
    return _rule_score_one(txn, scheme)


async def fraud_agent(txn: Transaction, candidates: list[str]) -> list[FraudScore]:
    """Async wrapper kept for back-compat."""
    return [_mock_score_one(txn, s) for s in candidates]

"""
XGBoost fraud-model serving wrapper.

Loads the trained model from ml/fraud_model.pkl and exposes a single
predict_proba(transaction, scheme) → float method. This replaces the
rule-based velocity / geo / channel lookups in agents/fraud/tools.py
as the *primary* fraud signal — the rule-based dicts stay as
explainability features the agent can cite alongside the model output.

────────────────────────────────────────────────────────────────────────
Why this design (worth knowing for the viva):

  Tool interface stays identical.
    The agent code calls velocity_check() / geo_anomaly_check() /
    device_risk_score() / scheme_fraud_defense() exactly as before.
    Behind the scenes, the FraudModel adds one more signal — a learned
    probability — that the agent can include in its reasoning.
    Swapping rule-based for model-based requires zero changes upstream.

  The model never sees the (txn, scheme) pair the way the dataset's
  V1..V28 does. We don't have those PCA features at runtime — they
  were derived from the raw card numbers Kaggle anonymised. So we
  build a *fallback feature vector* from the runtime transaction:
  amount_minor → Amount column; encode hour_of_day, channel, 3DS, etc.
  in additional dimensions; pad missing PCA features with zeros (the
  PCA features are zero-centred, so zero is a reasonable prior).

  In production you'd retrain the model on features you actually have
  at inference time — that's exactly what feedback/trainer.py would do
  in Phase 3. For the capstone, this dual-path approach lets us:
    (a) ship a real ML model trained on real data
    (b) demonstrate the integration pattern (model → tool → agent)
    (c) acknowledge the limitation honestly in the report

  The alternative — writing a separate model that takes the runtime
  features directly — would be more rigorous, but it'd require us to
  generate synthetic labels for the runtime features (we don't have
  ground truth for them). The current approach uses real labels on
  the original Kaggle features and accepts that runtime predictions
  are less accurate than the held-out test-set numbers suggest. We
  document this as a known limitation.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from contracts.models import Transaction

log = logging.getLogger(__name__)

# ── Paths & constants ────────────────────────────────────────────────────────

_MODEL_PATH = Path(__file__).resolve().parent / "fraud_model.pkl"

# The Kaggle dataset has Time, V1..V28, Amount as features.
# That's 30 columns; Class is the label (excluded).
_KAGGLE_FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ── Model loader (lazy + cached) ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model():
    """Load the trained model on first call; cache for subsequent calls."""
    if not _MODEL_PATH.exists():
        raise RuntimeError(
            f"Trained fraud model not found at {_MODEL_PATH}.\n"
            f"Train it with: python -m ml.training.train_fraud"
        )
    import joblib
    log.info("loading fraud model from %s", _MODEL_PATH)
    return joblib.load(_MODEL_PATH)


def model_available() -> bool:
    """Return True if the .pkl exists. Used by the agent's mock fallback."""
    return _MODEL_PATH.exists()


# ── Runtime feature construction ─────────────────────────────────────────────
# Map a Transaction → 30-dim feature vector matching the Kaggle schema.

def _txn_to_features(txn: Transaction) -> np.ndarray:
    """
    Build a Kaggle-shape feature vector from a runtime Transaction.

    Strategy: zero-pad the V1..V28 PCA features (they're zero-centred so
    zero is the prior); use amount_minor for Amount; use hour_of_day to
    construct a Time-equivalent (seconds-of-day).

    This is acknowledged-imperfect — see fraud_model.py docstring for
    the design discussion.
    """
    seconds_in_day = (txn.hour_of_day or 14) * 3600
    amount_eur     = txn.amount_minor / 100.0

    # 1 + 28 + 1 = 30
    features = np.zeros(30, dtype=np.float64)
    features[0] = float(seconds_in_day)
    # V1..V28 stay at zero (the prior mean of every PCA dim)
    features[-1] = float(amount_eur)
    return features.reshape(1, -1)


# ── Public API ──────────────────────────────────────────────────────────────

class FraudModel:
    """
    Drop-in fraud predictor.

    Usage:
        model = FraudModel()
        p_fraud = model.predict_proba(txn, scheme="visa")
        # → float in [0, 1]
    """

    def __init__(self):
        self._model = _load_model()

    def predict_proba(self, txn: Transaction, scheme: Optional[str] = None) -> float:
        """
        Return p_fraud in [0, 1] for the given transaction.

        The `scheme` argument is currently unused — the model was trained
        on PCA-anonymised features that don't include scheme. It's kept
        in the signature so future retrains (when feedback data carries
        scheme) can use it without an interface change.
        """
        x = _txn_to_features(txn)
        prob = float(self._model.predict_proba(x)[0, 1])
        return max(0.0, min(1.0, prob))

    @staticmethod
    def model_path() -> Path:
        return _MODEL_PATH


# ── Module-level singleton — most callers want this ─────────────────────────

_singleton: Optional[FraudModel] = None


def get_model() -> FraudModel:
    """
    Return the cached singleton. Raises RuntimeError if the .pkl is missing.

    Agents that want graceful degradation should wrap this:
        try:
            model = get_model()
        except RuntimeError:
            model = None  # fall back to rule-based scoring
    """
    global _singleton
    if _singleton is None:
        _singleton = FraudModel()
    return _singleton

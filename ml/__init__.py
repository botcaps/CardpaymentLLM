"""
ml/ — trained machine-learning models.

Contains:
  fraud_model.py        XGBoost serving wrapper (loaded by agents/fraud/tools.py)
  training/             offline training scripts
  fraud_model.pkl       persisted model (gitignored; rebuild via train script)

Design pattern: training is fully decoupled from serving. The serving
wrapper exposes the same interface as the original fraud_store dict
lookups, so agent code is untouched. Swapping rule-based lookups for a
trained model requires zero changes upstream.
"""

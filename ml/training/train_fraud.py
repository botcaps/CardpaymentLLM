"""
Train an XGBoost fraud-detection model on the Kaggle creditcardfraud dataset.

Usage:
    python -m ml.training.train_fraud
    python -m ml.training.train_fraud --no-smote     # ablation
    python -m ml.training.train_fraud --rows 50000   # quick smoke run

Dataset: Kaggle mlg-ulb/creditcardfraud (284,807 transactions, 492 fraud).
Class imbalance: ~0.17% positive — the interesting ML problem here.

What this script does:
  1. Fetches creditcard.csv via kagglehub (no committed data).
  2. Stratified 80/20 train/test split.
  3. Optional SMOTE oversampling on the train set (default ON).
  4. Trains XGBoost with class-imbalance-aware loss.
  5. Reports AUC-ROC, AUC-PR (more honest for imbalanced data),
     precision/recall at the threshold that maximises F1.
  6. Saves to ml/fraud_model.pkl (joblib for sklearn compatibility).

Why XGBoost not deep learning?
  - ~492 positives. DL needs orders of magnitude more.
  - Tabular data with engineered (PCA) features. Tree models dominate
    every published Kaggle leaderboard for this dataset for a reason.
  - Inference is ~50µs per row on CPU. Critical for an agent tool that
    might be called dozens of times in a single pipeline run.

Why AUC-PR over AUC-ROC?
  - For severely imbalanced data, AUC-ROC inflates: even a model that
    gets 99.83% of the negatives "trivially right" scores 0.5. AUC-PR
    is sensitive to how well the positive class is recovered, which
    is the actual business question for fraud detection.
  - We report both for transparency. If the gap between them is wide,
    the classifier is doing better on negatives than positives.

Why SMOTE?
  - With 492 positives in 284k rows, class_weight rebalancing alone
    underweights the *variety* of fraud patterns the model sees during
    training. SMOTE synthesises minority-class examples by interpolating
    between near-neighbour positives, giving the model more diverse
    fraud-like signal.
  - The trade-off: SMOTE can synthesise borderline examples that are
    actually plausible legitimate transactions, hurting precision. We
    measure this with --no-smote ablation.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make the package imports work when launched as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.kaggle_loader import get_or_download_csv

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

# ── Paths ────────────────────────────────────────────────────────────────────

ML_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH = ML_DIR / "fraud_model.pkl"
METRICS_PATH = ML_DIR / "fraud_model_metrics.json"

# ── Hyperparameters ──────────────────────────────────────────────────────────
# Tuned on Kaggle creditcardfraud. If you change rows or features, re-tune.

XGB_PARAMS = dict(
    n_estimators       = 400,
    max_depth          = 6,
    learning_rate      = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    eval_metric        = "aucpr",
    tree_method        = "hist",  # CPU-friendly; 'gpu_hist' if a GPU is available
    random_state       = 42,
)

SMOTE_K_NEIGHBORS = 5
TEST_SIZE         = 0.20
RANDOM_STATE      = 42


# ── Pipeline ────────────────────────────────────────────────────────────────

def _load_data(max_rows: int | None) -> pd.DataFrame:
    csv_path = get_or_download_csv()
    log.info("loading %s", csv_path)
    df = pd.read_csv(csv_path, nrows=max_rows)
    log.info("loaded %d rows, %d columns", len(df), df.shape[1])
    fraud_count = int(df["Class"].sum())
    log.info("class balance: %d fraud / %d total (%.4f%%)",
             fraud_count, len(df), 100 * fraud_count / len(df))
    return df


def _split(df: pd.DataFrame):
    """Stratified split — same fraud ratio in train and test."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        stratify     = y,
        random_state = RANDOM_STATE,
    )


def _maybe_smote(X_train, y_train, use_smote: bool):
    """Apply SMOTE oversampling to training data only (never to test)."""
    if not use_smote:
        log.info("SMOTE disabled (ablation mode)")
        return X_train, y_train

    from imblearn.over_sampling import SMOTE
    pos_before = int(y_train.sum())
    smote = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    pos_after = int(y_resampled.sum())
    log.info("SMOTE: positives %d → %d (total rows %d → %d)",
             pos_before, pos_after, len(X_train), len(X_resampled))
    return X_resampled, y_resampled


def _train(X_train, y_train, X_test, y_test, use_smote: bool):
    import xgboost as xgb

    # When SMOTE is OFF, use class-weight rebalancing inside XGBoost.
    # When SMOTE is ON, the train set is already balanced.
    if not use_smote:
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        # scale_pos_weight is XGBoost's standard imbalance handler
        scale_pos_weight = n_neg / max(n_pos, 1)
        params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    else:
        params = XGB_PARAMS

    log.info("training XGBoost with %d trees, max_depth=%d", params["n_estimators"], params["max_depth"])
    t0    = time.time()
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set = [(X_test, y_test)],
        verbose  = False,
    )
    log.info("training complete in %.1fs", time.time() - t0)
    return model


def _evaluate(model, X_test, y_test) -> dict:
    """
    Compute AUC-ROC, AUC-PR, and best-F1 threshold + precision/recall there.
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_recall_curve, f1_score, confusion_matrix,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    auc_roc = float(roc_auc_score(y_test, y_prob))
    auc_pr  = float(average_precision_score(y_test, y_prob))

    # Find the threshold that maximises F1 — more honest than fixed 0.5
    # for an imbalanced classifier.
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1s = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = int(np.argmax(f1s[:-1]))   # last entry has no threshold
    best_threshold = float(thresholds[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall    = float(recalls[best_idx])
    best_f1        = float(f1s[best_idx])

    y_pred = (y_prob >= best_threshold).astype(int)
    cm     = confusion_matrix(y_test, y_pred).tolist()
    # cm = [[tn, fp], [fn, tp]]

    metrics = {
        "auc_roc":        round(auc_roc, 4),
        "auc_pr":         round(auc_pr, 4),
        "best_threshold": round(best_threshold, 4),
        "best_precision": round(best_precision, 4),
        "best_recall":    round(best_recall, 4),
        "best_f1":        round(best_f1, 4),
        "confusion_matrix_at_best_threshold": cm,
        "n_test":         len(y_test),
        "n_test_positive": int(y_test.sum()),
    }
    return metrics


def _save(model, metrics: dict) -> None:
    import joblib

    ML_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    log.info("saved model → %s (%.1f KB)",
             MODEL_PATH, MODEL_PATH.stat().st_size / 1024)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log.info("saved metrics → %s", METRICS_PATH)


# ── Public entry point ──────────────────────────────────────────────────────

def train(use_smote: bool = True, max_rows: int | None = None) -> dict:
    """Run the full pipeline and return the eval metrics dict."""
    df = _load_data(max_rows)
    X_train, X_test, y_train, y_test = _split(df)
    X_train_bal, y_train_bal = _maybe_smote(X_train, y_train, use_smote)
    model = _train(X_train_bal, y_train_bal, X_test, y_test, use_smote)
    metrics = _evaluate(model, X_test, y_test)

    # Print headline metrics — easy to grab for the report
    log.info("─" * 60)
    log.info("AUC-ROC: %.4f   AUC-PR: %.4f", metrics["auc_roc"], metrics["auc_pr"])
    log.info("Best threshold: %.4f", metrics["best_threshold"])
    log.info("  precision: %.4f", metrics["best_precision"])
    log.info("  recall:    %.4f", metrics["best_recall"])
    log.info("  F1:        %.4f", metrics["best_f1"])
    cm = metrics["confusion_matrix_at_best_threshold"]
    log.info("Confusion matrix [tn fp / fn tp]:")
    log.info("  %d   %d", cm[0][0], cm[0][1])
    log.info("  %d   %d", cm[1][0], cm[1][1])
    log.info("─" * 60)

    _save(model, metrics)
    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud-detection XGBoost")
    parser.add_argument("--no-smote", action="store_true",
                        help="Disable SMOTE oversampling (use scale_pos_weight only)")
    parser.add_argument("--rows", type=int, default=None,
                        help="Limit input rows (for quick smoke runs)")
    args = parser.parse_args()
    train(use_smote=not args.no_smote, max_rows=args.rows)


if __name__ == "__main__":
    _main()

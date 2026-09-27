"""
Branch A — ML Ensemble.

Combines:
  - Random Forest (bagging: many trees on bootstrapped samples, votes)
  - XGBoost (boosting: sequential trees correcting prior errors)
  - SVM with RBF kernel (non-linear margin classifier)

into a soft-voting ensemble (probabilities averaged), which genuinely
implements the "bagging + boosting combined for a lower-variance,
lower-bias ensemble" claim from the report.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def build_ensemble(cfg: dict, n_classes: int) -> VotingClassifier:
    mc = cfg["ml_branch"]["models"]
    seed = cfg["project"]["seed"]

    rf = RandomForestClassifier(
        n_estimators=mc["random_forest"]["n_estimators"],
        max_depth=mc["random_forest"]["max_depth"],
        random_state=seed, n_jobs=-1,
    )
    # SVC(probability=True) is deprecated in sklearn>=1.9; use CalibratedClassifierCV instead.
    svm = CalibratedClassifierCV(
        SVC(kernel=mc["svm"]["kernel"], C=mc["svm"]["C"], random_state=seed),
        ensemble=False,
    )
    # num_class is set explicitly (rather than left to XGBoost's auto-inference)
    # because a training subsample (e.g. the chronological/stratified caps used
    # on large real datasets) can happen to be missing a very rare class
    # entirely, which otherwise makes XGBoost infer the wrong number of
    # classes and crash ("num_class should be >= 1").
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    xgb_kwargs = dict(
        n_estimators=mc["xgboost"]["n_estimators"],
        max_depth=mc["xgboost"]["max_depth"],
        learning_rate=mc["xgboost"]["learning_rate"],
        random_state=seed, eval_metric="logloss", objective=objective,
        n_jobs=-1,
    )
    if n_classes > 2:
        xgb_kwargs["num_class"] = n_classes
    xgb = XGBClassifier(**xgb_kwargs)

    ensemble = VotingClassifier(
        estimators=[("random_forest", rf), ("svm", svm), ("xgboost", xgb)],
        voting="soft", n_jobs=1,
    )
    return ensemble


def train_ensemble(X_train, y_train, cfg, n_classes) -> VotingClassifier:
    ensemble = build_ensemble(cfg, n_classes)
    ensemble.fit(X_train, y_train)
    return ensemble


def save_ensemble(ensemble, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, path)


def load_ensemble(path: str):
    return joblib.load(path)


def predict_proba(ensemble, X) -> np.ndarray:
    return ensemble.predict_proba(X)

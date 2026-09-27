import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config
from src.ml.ensemble import build_ensemble, train_ensemble, predict_proba
from src.ml.shap_selection import select_features

CFG = load_config(str(Path(__file__).resolve().parents[1] / "config.yaml"))


@pytest.fixture(scope="module")
def toy_data():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = (X["f0"] + X["f1"] > 0).astype(int).values
    return X, y


def test_ensemble_builds_with_three_estimators():
    ens = build_ensemble(CFG, n_classes=2)
    names = [name for name, _ in ens.estimators]
    assert set(names) == {"random_forest", "svm", "xgboost"}


def test_ensemble_trains_and_predicts(toy_data):
    X, y = toy_data
    ens = train_ensemble(X.values, y, CFG, n_classes=2)
    proba = predict_proba(ens, X.values)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_shap_feature_selection_returns_valid_subset(toy_data):
    X, y = toy_data
    small_cfg = {**CFG, "ml_branch": {**CFG["ml_branch"],
                 "shap_feature_selection": {"enabled": True, "top_k_features": 3}}}
    selected, importance_df = select_features(X, y, small_cfg, results_dir="/tmp/shap_test")
    assert len(selected) == 3
    assert set(selected).issubset(set(X.columns))
    assert list(importance_df.columns) == ["feature", "mean_abs_shap"]
    assert importance_df["mean_abs_shap"].is_monotonic_decreasing

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config
from src.preprocessing.pipeline import clean, encode, load_dataset

CFG = load_config(str(Path(__file__).resolve().parents[1] / "config.yaml"))


def test_load_sample_dataset():
    df = load_dataset("sample", CFG)
    assert len(df) > 0
    assert "Label" in df.columns


def test_clean_handles_missing_and_infinite_values():
    df = pd.DataFrame({
        "Flow Bytes/s": [1.0, np.nan, np.inf, 4.0],
        "Flow Duration": [10, 20, 30, 40],
        "Label": ["BENIGN", "BENIGN", "DoS", "DoS"],
    })
    cleaned = clean(df, CFG)
    assert cleaned["Flow Bytes/s"].isna().sum() == 0
    assert not np.isinf(cleaned["Flow Bytes/s"]).any()


def test_clean_drops_identifier_columns():
    df = pd.DataFrame({
        "Flow ID": ["a", "b"], "Src IP": ["1.1.1.1", "2.2.2.2"],
        "Flow Duration": [1, 2], "Label": ["BENIGN", "DoS"],
    })
    cleaned = clean(df, CFG)
    assert "Flow ID" not in cleaned.columns
    assert "Src IP" not in cleaned.columns


def test_encode_produces_numeric_features_and_label():
    df = pd.DataFrame({
        "Flow Duration": [1, 2, 3], "Protocol": ["TCP", "UDP", "TCP"],
        "Label": ["BENIGN", "DoS", "BENIGN"],
    })
    feature_df, y, y_encoder = encode(df, CFG)
    assert feature_df.select_dtypes(exclude=[np.number]).shape[1] == 0
    assert set(np.unique(y)) == {0, 1}
    assert list(y_encoder.classes_) == ["BENIGN", "DoS"]


def test_full_pipeline_no_leakage_shapes():
    from src.preprocessing.pipeline import run_pipeline
    import copy
    sample_cfg = copy.deepcopy(CFG)
    sample_cfg["data"]["dataset"] = "sample"  # test against the small bundled sample set regardless of config.yaml's current default
    data = run_pipeline(sample_cfg, save=False)
    total = len(data.X_train) + len(data.X_val) + len(data.X_test)
    df = load_dataset("sample", sample_cfg)
    df = clean(df, sample_cfg)
    assert total == len(df)
    # test set must never have been touched by SMOTE-Tomek oversampling
    assert data.X_train_res.shape[0] >= data.X_train.shape[0]
    assert data.X_test.shape[0] == len(data.y_test)

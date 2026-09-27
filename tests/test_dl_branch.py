import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.utils.config import load_config
from src.dl.sequence import make_sequences
from src.dl.model import build_model

CFG = load_config(str(Path(__file__).resolve().parents[1] / "config.yaml"))


def test_make_sequences_shapes_and_no_out_of_range():
    X = np.arange(50 * 4).reshape(50, 4).astype(float)
    y = np.arange(50)
    seqs, labels = make_sequences(X, y, window_size=5, stride=1)
    assert seqs.shape == (46, 5, 4)
    assert labels.shape == (46,)
    # label of first window must be the LAST row of that window
    assert labels[0] == y[4]
    assert np.array_equal(seqs[0], X[0:5])


def test_make_sequences_raises_if_too_few_samples():
    X = np.zeros((3, 2))
    y = np.zeros(3)
    with pytest.raises(ValueError):
        make_sequences(X, y, window_size=5)


def test_dl_model_builds_and_predicts_expected_shapes():
    window_size, n_features, n_classes = 6, 5, 3
    model = build_model(window_size, n_features, n_classes, CFG)
    X = np.random.default_rng(0).normal(size=(4, window_size, n_features)).astype(np.float32)
    proba, attn = model.predict(X, verbose=0)
    assert proba.shape == (4, n_classes)
    assert attn.shape == (4, window_size)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert np.allclose(attn.sum(axis=1), 1.0, atol=1e-4)  # softmax attention weights

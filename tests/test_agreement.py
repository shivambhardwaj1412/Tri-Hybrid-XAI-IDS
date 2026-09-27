import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.agreement.agreement_check import jaccard_topk, recency_alignment, cross_explainer_agreement


def _approx(x, tol=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _Approx()


def test_jaccard_identical_sets_is_one():
    a = {"f1": 0.5, "f2": 0.3}
    b = {"f1": 0.1, "f2": 0.9}
    assert jaccard_topk(a, b) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    a = {"f1": 0.5}
    b = {"f2": 0.5}
    assert jaccard_topk(a, b) == 0.0


def test_recency_alignment_recent_focus_scores_high():
    attn = np.array([0.0, 0.0, 0.0, 0.1, 0.9])  # concentrated on last steps
    assert recency_alignment(attn, recency_window=2) == _approx(1.0)


def test_recency_alignment_distant_focus_scores_low():
    attn = np.array([0.9, 0.1, 0.0, 0.0, 0.0])  # concentrated on old steps
    assert recency_alignment(attn, recency_window=2) == _approx(0.0)


def test_high_agreement_case_is_trusted():
    shap_top = {"f1": 0.5, "f2": 0.3}
    lime_top = {"f1": 0.4, "f2": 0.2}
    attn = np.array([0.0, 0.0, 0.1, 0.2, 0.7])
    result = cross_explainer_agreement(shap_top, lime_top, attn, threshold=0.5)
    assert result["status"] == "TRUSTED"
    assert result["feature_agreement_jaccard"] == 1.0


def test_low_agreement_case_flags_review():
    shap_top = {"f1": 0.5}
    lime_top = {"f9": 0.4}
    attn = np.array([0.9, 0.05, 0.02, 0.02, 0.01])
    result = cross_explainer_agreement(shap_top, lime_top, attn, threshold=0.5)
    assert result["status"] == "REVIEW REQUIRED"
    assert result["reason"] is not None

"""
Cross-Explainer Agreement Check — proposed methodology of this project
(NOT a peer-reviewed metric; defined and implemented here for the first
time in this codebase).

Combines two independent signals into one Cross-Explainer Agreement Score
in [0, 1]:

  1. feature_agreement = Jaccard(top-K SHAP features, top-K LIME features)
     Both SHAP and LIME explain the SAME flow in the SAME feature space,
     so a direct top-K overlap is meaningful: if two independent
     feature-attribution methods point at different features for the same
     prediction, that prediction's explanation is less trustworthy.

  2. recency_alignment = attention weight mass placed on the last
     `recency_window` time-steps of the DL branch's input sequence.
     SHAP/LIME explain the CURRENT flow (the last step of the window); if
     the DL branch's attention is concentrated on old, distant flows
     instead of the current one, its notion of "why" is not actually about
     the same event SHAP/LIME are describing, so we down-weight agreement.

  final_score = 0.5 * feature_agreement + 0.5 * recency_alignment

  score >= threshold  -> "TRUSTED"
  score <  threshold  -> "REVIEW REQUIRED"

The 0.5/0.5 weighting and the threshold are both configurable in
config.yaml (`agreement.trusted_threshold`) and are explicitly a design
choice, not a scientifically validated constant.
"""
from __future__ import annotations
import numpy as np


def jaccard_topk(dict_a: dict, dict_b: dict) -> float:
    set_a, set_b = set(dict_a.keys()), set(dict_b.keys())
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def recency_alignment(attn_weights_for_window: np.ndarray, recency_window: int = 2) -> float:
    return float(np.sum(attn_weights_for_window[-recency_window:]))


def cross_explainer_agreement(shap_top: dict, lime_top: dict, attn_weights_for_window: np.ndarray,
                               threshold: float = 0.5, recency_window: int = 2) -> dict:
    feature_agreement = jaccard_topk(shap_top, lime_top)
    recency = recency_alignment(attn_weights_for_window, recency_window)
    score = 0.5 * feature_agreement + 0.5 * recency

    status = "TRUSTED" if score >= threshold else "REVIEW REQUIRED"
    reason = None
    if status == "REVIEW REQUIRED":
        if feature_agreement < 0.3:
            reason = "SHAP and LIME disagree on which features drove this prediction."
        elif recency < 0.3:
            reason = "DL attention is focused on distant past flows, not the current one SHAP/LIME explain."
        else:
            reason = "Combined agreement score below the configured trust threshold."

    return {
        "feature_agreement_jaccard": round(feature_agreement, 4),
        "recency_alignment": round(recency, 4),
        "agreement_score": round(score, 4),
        "status": status,
        "reason": reason,
    }

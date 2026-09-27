"""
Stage 4 — Multi-Method Explainability Layer.

For a single flow (identified by its predicted class), produces three
independent explanations:

  - SHAP  : KernelExplainer over the ML ensemble's predict_proba, using a
            small background sample (real SHAP values, computed per call --
            not the same as the SHAP-guided *feature-selection* pass in
            src/ml/shap_selection.py, which ranks features globally before
            training; this one explains one specific prediction).
  - LIME  : LimeTabularExplainer over the same ML ensemble's predict_proba.
  - Attention: the DL branch's own per-time-step attention weights for the
            window ending at this flow (no extra computation needed --
            "no extra cost", as the report puts it).
"""
from __future__ import annotations
import logging
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

logging.getLogger("shap").setLevel(logging.WARNING)


def shap_local_explanation(ml_model, background_X: np.ndarray, instance_X: np.ndarray,
                            feature_names: list, top_k: int, class_idx: int):
    background = shap.sample(background_X, min(50, len(background_X)))
    explainer = shap.KernelExplainer(ml_model.predict_proba, background)
    sv = explainer.shap_values(instance_X.reshape(1, -1), nsamples=200, silent=True)
    # sv: list[per-class] of (1, n_features) OR (1, n_features, n_classes)
    if isinstance(sv, list):
        vals = sv[class_idx][0]
    elif np.ndim(sv) == 3:
        vals = sv[0, :, class_idx]
    else:
        vals = sv[0]
    order = np.argsort(-np.abs(vals))[:top_k]
    return {feature_names[i]: float(vals[i]) for i in order}


def lime_local_explanation(ml_model, train_X: np.ndarray, instance_X: np.ndarray,
                            feature_names: list, class_names: list, top_k: int,
                            num_samples: int, class_idx: int):
    # discretize_continuous=False keeps LIME's explanation keyed by plain
    # feature names (rather than binned conditions like "X <= 1.00"), so its
    # output is directly comparable to SHAP's feature-name keys in the
    # Cross-Explainer Agreement Check -- without this, the two explainers'
    # top-K sets would almost never literally match even when they agree on
    # the same underlying features.
    explainer = LimeTabularExplainer(
        train_X, feature_names=feature_names, class_names=class_names,
        mode="classification", discretize_continuous=False,
    )
    exp = explainer.explain_instance(
        instance_X, ml_model.predict_proba, num_features=top_k,
        num_samples=num_samples, labels=(class_idx,),
    )
    return dict(exp.as_list(label=class_idx))


def attention_explanation(attn_weights_for_window: np.ndarray, top_k: int):
    """attn_weights_for_window: (window_size,) — attention over the flows in
    the sequence ending at the instance being explained."""
    order = np.argsort(-attn_weights_for_window)[:top_k]
    # time-step index 0 = oldest flow in window, window_size-1 = current flow
    return {f"t-{len(attn_weights_for_window) - 1 - i}": float(attn_weights_for_window[i]) for i in order}

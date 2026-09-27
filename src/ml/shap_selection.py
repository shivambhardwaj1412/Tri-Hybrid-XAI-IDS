"""
SHAP-guided feature selection for Tri-Hybrid-XAI-IDS.

Stage 2 / Branch A:

1. Accept training data as pandas DataFrame or NumPy array.
2. Preserve the REAL feature names from the preprocessing pipeline.
3. Train an initial Random Forest.
4. Compute real SHAP values using TreeExplainer.
5. Rank features by mean absolute SHAP importance.
6. Select the configured top-K features.
7. Save:
      results/shap/feature_importance.csv
      results/shap/selected_features.json
      results/shap/shap_global_importance.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

from sklearn.ensemble import RandomForestClassifier


def select_features(
    X_train,
    y_train,
    cfg: dict,
    results_dir: str = "results/shap",
    feature_names=None,
):
    """
    Perform SHAP-based feature selection.

    Parameters
    ----------
    X_train:
        Training features. Can be pandas DataFrame or NumPy array.

    y_train:
        Training labels.

    cfg:
        Project configuration dictionary.

    results_dir:
        Directory where SHAP artifacts are saved.

    feature_names:
        Actual feature names from preprocessing.
        This should normally be data.feature_names.

    Returns
    -------
    selected_features:
        List of selected feature names.

    importance_df:
        Complete SHAP feature ranking DataFrame.
    """

    # ================================================================
    # OUTPUT DIRECTORY
    # ================================================================

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # VALIDATION
    # ================================================================

    if X_train is None:
        raise ValueError("X_train cannot be None.")

    if y_train is None:
        raise ValueError("y_train cannot be None.")

    # ================================================================
    # PRESERVE REAL FEATURE NAMES
    # ================================================================

    if isinstance(X_train, pd.DataFrame):

        X_train = X_train.copy()

        if feature_names is not None:

            feature_names = list(feature_names)

            if len(feature_names) != X_train.shape[1]:
                raise ValueError(
                    "feature_names length does not match "
                    f"X_train columns: "
                    f"{len(feature_names)} != {X_train.shape[1]}"
                )

            X_train.columns = feature_names

        else:
            feature_names = list(X_train.columns)

    else:

        X_train = np.asarray(X_train)

        if X_train.ndim != 2:
            raise ValueError(
                f"X_train must be 2-dimensional. "
                f"Received shape: {X_train.shape}"
            )

        n_features = X_train.shape[1]

        if feature_names is None:

            # Fallback only.
            feature_names = [
                f"feature_{i}"
                for i in range(n_features)
            ]

        else:

            feature_names = list(feature_names)

            if len(feature_names) != n_features:
                raise ValueError(
                    "feature_names length does not match "
                    f"X_train features: "
                    f"{len(feature_names)} != {n_features}"
                )

        X_train = pd.DataFrame(
            X_train,
            columns=feature_names
        )

    # ================================================================
    # LABELS
    # ================================================================

    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    else:
        y_train = np.asarray(y_train)

    y_train = y_train.reshape(-1)

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train contain different numbers of rows: "
            f"{len(X_train)} != {len(y_train)}"
        )

    # ================================================================
    # CONFIGURATION
    # ================================================================

    shap_cfg = (
        cfg
        .get("ml_branch", {})
        .get("shap_feature_selection", {})
    )

    top_k = int(
        shap_cfg.get("top_k_features", 20)
    )

    max_samples = shap_cfg.get(
        "max_samples",
        3000
    )

    seed = (
        cfg
        .get("project", {})
        .get("seed", 42)
    )

    top_k = min(
        top_k,
        X_train.shape[1]
    )

    # ================================================================
    # CAP SHAP TRAINING SAMPLE
    # ================================================================

    if max_samples and len(X_train) > max_samples:

        from src.utils.config import subsample_stratified

        X_train, y_train = subsample_stratified(
            X_train,
            y_train,
            int(max_samples),
            seed
        )

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(
                X_train,
                columns=feature_names
            )

        print(
            f"[SHAP] Training initial Random Forest with "
            f"{len(y_train):,} samples and "
            f"{X_train.shape[1]} features..."
        )

    else:

        print(
            f"[SHAP] Training initial Random Forest with "
            f"{len(y_train):,} samples and "
            f"{X_train.shape[1]} features..."
        )

    # ================================================================
    # RANDOM FOREST
    # ================================================================

    n_classes_local = len(
        np.unique(y_train)
    )

    if n_classes_local > 5:
        n_estimators = 60
        max_depth = 10
    else:
        n_estimators = 150
        max_depth = 15

    initial_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    initial_model.fit(
        X_train,
        y_train
    )

    # ================================================================
    # SHAP
    # ================================================================

    print(
        "[SHAP] Computing real SHAP values..."
    )

    explainer = shap.TreeExplainer(
        initial_model
    )

    shap_values = explainer.shap_values(
        X_train,
        check_additivity=False
    )

    # ================================================================
    # SHAP OUTPUT COMPATIBILITY
    # ================================================================

    if isinstance(shap_values, list):

        # Older SHAP:
        # list[class] -> samples x features

        abs_values = np.mean(
            [
                np.abs(values)
                for values in shap_values
            ],
            axis=0
        )

    else:

        shap_values = np.asarray(
            shap_values
        )

        if shap_values.ndim == 3:

            # Newer SHAP:
            # samples x features x classes

            abs_values = np.mean(
                np.abs(shap_values),
                axis=2
            )

        elif shap_values.ndim == 2:

            # Binary/single-output SHAP

            abs_values = np.abs(
                shap_values
            )

        else:

            raise ValueError(
                "Unexpected SHAP output shape: "
                f"{shap_values.shape}"
            )

    # ================================================================
    # FEATURE IMPORTANCE
    # ================================================================

    mean_importance = abs_values.mean(
        axis=0
    )

    if len(mean_importance) != X_train.shape[1]:
        raise ValueError(
            "SHAP importance length does not match "
            "number of features: "
            f"{len(mean_importance)} != {X_train.shape[1]}"
        )

    importance_df = pd.DataFrame(
        {
            "feature": list(X_train.columns),
            "mean_abs_shap": mean_importance,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # ================================================================
    # TOP-K FEATURES
    # ================================================================

    selected_features = (
        importance_df[
            "feature"
        ]
        .head(top_k)
        .tolist()
    )

    print(
        f"[SHAP] Selected top "
        f"{len(selected_features)} features."
    )

    # ================================================================
    # SAVE FEATURE RANKING
    # ================================================================

    importance_file = (
        results_path /
        "feature_importance.csv"
    )

    importance_df.to_csv(
        importance_file,
        index=False
    )

    print(
        f"[SHAP] Feature ranking saved to: "
        f"{importance_file}"
    )

    # ================================================================
    # SAVE SELECTED FEATURES
    # ================================================================

    selected_file = (
        results_path /
        "selected_features.json"
    )

    with open(
        selected_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            selected_features,
            f,
            indent=2
        )

    print(
        f"[SHAP] Selected feature list saved to: "
        f"{selected_file}"
    )

    # ================================================================
    # SHAP GLOBAL IMPORTANCE PLOT
    # ================================================================

    plot_file = (
        results_path /
        "shap_global_importance.png"
    )

    plot_df = (
        importance_df
        .head(15)
        .copy()
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_df["feature"][::-1],
        plot_df["mean_abs_shap"][::-1]
    )

    plt.xlabel(
        "Mean |SHAP value|"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "SHAP Global Feature Importance "
        "(Initial Random Forest)"
    )

    plt.tight_layout()

    plt.savefig(
        plot_file,
        dpi=140,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[SHAP] Global importance plot saved to: "
        f"{plot_file}"
    )

    # ================================================================
    # RETURN
    # ================================================================

    return (
        selected_features,
        importance_df
    )
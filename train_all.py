"""
Tri-Hybrid-XAI-IDS
Complete End-to-End Training Pipeline

Supported datasets:
    cicids2017
    nsl_kdd
    unsw_nb15
    sample

Usage:

    python train_all.py

or explicitly select a dataset:

    python train_all.py --dataset cicids2017
    python train_all.py --dataset nsl_kdd
    python train_all.py --dataset unsw_nb15
    python train_all.py --dataset sample

The selected dataset overrides config.yaml for this run.

Pipeline:

    Stage 1
        Dataset loading
        Cleaning
        Encoding
        Train/Validation/Test split
        Scaling
        SMOTE-Tomek
        Sequence dataset

    Stage 2a
        SHAP feature selection
        Random Forest / SVM / XGBoost ensemble

    Stage 2b
        CNN-BiLSTM-Attention temporal branch

    Stage 3
        Leakage-free meta-learner fusion

    Stage 4
        Metrics
        Confusion matrix
        Comparison plots
        Experiment summary
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)

from src.utils.config import load_config, subsample_stratified

from src.preprocessing.pipeline import (
    run_pipeline,
    ProcessedData,  # Required for joblib compatibility
)

from src.ml.shap_selection import select_features

from src.ml.ensemble import (
    train_ensemble,
    save_ensemble,
)

from src.fusion.train_meta import run_fusion


# ============================================================
# DATASET SELECTION
# ============================================================

SUPPORTED_DATASETS = [
    "cicids2017",
    "nsl_kdd",
    "unsw_nb15",
    "sample",
]


def parse_arguments():
    """
    Read optional dataset selection from command line.
    """

    parser = argparse.ArgumentParser(
        description="Tri-Hybrid-XAI-IDS complete training pipeline"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=SUPPORTED_DATASETS,
        help=(
            "Dataset to use. "
            "If omitted, dataset from config.yaml is used."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # START TIMER
    # ========================================================

    t_start = time.time()

    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    cfg = load_config()

    args = parse_arguments()

    # --------------------------------------------------------
    # Command-line dataset overrides config.yaml
    # --------------------------------------------------------

    if args.dataset is not None:

        cfg["data"]["dataset"] = args.dataset

    dataset_name = (
        cfg["data"]["dataset"]
        .lower()
        .strip()
    )

    if dataset_name not in SUPPORTED_DATASETS:

        raise ValueError(
            f"Unsupported dataset: {dataset_name}\n"
            f"Supported datasets: {SUPPORTED_DATASETS}"
        )

    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "                 TRI-HYBRID XAI-IDS"
    )

    print(
        "          COMPLETE TRAINING PIPELINE"
    )

    print("=" * 80)

    print(
        f"Selected dataset : {dataset_name}"
    )

    print(
        f"Random seed      : {cfg['project']['seed']}"
    )

    print(
        f"Window size      : "
        f"{cfg['sequence']['window_size']}"
    )

    print(
        f"ML max samples   : "
        f"{cfg['ml_branch'].get('max_training_samples')}"
    )

    print(
        f"DL max rows      : "
        f"{cfg['dl_branch'].get('max_training_rows')}"
    )

    print("=" * 80)

    # ========================================================
    # DATASET-SPECIFIC OUTPUT DIRECTORIES
    # ========================================================

    processed_dir_map = {
        "cicids2017": "data/processed_cicids2017",
        "nsl_kdd": "data/processed_nsl_kdd",
        "unsw_nb15": "data/processed_unsw_nb15",
        "sample": "data/processed_sample",
    }

    cfg["data"]["processed_dir"] = processed_dir_map[dataset_name]

    base_models_dir = Path(cfg["paths"].get("models_dir", "models"))
    base_results_dir = Path(cfg["paths"].get("results_dir", "results"))

    dataset_models_dir = base_models_dir / dataset_name
    dataset_results_dir = base_results_dir / dataset_name

    ml_models_dir = dataset_models_dir / "ml"
    dl_models_dir = dataset_models_dir / "dl"
    meta_models_dir = dataset_models_dir / "meta"

    shap_results_dir = dataset_results_dir / "shap"
    plots_results_dir = dataset_results_dir / "plots"
    confusion_results_dir = dataset_results_dir / "confusion_matrices"

    for directory in [
        ml_models_dir,
        dl_models_dir,
        meta_models_dir,
        shap_results_dir,
        plots_results_dir,
        confusion_results_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    cfg["paths"] = dict(cfg.get("paths", {}))
    cfg["paths"]["models_dir"] = str(dataset_models_dir)
    cfg["paths"]["results_dir"] = str(dataset_results_dir)

    print("\nDataset-specific directories:")
    print(f"  Processed : {cfg['data']['processed_dir']}")
    print(f"  Models    : {dataset_models_dir}")
    print(f"  Results   : {dataset_results_dir}")

    # ========================================================
    # STAGE 1
    # PREPROCESSING
    # ========================================================

    print("\n" + "=" * 80)

    print(
        f"STAGE 1 — PREPROCESSING"
    )

    print(
        f"Dataset = {dataset_name}"
    )

    print("=" * 80)

    data = run_pipeline(cfg)

    # --------------------------------------------------------
    # Verify preprocessing result
    # --------------------------------------------------------

    print("\nPREPROCESSING OUTPUT")

    print(
        f"Train shape      : {data.X_train.shape}"
    )

    print(
        f"Validation shape : {data.X_val.shape}"
    )

    print(
        f"Test shape       : {data.X_test.shape}"
    )

    print(
        f"Number features  : {len(data.feature_names)}"
    )

    print(
        f"Classes          : "
        f"{list(data.label_encoder.classes_)}"
    )

    # ========================================================
    # VERIFY FEATURE NAMES
    # ========================================================

    if not hasattr(
        data,
        "feature_names"
    ):

        raise AttributeError(
            "ProcessedData does not contain feature_names."
        )

    if len(data.feature_names) != data.X_train.shape[1]:

        raise ValueError(
            "Number of feature names does not match "
            "training features:\n"
            f"{len(data.feature_names)} != "
            f"{data.X_train.shape[1]}"
        )

    print(
        f"Feature names verified: "
        f"{len(data.feature_names)}"
    )

    # ========================================================
    # STAGE 2a
    # SHAP + ML ENSEMBLE
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "STAGE 2a — BRANCH A"
    )

    print(
        "SHAP Feature Selection + ML Ensemble"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Cap ML training data
    # --------------------------------------------------------

    max_n = cfg["ml_branch"].get(
        "max_training_samples"
    )

    (
        X_train_ml_full,
        y_train_ml_full
    ) = subsample_stratified(
        data.X_train,
        data.y_train,
        max_n,
        cfg["project"]["seed"],
    )

    if (
        max_n
        and len(data.y_train) > max_n
    ):

        print(
            f"Branch A training set capped: "
            f"{len(data.y_train):,} -> "
            f"{len(y_train_ml_full):,} rows"
        )

        print(
            "Reason: SVM probability training "
            "becomes expensive on large datasets."
        )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    if isinstance(
        X_train_ml_full,
        pd.DataFrame
    ):

        X_train_ml_full = (
            X_train_ml_full.copy()
        )

        X_train_ml_full.columns = (
            list(data.feature_names)
        )

    else:

        X_train_ml_full = pd.DataFrame(
            X_train_ml_full,
            columns=data.feature_names,
        )

    # ========================================================
    # SHAP FEATURE SELECTION
    # ========================================================

    print(
        "\nRunning SHAP feature selection..."
    )

    selected_features, importance_df = (
        select_features(
            X_train_ml_full,
            y_train_ml_full,
            cfg,
            results_dir=str(shap_results_dir),
            feature_names=data.feature_names,
        )
    )

    print(
        f"Selected "
        f"{len(selected_features)} / "
        f"{len(data.feature_names)} "
        f"features via SHAP."
    )

    print(
        "Selected features:"
    )

    for i, feature in enumerate(
        selected_features,
        start=1
    ):

        print(
            f"  {i:02d}. {feature}"
        )

    # ========================================================
    # VALIDATE SELECTED FEATURES
    # ========================================================

    missing_selected = [
        feature
        for feature in selected_features
        if feature not in data.feature_names
    ]

    if missing_selected:

        raise ValueError(
            "SHAP selected features that do not exist "
            "in preprocessing feature list:\n"
            f"{missing_selected}"
        )

    # ========================================================
    # SELECT ML FEATURES
    # ========================================================

    X_train_ml_selected = (
        X_train_ml_full[
            selected_features
        ]
        .values
    )

    # ========================================================
    # TRAIN ML ENSEMBLE
    # ========================================================

    n_classes = len(
        data.label_encoder.classes_
    )

    print("\nTraining ML ensemble...")

    print(
        f"Number of classes: {n_classes}"
    )

    print(
        f"Number of ML features: "
        f"{X_train_ml_selected.shape[1]}"
    )

    ml_model = train_ensemble(
        X_train_ml_selected,
        y_train_ml_full,
        cfg,
        n_classes,
    )

    # ========================================================
    # PREPARE TEST DATA
    # ========================================================

    if isinstance(
        data.X_test,
        pd.DataFrame
    ):

        X_test_full = (
            data.X_test.copy()
        )

        X_test_full.columns = (
            list(data.feature_names)
        )

    else:

        X_test_full = pd.DataFrame(
            data.X_test,
            columns=data.feature_names,
        )

    X_test_ml_selected = (
        X_test_full[
            selected_features
        ]
        .values
    )

    # ========================================================
    # ML TEST PREDICTIONS
    # ========================================================

    ml_preds = ml_model.predict(
        X_test_ml_selected
    )

    ml_only_acc = accuracy_score(
        data.y_test,
        ml_preds
    )

    ml_only_f1 = f1_score(
        data.y_test,
        ml_preds,
        average="macro",
        zero_division=0,
    )

    print("\nBranch A results")

    print(
        f"Accuracy : {ml_only_acc:.4f}"
    )

    print(
        f"Macro F1 : {ml_only_f1:.4f}"
    )

    # ========================================================
    # SAVE ML MODEL
    # ========================================================

    ml_model_path = ml_models_dir / "ml_ensemble.joblib"
    selected_features_path = ml_models_dir / "selected_features.json"
    importance_path = ml_models_dir / "feature_importance.csv"

    save_ensemble(
        ml_model,
        str(ml_model_path)
    )

    with open(
        selected_features_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            selected_features,
            f,
            indent=2,
        )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    print("\nML artifacts saved:")
    print(f"  {ml_model_path}")
    print(f"  {selected_features_path}")
    print(f"  {importance_path}")

    # ========================================================
    # STAGE 2b + STAGE 3
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "STAGE 2b / STAGE 3"
    )

    print(
        "Branch B — CNN-BiLSTM-Attention"
    )

    print(
        "Meta-Learner — Tri-Hybrid Fusion"
    )

    print("=" * 80)

    # ========================================================
    # LOAD SEQUENCE DATA
    # ========================================================

    sequence_path = (
        Path(
            cfg["data"]["processed_dir"]
        )
        / "sequence_ordered.parquet"
    )

    if not sequence_path.exists():

        raise FileNotFoundError(
            "Chronological sequence file not found:\n"
            f"{sequence_path}\n\n"
            "Run Stage 1 preprocessing first."
        )

    seq_df = pd.read_parquet(
        sequence_path
    )

    print(
        f"Loaded chronological sequence data: "
        f"{seq_df.shape}"
    )

    # --------------------------------------------------------
    # Verify Label
    # --------------------------------------------------------

    if "Label" not in seq_df.columns:

        raise ValueError(
            "sequence_ordered.parquet does not contain "
            "the required 'Label' column."
        )

    print(
        "Sequence dataset contains Label: True"
    )

    # ========================================================
    # RUN FUSION
    # ========================================================

    fusion_result = run_fusion(
        cfg,
        seq_df,
        data.feature_names,
        selected_features,
        data.scaler,
        n_classes,
    )

    # ========================================================
    # PRINT FUSION METRICS
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "FINAL EXPERIMENT METRICS"
    )

    print("=" * 80)

    print(
        json.dumps(
            fusion_result["metrics"],
            indent=2
        )
    )

    # ========================================================
    # METRICS CSV
    # ========================================================

    metrics_rows = [

        {
            "model":
                "Branch A - ML Ensemble "
                "(chronological split)",

            "accuracy":
                fusion_result["metrics"][
                    "branch_a_ml_only_acc"
                ],

            "macro_f1":
                fusion_result["metrics"].get(
                    "branch_a_ml_only_f1",
                    np.nan
                ),
        },

        {
            "model":
                "Branch B - CNN-BiLSTM-Attention "
                "(chronological split)",

            "accuracy":
                fusion_result["metrics"][
                    "branch_b_dl_only_acc"
                ],

            "macro_f1":
                fusion_result["metrics"].get(
                    "branch_b_dl_only_f1",
                    np.nan
                ),
        },

        {
            "model":
                "Proposed Tri-Hybrid "
                "(Meta-Learner Fusion)",

            "accuracy":
                fusion_result["metrics"][
                    "fused_meta_acc"
                ],

            "macro_f1":
                fusion_result["metrics"][
                    "fused_meta_f1_macro"
                ],
        },

        {
            "model":
                "Branch A - ML Ensemble "
                "(i.i.d. shuffled split)",

            "accuracy":
                ml_only_acc,

            "macro_f1":
                ml_only_f1,
        },
    ]

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_path = dataset_results_dir / "metrics.csv"

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print(
        f"\nSaved metrics to: "
        f"{metrics_path}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print(
        "\nGenerating confusion matrix..."
    )

    cm = confusion_matrix(
        fusion_result["y_test_aligned"],
        fusion_result["final_preds"],
        labels=np.arange(
            n_classes
        ),
    )

    disp = ConfusionMatrixDisplay(
        cm,
        display_labels=data.label_encoder.classes_,
    )

    fig, ax = plt.subplots(
        figsize=(
            max(6, n_classes * 0.4),
            max(6, n_classes * 0.4),
        )
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        xticks_rotation=90,
    )

    plt.title(
        "Confusion Matrix — "
        "Proposed Tri-Hybrid "
        f"({dataset_name})"
    )

    plt.tight_layout()

    confusion_path = (
        confusion_results_dir / "fused_test_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=140,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved confusion matrix to: "
        f"{confusion_path}"
    )

    # ========================================================
    # ACCURACY COMPARISON PLOT
    # ========================================================

    print(
        "Generating accuracy comparison plot..."
    )

    labels = [
        "Branch A\nML only",
        "Branch B\nDL only",
        "Proposed\nFused",
    ]

    accs = [

        fusion_result["metrics"][
            "branch_a_ml_only_acc"
        ],

        fusion_result["metrics"][
            "branch_b_dl_only_acc"
        ],

        fusion_result["metrics"][
            "fused_meta_acc"
        ],
    ]

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    bars = ax.bar(
        labels,
        accs,
    )

    for bar, accuracy in zip(
        bars,
        accs
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            accuracy + 0.005,

            f"{accuracy:.3f}",

            ha="center",
        )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_ylabel(
        "Test Accuracy"
    )

    ax.set_title(
        "Branch A vs Branch B vs Fused\n"
        f"Dataset: {dataset_name}"
    )

    plt.tight_layout()

    comparison_path = (
        plots_results_dir / "branch_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=140,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved comparison plot to: "
        f"{comparison_path}"
    )

    # ========================================================
    # EXPERIMENT SUMMARY
    # ========================================================

    runtime = time.time() - t_start

    experiment_summary = {

        "dataset_used":
            dataset_name,

        "dataset_source":
            {
                "cicids2017":
                    "data/raw/cicids2017",

                "nsl_kdd":
                    "data/raw/nsl_kdd",

                "unsw_nb15":
                    "data/raw/unsw_nb15",

                "sample":
                    "data/sample",
            }.get(
                dataset_name,
                "unknown"
            ),

        "selected_ml_features":
            selected_features,

        "n_selected_features":
            len(selected_features),

        "total_features":
            len(data.feature_names),

        "n_classes":
            n_classes,

        "classes":
            list(
                data.label_encoder.classes_
            ),

        "train_shape":
            list(
                data.X_train.shape
            ),

        "validation_shape":
            list(
                data.X_val.shape
            ),

        "test_shape":
            list(
                data.X_test.shape
            ),

        **fusion_result["metrics"],

        "branch_a_iid_split_accuracy":
            ml_only_acc,

        "branch_a_iid_split_macro_f1":
            ml_only_f1,

        "runtime_seconds":
            round(
                runtime,
                1
            ),
    }

    summary_path = dataset_results_dir / "experiment_summary.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            experiment_summary,
            f,
            indent=2,
            default=str,
        )

    print(
        f"Saved experiment summary to: "
        f"{summary_path}"
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "                    TRAINING COMPLETE"
    )

    print("=" * 80)

    print(
        f"Dataset used : {dataset_name}"
    )

    print(
        f"Runtime      : {runtime:.1f} seconds"
    )

    print("\nResults:")
    print(f"  {metrics_path}")
    print(f"  {summary_path}")
    print(f"  {comparison_path}")
    print(f"  {confusion_results_dir}")

    print("\nModels:")
    print(f"  {ml_models_dir}")
    print(f"  {dl_models_dir}")
    print(f"  {meta_models_dir}")

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
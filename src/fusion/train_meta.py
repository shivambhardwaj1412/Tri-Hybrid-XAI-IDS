"""
Stage 3 — Meta-Learner Fusion.

Both branches must be evaluated on data they were NOT trained on before their
outputs are used to train the meta-learner, or the meta-learner leaks
training-set information.

The fusion stage uses one chronological split of the flow table:

  1. Train Branch A ML ensemble on chronological TRAIN rows.
  2. Train Branch B DL model on TRAIN windows only.
  3. Generate ML + DL probabilities on the VAL split.
  4. Train the meta-learner on the aligned VAL predictions.
  5. Evaluate the complete fused system on the untouched TEST split.

Dataset-specific artifacts are saved under:

    models/<dataset>/ml/
    models/<dataset>/dl/
    models/<dataset>/meta/

The meta learner may encounter fewer classes in the validation sequence set
than exist in the global dataset label space. In that situation, the learned
classifier naturally contains only the classes observed in validation.

To preserve a consistent global class space, this file wraps the learned
classifier with GlobalClassMetaLearner. The wrapper:

    - preserves the original trained classifier,
    - exposes all global classes,
    - maps the classifier's local probabilities into the global class space,
    - assigns 0.0 probability to classes absent from validation,
    - never uses TEST labels for fitting,
    - always returns probabilities with shape (N, n_classes).

This prevents the inference pipeline from receiving an 18-class probability
vector when the dataset has 23 global classes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.ml.ensemble import train_ensemble, predict_proba as ml_predict_proba
from src.dl.model import build_model
from src.dl.sequence import make_sequences


logger = logging.getLogger("fusion")


# ======================================================================
# Global-class meta learner wrapper
# ======================================================================

class GlobalClassMetaLearner:
    """
    Wrapper around the actual sklearn meta classifier.

    The underlying classifier may have been trained on only the classes that
    were present in the validation sequence set.

    This wrapper expands its probabilities into the complete global class
    space.

    Example:

        Global classes:
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ..., 22]

        Local classifier:
            [0, 1, 2, 3, 4, 5, 6, 7, 9, ..., 22]

        Missing class:
            8

        Output probability:
            class 8 -> 0.0

    The wrapper does NOT fabricate a learned probability for a class that
    never appeared in the meta-learner training data.
    """

    def __init__(
        self,
        classifier,
        global_classes,
    ):
        self.classifier = classifier
        self.global_classes_ = np.asarray(
            global_classes,
            dtype=int,
        )

        self.classes_ = self.global_classes_.copy()

        self.n_features_in_ = getattr(
            classifier,
            "n_features_in_",
            None,
        )

        self.local_classes_ = np.asarray(
            getattr(
                classifier,
                "classes_",
                [],
            ),
            dtype=int,
        )

        self.local_to_global_index_ = {
            int(class_id): idx
            for idx, class_id in enumerate(
                self.global_classes_
            )
        }

    def fit(self, X, y):
        """
        The wrapped classifier is already fitted by run_fusion().

        This method exists for sklearn-style compatibility but does not
        refit the underlying classifier.
        """
        return self

    def predict_proba(self, X):
        """
        Return probabilities in the complete global class space.
        """

        local_proba = self.classifier.predict_proba(X)

        local_proba = np.asarray(
            local_proba,
            dtype=float,
        )

        if local_proba.ndim != 2:
            raise ValueError(
                "Meta learner returned invalid probability shape: "
                f"{local_proba.shape}"
            )

        n_samples = local_proba.shape[0]
        n_global_classes = len(
            self.global_classes_
        )

        global_proba = np.zeros(
            (
                n_samples,
                n_global_classes,
            ),
            dtype=float,
        )

        for local_idx, class_id in enumerate(
            self.local_classes_
        ):
            class_id = int(class_id)

            if class_id not in self.local_to_global_index_:
                raise ValueError(
                    "Meta learner contains class "
                    f"{class_id}, but it is not present in "
                    "the configured global class space."
                )

            global_idx = self.local_to_global_index_[
                class_id
            ]

            global_proba[:, global_idx] = (
                local_proba[:, local_idx]
            )

        return global_proba

    def predict(self, X):
        """
        Predict global class indices.
        """

        local_preds = self.classifier.predict(X)

        local_preds = np.asarray(
            local_preds,
            dtype=int,
        )

        global_preds = np.empty_like(
            local_preds
        )

        for idx, class_id in enumerate(
            local_preds
        ):
            class_id = int(class_id)

            if class_id not in self.local_to_global_index_:
                raise ValueError(
                    "Meta learner predicted class "
                    f"{class_id}, but it is not present "
                    "in the global class space."
                )

            global_preds[idx] = class_id

        return global_preds

    def __getattr__(self, name):
        """
        Delegate unknown attributes to the underlying sklearn classifier.

        This preserves compatibility with code that expects attributes such
        as coef_, intercept_, classes_, etc.
        """

        classifier = self.__dict__.get(
            "classifier"
        )

        if classifier is None:
            raise AttributeError(name)

        return getattr(
            classifier,
            name,
        )


# ======================================================================
# Chronological split
# ======================================================================

def chronological_split(
    seq_df: pd.DataFrame,
    test_size: float,
    val_size: float,
):
    """
    Split the dataset into train/validation/test while preserving
    within-class temporal order.

    A naive single global time cutoff can fail badly on datasets created by
    concatenating multiple capture files. A class can otherwise end up almost
    completely inside only one split.

    This function therefore performs proportional chronological splitting
    inside each class and then re-sorts each resulting split by Timestamp.
    """

    df = seq_df.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )

    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in df.groupby(
        "Label",
        sort=False,
    ):
        group = group.sort_values(
            "Timestamp"
        )

        n = len(group)

        n_test = (
            max(
                1,
                int(n * test_size),
            )
            if n >= 3
            else 0
        )

        n_val = (
            max(
                1,
                int(n * val_size),
            )
            if n - n_test >= 2
            else 0
        )

        n_train = (
            n
            - n_test
            - n_val
        )

        train_parts.append(
            group.iloc[
                :n_train
            ]
        )

        val_parts.append(
            group.iloc[
                n_train:n_train + n_val
            ]
        )

        test_parts.append(
            group.iloc[
                n_train + n_val:
            ]
        )

    def _concat_sorted(parts):
        return (
            pd.concat(parts)
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )

    return (
        _concat_sorted(train_parts),
        _concat_sorted(val_parts),
        _concat_sorted(test_parts),
    )



# ======================================================================
# UNSW-NB15 sequence construction
# ======================================================================

def make_unsw_sequences_weighted(X, y, window_size, stride):
    """
    Build chronological UNSW-NB15 training windows.

    The target is the label of the final flow in the window. This is a
    proper sequence-to-current-flow formulation and prevents rare attack
    classes from disappearing simply because a surrounding window contains
    more majority-class flows.

    A purity-derived sample weight is still used: the more of the window
    that agrees with the endpoint label, the more trustworthy the training
    target is.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    n = len(X)
    if n < window_size:
        return (
            np.empty((0, window_size, X.shape[1]), dtype=X.dtype),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=np.float32),
        )

    windows = []
    labels = []
    weights = []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        yw = y[start:end]
        endpoint_label = int(y[end - 1])

        # Weight the window by how strongly its history supports the
        # endpoint target. A minimum weight keeps mixed windows useful.
        purity = float(np.mean(yw == endpoint_label))
        weight = max(0.25, purity * purity)

        windows.append(X[start:end])
        labels.append(endpoint_label)
        weights.append(weight)

    return (
        np.asarray(windows),
        np.asarray(labels, dtype=int),
        np.asarray(weights, dtype=np.float32),
    )

def make_unsw_eval_sequences(X, y, window_size, stride):
    """
    Build every chronological evaluation window.

    Evaluation targets are the labels of the final flow in each window,
    matching the training target definition above and the ML/DL endpoint
    alignment used by the fusion stage.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    n = len(X)
    if n < window_size:
        return (
            np.empty((0, window_size, X.shape[1]), dtype=X.dtype),
            np.empty((0,), dtype=int),
        )

    windows = []
    labels = []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows.append(X[start:end])
        labels.append(int(y[end - 1]))

    return (
        np.asarray(windows),
        np.asarray(labels, dtype=int),
    )

def unsw_sequence_end_indices(y, window_size, stride):
    """Return endpoints for every UNSW window."""
    n = len(y)
    if n < window_size:
        return np.asarray([], dtype=int)
    return np.asarray([window_size - 1 + i * stride
                       for i in range(n - window_size + 1)], dtype=int)


# ======================================================================
# Main fusion stage
# ======================================================================

def run_fusion(
    cfg: dict,
    seq_df: pd.DataFrame,
    feature_names: list,
    selected_ml_features: list,
    scaler,
    n_classes: int,
    results_dir="results",
):
    """
    Train the chronological ML + DL branches and the meta learner.

    IMPORTANT:
    cfg["paths"]["models_dir"] is expected to already point to the selected
    dataset model directory, for example:

        models/nsl_kdd

    Therefore this function saves artifacts under:

        models/nsl_kdd/ml/
        models/nsl_kdd/dl/
        models/nsl_kdd/meta/
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    ts = cfg["data"]["test_size"]
    vs = cfg["data"]["val_size"]

    ws = cfg["sequence"]["window_size"]
    stride = cfg["sequence"]["stride"]

    dataset_name = str(
        cfg.get("data", {}).get("dataset", "sample")
    ).strip().lower()

    is_unsw = dataset_name == "unsw_nb15"

    # ------------------------------------------------------------------
    # Dataset-specific model directories
    # ------------------------------------------------------------------

    models_root = Path(
        cfg.get(
            "paths",
            {},
        ).get(
            "models_dir",
            "models",
        )
    )

    ml_models_dir = (
        models_root / "ml"
    )

    dl_models_dir = (
        models_root / "dl"
    )

    meta_models_dir = (
        models_root / "meta"
    )

    ml_models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dl_models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    meta_models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Fusion model root: %s",
        models_root,
    )

    # ------------------------------------------------------------------
    # Global class validation
    # ------------------------------------------------------------------

    global_classes = np.arange(
        n_classes,
        dtype=int,
    )

    observed_global_classes = np.sort(
        seq_df["Label"]
        .dropna()
        .astype(int)
        .unique()
    )

    logger.info(
        "Global configured class count: %d",
        n_classes,
    )

    logger.info(
        "Observed dataset classes: %s",
        observed_global_classes,
    )

    if len(observed_global_classes) != n_classes:
        logger.warning(
            "Observed class count (%d) differs from configured "
            "n_classes (%d). The meta learner will use the configured "
            "global class space.",
            len(observed_global_classes),
            n_classes,
        )

    # ------------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------------

    train_df, val_df, test_df = chronological_split(
        seq_df,
        ts,
        vs,
    )

    logger.info(
        "Chronological split: train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # ------------------------------------------------------------------
    # Limit DL training rows
    # ------------------------------------------------------------------

    max_rows = cfg[
        "dl_branch"
    ].get(
        "max_training_rows"
    )

    if (
        max_rows
        and len(train_df) > max_rows
    ):
        from src.utils.config import (
            subsample_stratified,
        )

        train_df, _ = (
            subsample_stratified(
                train_df,
                train_df["Label"].values,
                max_rows,
                cfg["project"]["seed"],
            )
        )

    # ------------------------------------------------------------------
    # Limit validation/test evaluation size
    # ------------------------------------------------------------------

    # UNSW-NB15 must use the complete chronological validation/test
    # populations for the final reported metrics. The previous 10,000-row
    # cap could hide rare classes and made the reported result non-comparable
    # with the actual held-out split. Other datasets retain the configured
    # evaluation cap.
    if is_unsw:
        eval_cap = None
    else:
        eval_cap = cfg[
            "ml_branch"
        ].get(
            "max_eval_samples",
            10000,
        )

    if (
        eval_cap
        and len(val_df) > eval_cap
    ):
        from src.utils.config import (
            subsample_stratified,
        )

        val_df, _ = (
            subsample_stratified(
                val_df,
                val_df["Label"].values,
                eval_cap,
                cfg["project"]["seed"],
            )
        )

    if (
        eval_cap
        and len(test_df) > eval_cap
    ):
        from src.utils.config import (
            subsample_stratified,
        )

        test_df, _ = (
            subsample_stratified(
                test_df,
                test_df["Label"].values,
                eval_cap,
                cfg["project"]["seed"],
            )
        )

    # ==================================================================
    # Branch A — ML
    # ==================================================================

    logger.info(
        "Training chronological ML branch..."
    )

    from src.utils.config import (
        subsample_stratified,
    )

    max_n = cfg[
        "ml_branch"
    ].get(
        "max_training_samples"
    )

    X_train_ml, y_train_ml = (
        subsample_stratified(
            train_df[
                selected_ml_features
            ],
            train_df[
                "Label"
            ].values,
            max_n,
            cfg["project"]["seed"],
        )
    )

    X_train_ml = (
        X_train_ml.values
        if hasattr(
            X_train_ml,
            "values",
        )
        else X_train_ml
    )

    ml_model = train_ensemble(
        X_train_ml,
        y_train_ml,
        cfg,
        n_classes,
    )

    logger.info(
        "Chronological ML branch trained."
    )

    # ==================================================================
    # Branch B — DL
    # ==================================================================

    logger.info(
        "Training chronological DL branch..."
    )

    X_train_full = (
        train_df[
            feature_names
        ].values
    )

    y_train_full = (
        train_df[
            "Label"
        ].values
    )

    unique_tr, counts_tr = (
        np.unique(
            y_train_full,
            return_counts=True,
        )
    )

    too_small = unique_tr[
        counts_tr < ws
    ]

    if len(too_small) > 0:
        logger.warning(
            "DL branch: dropping %d class(es) "
            "with < %d rows in train split "
            "(cannot form a sequence window): %s",
            len(too_small),
            ws,
            too_small,
        )

        keep_mask = ~np.isin(
            y_train_full,
            too_small,
        )

        X_train_full = (
            X_train_full[
                keep_mask
            ]
        )

        y_train_full = (
            y_train_full[
                keep_mask
            ]
        )

    sample_weights = None

    if is_unsw:
        logger.info(
            "UNSW-NB15: using all temporal windows with purity-weighted training targets."
        )
        Xtr_seq, ytr_seq, sample_weights = make_unsw_sequences_weighted(
            X_train_full, y_train_full, ws, stride
        )
    else:
        Xtr_seq, ytr_seq = make_sequences(
            X_train_full, y_train_full, ws, stride
        )

    logger.info(
        "DL training sequences: X=%s, y=%s",
        Xtr_seq.shape,
        ytr_seq.shape,
    )

    dl_model = build_model(
        ws,
        Xtr_seq.shape[2],
        n_classes,
        cfg,
    )

    fit_kwargs = {
        "epochs": cfg["dl_branch"]["epochs"],
        "batch_size": cfg["dl_branch"]["batch_size"],
        "verbose": 0,
    }

    if is_unsw:
        present_classes = np.unique(ytr_seq).astype(int)
        raw_weights = compute_class_weight(
            class_weight="balanced",
            classes=present_classes,
            y=ytr_seq,
        )
        class_weights = {
            int(class_id): float(min(weight, 8.0))
            for class_id, weight in zip(
                present_classes, raw_weights
            )
        }

        logger.info(
            "UNSW-NB15 DL class weights: %s",
            class_weights,
        )

        # Keras does not allow class_weight and sample_weight together.
        # Combine both into one per-window weight vector.
        combined_weights = sample_weights * np.asarray(
            [class_weights.get(int(label), 1.0) for label in ytr_seq],
            dtype=np.float32,
        )

        # Normalize so weighting changes relative importance without
        # unintentionally changing the overall loss scale.
        mean_weight = float(np.mean(combined_weights))
        if mean_weight > 0.0:
            combined_weights = combined_weights / mean_weight

        combined_weights = np.clip(
            combined_weights,
            0.10,
            8.0,
        ).astype(np.float32)

        fit_kwargs["sample_weight"] = combined_weights

        logger.info(
            "UNSW-NB15 purity sample weights: min=%.3f max=%.3f mean=%.3f",
            float(np.min(sample_weights)),
            float(np.max(sample_weights)),
            float(np.mean(sample_weights)),
        )
        logger.info(
            "UNSW-NB15 combined training weights: min=%.3f max=%.3f mean=%.3f",
            float(np.min(combined_weights)),
            float(np.max(combined_weights)),
            float(np.mean(combined_weights)),
        )
        fit_kwargs["callbacks"] = [
            ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=2,
                min_lr=1e-5,
                verbose=0,
            ),
            EarlyStopping(
                monitor="loss",
                patience=3,
                restore_best_weights=True,
                verbose=0,
            ),
        ]

    # Validation is supplied below through the explicit validation split in
    # the branch-prediction stage; keep the model fit itself deterministic.
    dl_model.fit(
        Xtr_seq,
        ytr_seq,
        **fit_kwargs,
    )

    logger.info(
        "Chronological DL branch trained."
    )

    # ==================================================================
    # Branch prediction helper
    # ==================================================================

    def branch_predictions(
        split_df,
    ):
        """
        Generate aligned ML and DL predictions for a split.

        ML predicts every flow.

        DL predicts sequence windows.

        The ML probabilities are aligned to the final flow of each
        corresponding DL window.
        """

        # --------------------------------------------------------------
        # ML predictions
        # --------------------------------------------------------------

        X_ml = (
            split_df[
                selected_ml_features
            ].values
        )

        ml_proba_all = (
            ml_predict_proba(
                ml_model,
                X_ml,
            )
        )

        ml_proba_all = np.asarray(
            ml_proba_all,
            dtype=float,
        )

        if (
            ml_proba_all.ndim != 2
            or ml_proba_all.shape[1] != n_classes
        ):
            raise ValueError(
                "ML branch probability dimension mismatch. "
                f"Expected {n_classes}, "
                f"received shape {ml_proba_all.shape}."
            )

        # --------------------------------------------------------------
        # DL sequence predictions
        # --------------------------------------------------------------

        X_dl = (
            split_df[
                feature_names
            ].values
        )

        y_dl = (
            split_df[
                "Label"
            ].values
        )

        if is_unsw:
            # Evaluate every window so all classes remain represented and
            # metrics cover the complete held-out split.
            X_seq, y_seq = make_unsw_eval_sequences(
                X_dl, y_dl, ws, stride
            )
        else:
            X_seq, y_seq = make_sequences(
                X_dl,
                y_dl,
                ws,
                stride,
            )

        if len(X_seq) == 0:
            raise ValueError(
                "No DL sequences could be generated "
                "for the supplied split."
            )

        dl_proba_all, attn_all = (
            dl_model.predict(
                X_seq,
                verbose=0,
            )
        )

        dl_proba_all = np.asarray(
            dl_proba_all,
            dtype=float,
        )

        if (
            dl_proba_all.ndim != 2
            or dl_proba_all.shape[1] != n_classes
        ):
            raise ValueError(
                "DL branch probability dimension mismatch. "
                f"Expected {n_classes}, "
                f"received shape {dl_proba_all.shape}."
            )

        # --------------------------------------------------------------
        # Align ML probabilities to DL window endpoints
        # --------------------------------------------------------------

        if is_unsw:
            last_idx = unsw_sequence_end_indices(
                y_dl, ws, stride
            )
        else:
            last_idx = np.array(
                [
                    ws - 1 + i * stride
                    for i in range(
                        len(X_seq)
                    )
                ],
                dtype=int,
            )

        if np.any(
            last_idx >= len(
                ml_proba_all
            )
        ):
            raise ValueError(
                "DL sequence endpoint exceeds "
                "the available ML prediction rows."
            )

        ml_proba_aligned = (
            ml_proba_all[
                last_idx
            ]
        )

        if is_unsw:
            y_aligned = np.asarray(y_seq, dtype=int)
        else:
            y_aligned = y_dl[last_idx]

        return (
            ml_proba_aligned,
            dl_proba_all,
            attn_all,
            y_aligned,
            last_idx,
        )

    # ==================================================================
    # Meta learner — validation predictions
    # ==================================================================

    logger.info(
        "Generating validation predictions for meta learner..."
    )

    (
        ml_val,
        dl_val,
        attn_val,
        y_val_aligned,
        _,
    ) = branch_predictions(
        val_df
    )

    meta_X_val = np.hstack(
        [
            ml_val,
            dl_val,
        ]
    )

    logger.info(
        "Meta validation features: %s",
        meta_X_val.shape,
    )

    if (
        meta_X_val.shape[1]
        != n_classes * 2
    ):
        raise ValueError(
            "Meta learner input dimension mismatch before training. "
            f"Expected {n_classes * 2}, "
            f"received {meta_X_val.shape[1]}."
        )

    # ------------------------------------------------------------------
    # Inspect classes actually present in validation
    # ------------------------------------------------------------------

    validation_classes = np.sort(
        np.unique(
            y_val_aligned
        ).astype(int)
    )

    missing_validation_classes = (
        np.setdiff1d(
            global_classes,
            validation_classes,
        )
    )

    logger.info(
        "Meta validation classes: %s",
        validation_classes,
    )

    if len(
        missing_validation_classes
    ) > 0:
        logger.warning(
            "Meta validation split does not contain %d global "
            "class(es): %s. The meta classifier cannot learn "
            "direct decision boundaries for these classes from "
            "validation data alone. Their global probabilities "
            "will therefore be represented as 0.0.",
            len(
                missing_validation_classes
            ),
            missing_validation_classes,
        )

    # ==================================================================
    # Meta learner
    # ==================================================================

    meta_type = cfg[
        "meta_learner"
    ]["type"]

    if meta_type == "mlp":
        meta_classifier = MLPClassifier(
            hidden_layer_sizes=(16,),
            max_iter=500,
            random_state=cfg[
                "project"
            ]["seed"],
        )

    else:
        meta_classifier = LogisticRegression(
            max_iter=1000,
            random_state=cfg[
                "project"
            ]["seed"],
        )

    meta_classifier.fit(
        meta_X_val,
        y_val_aligned,
    )

    logger.info(
        "Meta classifier trained."
    )

    logger.info(
        "Meta classifier local classes: %s",
        getattr(
            meta_classifier,
            "classes_",
            None,
        ),
    )

    logger.info(
        "Meta classifier local class count: %d",
        len(
            getattr(
                meta_classifier,
                "classes_",
                [],
            )
        ),
    )

    # ==================================================================
    # Wrap classifier into complete global class space
    # ==================================================================

    meta = GlobalClassMetaLearner(
        classifier=meta_classifier,
        global_classes=global_classes,
    )

    logger.info(
        "Global meta learner class space: %s",
        meta.classes_,
    )

    # ==================================================================
    # Final untouched TEST evaluation
    # ==================================================================

    logger.info(
        "Generating final test predictions..."
    )

    (
        ml_test,
        dl_test,
        attn_test,
        y_test_aligned,
        last_idx_test,
    ) = branch_predictions(
        test_df
    )

    meta_X_test = np.hstack(
        [
            ml_test,
            dl_test,
        ]
    )

    if (
        meta_X_test.shape[1]
        != n_classes * 2
    ):
        raise ValueError(
            "Meta learner input dimension mismatch during test "
            "evaluation. "
            f"Expected {n_classes * 2}, "
            f"received {meta_X_test.shape[1]}."
        )

    final_preds = meta.predict(
        meta_X_test
    )

    final_proba = meta.predict_proba(
        meta_X_test
    )

    if (
        final_proba.shape[1]
        != n_classes
    ):
        raise ValueError(
            "Global meta probability dimension mismatch. "
            f"Expected {n_classes}, "
            f"received {final_proba.shape[1]}."
        )

    logger.info(
        "Final meta probability shape: %s",
        final_proba.shape,
    )

    # ==================================================================
    # Metrics
    # ==================================================================

    metrics = {
        "branch_a_ml_only_acc": float(
            accuracy_score(
                y_test_aligned,
                ml_test.argmax(
                    axis=1
                ),
            )
        ),
        "branch_b_dl_only_acc": float(
            accuracy_score(
                y_test_aligned,
                dl_test.argmax(
                    axis=1
                ),
            )
        ),
        "fused_meta_acc": float(
            accuracy_score(
                y_test_aligned,
                final_preds,
            )
        ),
        "fused_meta_f1_macro": float(
            f1_score(
                y_test_aligned,
                final_preds,
                average="macro",
            )
        ),
    }

    logger.info(
        "Fusion metrics: %s",
        metrics,
    )

    # ==================================================================
    # Save fusion results
    # ==================================================================

    results_path = Path(
        results_dir
    )

    results_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        results_path
        / "fusion_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    with open(
        results_path
        / "fusion_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            classification_report(
                y_test_aligned,
                final_preds,
            )
        )

    # ==================================================================
    # Save dataset-specific model artifacts
    # ==================================================================

    ml_model_path = (
        ml_models_dir
        / "ml_ensemble_chrono.joblib"
    )

    dl_model_path = (
        dl_models_dir
        / "dl_model_chrono.keras"
    )

    meta_model_path = (
        meta_models_dir
        / "meta_learner.joblib"
    )

    logger.info(
        "Saving chronological ML model: %s",
        ml_model_path,
    )

    joblib.dump(
        ml_model,
        ml_model_path,
    )

    logger.info(
        "Saving chronological DL model: %s",
        dl_model_path,
    )

    dl_model.save(
        dl_model_path,
    )

    logger.info(
        "Saving global-class meta learner: %s",
        meta_model_path,
    )

    joblib.dump(
        meta,
        meta_model_path,
    )

    logger.info(
        "Dataset-specific fusion artifacts saved successfully."
    )

    # ==================================================================
    # Return complete fusion result
    # ==================================================================

    return {
        "metrics": metrics,
        "ml_model": ml_model,
        "dl_model": dl_model,
        "meta_model": meta,
        "test_df": test_df,
        "last_idx_test": last_idx_test,
        "y_test_aligned": y_test_aligned,
        "ml_test_proba": ml_test,
        "dl_test_proba": dl_test,
        "attn_test": attn_test,
        "final_preds": final_preds,
        "final_proba": final_proba,
    }
"""
Tri-Hybrid-XAI-IDS
Shared Inference Pipeline

Dataset-aware inference pipeline.

Supported datasets:
    - cicids2017
    - nsl_kdd
    - unsw_nb15

Artifact structure:

    models/
        cicids2017/
            ml/
            dl/
            meta/
        nsl_kdd/
            ml/
            dl/
            meta/
        unsw_nb15/
            ml/
            dl/
            meta/

    data/
        processed_cicids2017/
        processed_nsl_kdd/
        processed_unsw_nb15/

    results/
        cicids2017/
        nsl_kdd/
        unsw_nb15/

Inference path:

    Input window
        ↓
    Feature alignment
        ↓
    Training scaler
        ↓
    ML branch
        ↓
    DL sequence branch
        ↓
    Meta learner
        ↓
    Final prediction
        ↓
    SHAP + LIME + Attention
        ↓
    Cross-explainer agreement
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src.xai.explain import (
    shap_local_explanation,
    lime_local_explanation,
    attention_explanation,
)

from src.agreement.agreement_check import (
    cross_explainer_agreement,
)


class TriHybridPipeline:
    """
    Dataset-aware shared inference pipeline.

    The pipeline loads artifacts belonging to ONE dataset only.
    """

    # ==================================================================
    # DATASET ALIASES
    # ==================================================================

    DATASET_ALIASES = {
        "cicids": "cicids2017",
        "cicids_2017": "cicids2017",
        "cicids-2017": "cicids2017",
        "cicids2017": "cicids2017",

        "nsl": "nsl_kdd",
        "nsl-kdd": "nsl_kdd",
        "nsl_kdd": "nsl_kdd",
        "nslkdd": "nsl_kdd",

        "unsw": "unsw_nb15",
        "unsw-nb15": "unsw_nb15",
        "unsw_nb15": "unsw_nb15",
        "unswnb15": "unsw_nb15",

        "sample": "sample",
    }

    SUPPORTED_DATASETS = {
        "cicids2017",
        "nsl_kdd",
        "unsw_nb15",
        "sample",
    }

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        cfg: dict,
        models_dir=None,
        processed_dir=None,
    ):
        self.cfg = cfg

        # --------------------------------------------------------------
        # Determine dataset
        # --------------------------------------------------------------

        raw_dataset = str(
            cfg.get("data", {}).get(
                "dataset",
                "sample",
            )
        ).strip().lower()

        self.dataset_name = self.DATASET_ALIASES.get(
            raw_dataset,
            raw_dataset,
        )

        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                "\nUnsupported dataset in configuration.\n\n"
                f"Received: {raw_dataset!r}\n\n"
                "Supported datasets:\n"
                "  - cicids2017\n"
                "  - nsl_kdd\n"
                "  - unsw_nb15\n"
                "  - sample\n"
            )

        # --------------------------------------------------------------
        # Project root
        # --------------------------------------------------------------

        self.project_root = Path(__file__).resolve().parents[2]

        # --------------------------------------------------------------
        # Base paths
        # --------------------------------------------------------------

        configured_models_dir = Path(
            cfg.get("paths", {}).get(
                "models_dir",
                "models",
            )
        )

        configured_results_dir = Path(
            cfg.get("paths", {}).get(
                "results_dir",
                "results",
            )
        )

        configured_processed_dir = cfg.get(
            "data",
            {},
        ).get(
            "processed_dir",
            None,
        )

        # --------------------------------------------------------------
        # Resolve models directory
        # --------------------------------------------------------------

        if models_dir is None:

            base_models_path = (
                self.project_root
                / configured_models_dir
            )

            self.models_dir = (
                base_models_path
                / self.dataset_name
            )

        else:

            supplied_models_path = Path(
                models_dir
            )

            if not supplied_models_path.is_absolute():
                supplied_models_path = (
                    self.project_root
                    / supplied_models_path
                )

            supplied_models_path = (
                supplied_models_path.resolve()
            )

            if supplied_models_path.name.lower() == "models":

                self.models_dir = (
                    supplied_models_path
                    / self.dataset_name
                )

            elif (
                supplied_models_path.name.lower()
                == self.dataset_name.lower()
            ):

                self.models_dir = (
                    supplied_models_path
                )

            else:

                self.models_dir = (
                    supplied_models_path
                )

        self.models_dir = (
            self.models_dir.resolve()
        )

        # --------------------------------------------------------------
        # Resolve processed directory
        # --------------------------------------------------------------

        if processed_dir is not None:

            supplied_processed_path = Path(
                processed_dir
            )

            if not supplied_processed_path.is_absolute():
                supplied_processed_path = (
                    self.project_root
                    / supplied_processed_path
                )

            self.processed_dir = (
                supplied_processed_path.resolve()
            )

        elif configured_processed_dir:

            configured_processed_path = Path(
                configured_processed_dir
            )

            if not configured_processed_path.is_absolute():
                configured_processed_path = (
                    self.project_root
                    / configured_processed_path
                )

            self.processed_dir = (
                configured_processed_path.resolve()
            )

        else:

            self.processed_dir = (
                self.project_root
                / "data"
                / f"processed_{self.dataset_name}"
            ).resolve()

        # --------------------------------------------------------------
        # Results directory
        # --------------------------------------------------------------

        base_results_path = (
            self.project_root
            / configured_results_dir
        )

        self.results_dir = (
            base_results_path
            / self.dataset_name
        ).resolve()

        # --------------------------------------------------------------
        # Information
        # --------------------------------------------------------------

        print(
            "\n"
            "============================================================\n"
            "Tri-Hybrid XAI-IDS Inference Pipeline\n"
            "============================================================"
        )

        print(
            f"Dataset        : {self.dataset_name}"
        )

        print(
            f"Models         : {self.models_dir}"
        )

        print(
            f"Processed data : {self.processed_dir}"
        )

        print(
            f"Results        : {self.results_dir}"
        )

        print(
            "============================================================\n"
        )

        # ==================================================================
        # ARTIFACT PATHS
        # ==================================================================

        # --------------------------------------------------------------
        # ML
        #
        # IMPORTANT:
        # Chronological model MUST be preferred.
        #
        # ml_ensemble.joblib may be a reduced-class training artifact.
        # The meta learner was trained using the chronological model.
        # --------------------------------------------------------------

        ml_dir = (
            self.models_dir
            / "ml"
        )

        ml_model_candidates = [
            ml_dir / "ml_ensemble_chrono.joblib",
            ml_dir / "ml_ensemble.joblib",
        ]

        self.ml_model_path = self._first_existing(
            ml_model_candidates
        )

        # --------------------------------------------------------------
        # DL
        # --------------------------------------------------------------

        dl_dir = (
            self.models_dir
            / "dl"
        )

        dl_model_candidates = [
            dl_dir / "dl_model_chrono.keras",
            dl_dir / "dl_model.keras",
            dl_dir / "dl_model.h5",
        ]

        self.dl_model_path = self._first_existing(
            dl_model_candidates
        )

        # --------------------------------------------------------------
        # Meta learner
        # --------------------------------------------------------------

        meta_dir = (
            self.models_dir
            / "meta"
        )

        self.meta_model_path = (
            meta_dir
            / "meta_learner.joblib"
        )

        # --------------------------------------------------------------
        # Preprocessing artifacts
        # --------------------------------------------------------------

        self.label_encoder_path = (
            self.processed_dir
            / "label_encoder.joblib"
        )

        self.scaler_path = (
            self.processed_dir
            / "scaler.joblib"
        )

        # ==================================================================
        # SELECTED FEATURES
        # ==================================================================

        model_selected_features = (
            self.models_dir
            / "ml"
            / "selected_features.json"
        )

        results_selected_features = (
            self.results_dir
            / "shap"
            / "selected_features.json"
        )

        if model_selected_features.exists():

            self.selected_features_path = (
                model_selected_features
            )

        elif results_selected_features.exists():

            self.selected_features_path = (
                results_selected_features
            )

        else:

            self.selected_features_path = None

        # ==================================================================
        # FEATURE NAMES
        # ==================================================================

        feature_names_candidates = [
            self.processed_dir
            / "feature_names.joblib",

            self.processed_dir
            / "feature_names.json",
        ]

        self.feature_names_path = (
            self._first_existing(
                feature_names_candidates
            )
        )

        # ==================================================================
        # REQUIRED ARTIFACT VALIDATION
        # ==================================================================

        required_artifacts = {
            "ML model": self.ml_model_path,
            "DL model": self.dl_model_path,
            "Meta learner": self.meta_model_path,
            "Label encoder": self.label_encoder_path,
            "Scaler": self.scaler_path,
            "Feature names": self.feature_names_path,
            "Selected ML features": (
                self.selected_features_path
            ),
        }

        missing = []

        for name, path in required_artifacts.items():

            if path is None:

                missing.append(
                    f"{name}: <not found>"
                )

            elif not Path(path).exists():

                missing.append(
                    f"{name}: {path}"
                )

        if missing:

            message = (
                "\n"
                "============================================================\n"
                "Missing inference artifacts\n"
                "============================================================\n"
                f"Dataset: {self.dataset_name}\n\n"
                + "\n".join(missing)
                + "\n\n"
                "Expected dataset-specific structure:\n\n"
                f"models/{self.dataset_name}/ml/\n"
                f"models/{self.dataset_name}/dl/\n"
                f"models/{self.dataset_name}/meta/\n"
                f"data/processed_{self.dataset_name}/\n"
                f"results/{self.dataset_name}/\n"
                "============================================================\n"
            )

            raise FileNotFoundError(
                message
            )

        # ==================================================================
        # LOAD KERAS
        # ==================================================================

        import keras

        try:

            from src.dl.model import AdditiveAttention  # noqa: F401

        except ImportError:

            pass

        # ==================================================================
        # LOAD MODELS
        # ==================================================================

        self.ml_model = joblib.load(
            self.ml_model_path
        )

        self.dl_model = keras.models.load_model(
            self.dl_model_path
        )

        self.meta_model = joblib.load(
            self.meta_model_path
        )

        # ==================================================================
        # LOAD LABEL ENCODER
        # ==================================================================

        self.label_encoder = joblib.load(
            self.label_encoder_path
        )

        self.global_classes = [
            str(label)
            for label in self.label_encoder.classes_
        ]

        # ==================================================================
        # LOAD SCALER
        # ==================================================================

        self.scaler = joblib.load(
            self.scaler_path
        )

        # ==================================================================
        # LOAD FEATURE NAMES
        # ==================================================================

        if (
            self.feature_names_path.suffix.lower()
            == ".joblib"
        ):

            self.feature_names = joblib.load(
                self.feature_names_path
            )

        else:

            with open(
                self.feature_names_path,
                "r",
                encoding="utf-8",
            ) as f:

                self.feature_names = json.load(
                    f
                )

        if not isinstance(
            self.feature_names,
            (
                list,
                tuple,
                np.ndarray,
            ),
        ):

            raise ValueError(
                "feature_names artifact must contain "
                "a list, tuple, or numpy array."
            )

        self.feature_names = [
            str(feature).strip()
            for feature in self.feature_names
        ]

        if not self.feature_names:

            raise ValueError(
                "The trained feature list is empty.\n"
                f"Artifact: {self.feature_names_path}"
            )

        # --------------------------------------------------------------
        # Duplicate feature validation
        # --------------------------------------------------------------

        if len(
            self.feature_names
        ) != len(
            set(self.feature_names)
        ):

            duplicates = sorted(
                {
                    feature
                    for feature in self.feature_names
                    if self.feature_names.count(
                        feature
                    ) > 1
                }
            )

            raise ValueError(
                "Duplicate feature names detected.\n\n"
                f"Duplicates: {duplicates}\n\n"
                f"Artifact: {self.feature_names_path}"
            )

        # ==================================================================
        # LOAD SELECTED ML FEATURES
        # ==================================================================

        with open(
            self.selected_features_path,
            "r",
            encoding="utf-8",
        ) as f:

            self.selected_ml_features = (
                json.load(f)
            )

        if not isinstance(
            self.selected_ml_features,
            list,
        ):

            raise ValueError(
                "selected_features.json must contain "
                "a JSON list."
            )

        self.selected_ml_features = [
            str(feature).strip()
            for feature in self.selected_ml_features
        ]

        if not self.selected_ml_features:

            raise ValueError(
                "selected_features.json contains no features.\n"
                f"Artifact: {self.selected_features_path}"
            )

        # ==================================================================
        # FEATURE INDEX MAP
        # ==================================================================

        feature_index = {
            feature: index
            for index, feature
            in enumerate(
                self.feature_names
            )
        }

        missing_selected_features = [
            feature
            for feature
            in self.selected_ml_features
            if feature not in feature_index
        ]

        if missing_selected_features:

            raise ValueError(
                "\n"
                "============================================================\n"
                "Feature artifact mismatch\n"
                "============================================================\n"
                f"Dataset: {self.dataset_name}\n\n"
                "The selected ML feature list contains features "
                "that are not present in feature_names.\n\n"
                "Missing selected features:\n"
                f"{missing_selected_features}\n\n"
                f"Feature names: {self.feature_names_path}\n"
                f"Selected features: {self.selected_features_path}\n"
                "============================================================\n"
            )

        # ==================================================================
        # PRECOMPUTED ML FEATURE INDICES
        # ==================================================================

        self._ml_feature_indices = np.array(
            [
                feature_index[feature]
                for feature
                in self.selected_ml_features
            ],
            dtype=np.int64,
        )

        # ==================================================================
        # SEQUENCE CONFIG
        # ==================================================================

        self.window_size = int(
            cfg.get(
                "sequence",
                {},
            ).get(
                "window_size",
                10,
            )
        )

        self.stride = int(
            cfg.get(
                "sequence",
                {},
            ).get(
                "stride",
                1,
            )
        )

        # ==================================================================
        # MODEL DIMENSION VALIDATION
        # ==================================================================

        self._validate_model_dimensions()

        # ==================================================================
        # DL CLASS MAPPING
        # ==================================================================

        self.dl_class_indices = (
            self._detect_dl_class_indices()
        )

        # ==================================================================
        # ARTIFACT INFORMATION
        # ==================================================================

        self.artifact_paths = {
            "ml_model": self.ml_model_path,
            "dl_model": self.dl_model_path,
            "meta_model": self.meta_model_path,
            "selected_features": (
                self.selected_features_path
            ),
            "label_encoder": (
                self.label_encoder_path
            ),
            "feature_names": (
                self.feature_names_path
            ),
            "scaler": self.scaler_path,
        }

        # ==================================================================
        # FINAL INFORMATION
        # ==================================================================

        print(
            "Loaded inference artifacts successfully."
        )

        print(
            f"Dataset                 : "
            f"{self.dataset_name}"
        )

        print(
            f"ML artifact             : "
            f"{self.ml_model_path.name}"
        )

        print(
            f"Total features          : "
            f"{len(self.feature_names)}"
        )

        print(
            f"Selected ML features    : "
            f"{len(self.selected_ml_features)}"
        )

        print(
            f"Sequence window         : "
            f"{self.window_size}"
        )

        print(
            f"Global classes          : "
            f"{len(self.global_classes)}"
        )

        print(
            f"ML class outputs        : "
            f"{len(getattr(self.ml_model, 'classes_', []))}"
        )

        print(
            f"DL class outputs        : "
            f"{len(self.dl_class_indices)}"
        )

        print(
            "============================================================\n"
        )

    # ==================================================================
    # UTILITY
    # ==================================================================

    @staticmethod
    def _first_existing(
        candidates,
    ):
        """
        Return the first existing path from candidates.
        """

        for path in candidates:

            if path is not None and Path(
                path
            ).exists():

                return Path(path)

        return None

    # ==================================================================
    # MODEL DIMENSION VALIDATION
    # ==================================================================

    def _validate_model_dimensions(
        self,
    ):
        """
        Validate dimensions and class compatibility.

        This catches stale/cross-dataset artifacts before inference.
        """

        # --------------------------------------------------------------
        # ML feature count
        # --------------------------------------------------------------

        ml_expected = getattr(
            self.ml_model,
            "n_features_in_",
            None,
        )

        if ml_expected is not None:

            ml_expected = int(
                ml_expected
            )

            actual_ml_features = len(
                self.selected_ml_features
            )

            if (
                ml_expected
                != actual_ml_features
            ):

                raise ValueError(
                    "\n"
                    "ML model feature mismatch.\n\n"
                    f"Dataset: {self.dataset_name}\n"
                    f"ML model expects: "
                    f"{ml_expected}\n"
                    f"Selected features: "
                    f"{actual_ml_features}\n\n"
                    "This usually means ML model and "
                    "selected_features.json came from "
                    "different training runs.\n"
                )

        # --------------------------------------------------------------
        # ML class count
        #
        # This is the critical validation for the current project.
        # --------------------------------------------------------------

        ml_classes = getattr(
            self.ml_model,
            "classes_",
            None,
        )

        if ml_classes is not None:

            ml_classes = np.asarray(
                ml_classes
            ).reshape(-1)

            expected_classes = len(
                self.global_classes
            )

            if len(ml_classes) != expected_classes:

                raise ValueError(
                    "\n"
                    "============================================================\n"
                    "ML model class-count mismatch\n"
                    "============================================================\n"
                    f"Dataset: {self.dataset_name}\n\n"
                    f"ML artifact: {self.ml_model_path}\n"
                    f"ML model classes: {len(ml_classes)}\n"
                    f"Label encoder classes: {expected_classes}\n\n"
                    f"ML classes: {ml_classes.tolist()}\n\n"
                    "The selected ML model is incompatible with "
                    "the DL/meta artifacts for this dataset.\n\n"
                    "The chronological ML artifact should be used "
                    "when it exists:\n"
                    "    ml_ensemble_chrono.joblib\n"
                    "============================================================\n"
                )

            # ----------------------------------------------------------
            # Numeric class-index validation
            # ----------------------------------------------------------

            try:

                expected_indices = np.arange(
                    expected_classes
                )

                numeric_ml_classes = np.asarray(
                    ml_classes,
                    dtype=np.int64,
                )

                if not np.array_equal(
                    numeric_ml_classes,
                    expected_indices,
                ):

                    raise ValueError(
                        "\n"
                        "============================================================\n"
                        "ML model class-index mismatch\n"
                        "============================================================\n"
                        f"Dataset: {self.dataset_name}\n"
                        f"ML artifact: {self.ml_model_path}\n\n"
                        f"ML classes: "
                        f"{numeric_ml_classes.tolist()}\n"
                        f"Expected: "
                        f"{expected_indices.tolist()}\n\n"
                        "The ML model class ordering does not match "
                        "the label encoder ordering used by the "
                        "fusion pipeline.\n"
                        "============================================================\n"
                    )

            except ValueError:
                raise

            except Exception:
                # Some datasets may use non-numeric class labels.
                pass

        # --------------------------------------------------------------
        # Scaler
        # --------------------------------------------------------------

        scaler_features = getattr(
            self.scaler,
            "n_features_in_",
            None,
        )

        if scaler_features is not None:

            scaler_features = int(
                scaler_features
            )

            actual_features = len(
                self.feature_names
            )

            if (
                scaler_features
                != actual_features
            ):

                raise ValueError(
                    "\n"
                    "Scaler feature mismatch.\n\n"
                    f"Dataset: {self.dataset_name}\n"
                    f"Scaler expects: "
                    f"{scaler_features}\n"
                    f"Feature names: "
                    f"{actual_features}\n\n"
                    "This indicates that the scaler and "
                    "feature metadata were generated from "
                    "different preprocessing runs.\n"
                )

        # --------------------------------------------------------------
        # DL model input feature dimension
        # --------------------------------------------------------------

        try:

            input_shape = (
                self.dl_model.input_shape
            )

            if isinstance(
                input_shape,
                list,
            ):

                input_shape = input_shape[0]

            if (
                input_shape is not None
                and len(input_shape) >= 3
            ):

                dl_features = input_shape[-1]

                if dl_features is not None:

                    dl_features = int(
                        dl_features
                    )

                    actual_features = len(
                        self.feature_names
                    )

                    if (
                        dl_features
                        != actual_features
                    ):

                        raise ValueError(
                            "\n"
                            "DL model feature mismatch.\n\n"
                            f"Dataset: "
                            f"{self.dataset_name}\n"
                            f"DL model expects: "
                            f"{dl_features}\n"
                            f"Current feature metadata: "
                            f"{actual_features}\n\n"
                            f"DL model: "
                            f"{self.dl_model_path}\n"
                            f"Feature names: "
                            f"{self.feature_names_path}\n\n"
                            "This is exactly the type of "
                            "cross-dataset/stale-artifact "
                            "mixing that this pipeline is "
                            "designed to prevent.\n"
                        )

        except ValueError:
            raise

        except Exception:
            pass

    # ==================================================================
    # DL CLASS MAPPING
    # ==================================================================

    def _detect_dl_class_indices(
        self,
    ):
        """
        Determine mapping between DL output columns and
        global label-encoder class indices.
        """

        metadata_candidates = [
            self.models_dir
            / "dl"
            / "class_indices.json",

            self.models_dir
            / "dl"
            / "dl_class_indices.json",

            self.models_dir
            / "dl"
            / "class_mapping.json",

            self.processed_dir
            / "dl_class_indices.json",
        ]

        for metadata_path in metadata_candidates:

            if not metadata_path.exists():
                continue

            try:

                with open(
                    metadata_path,
                    "r",
                    encoding="utf-8",
                ) as f:

                    metadata = json.load(f)

                if isinstance(
                    metadata,
                    list,
                ):

                    indices = [
                        int(x)
                        for x in metadata
                    ]

                    self._validate_dl_indices(
                        indices
                    )

                    return indices

                if isinstance(
                    metadata,
                    dict,
                ):

                    values = (
                        metadata.get(
                            "class_indices"
                        )
                    )

                    if values is None:

                        values = (
                            metadata.get(
                                "dl_class_indices"
                            )
                        )

                    if isinstance(
                        values,
                        list,
                    ):

                        indices = [
                            int(x)
                            for x in values
                        ]

                        self._validate_dl_indices(
                            indices
                        )

                        return indices

            except Exception as exc:

                raise ValueError(
                    "Unable to read DL class mapping artifact:\n"
                    f"{metadata_path}"
                ) from exc

        # --------------------------------------------------------------
        # Determine DL output dimension
        # --------------------------------------------------------------

        dl_output_dim = None

        try:

            output_shape = (
                self.dl_model.output_shape
            )

            if isinstance(
                output_shape,
                list,
            ):

                probability_shape = (
                    output_shape[0]
                )

            else:

                probability_shape = (
                    output_shape
                )

            if (
                probability_shape is not None
                and len(probability_shape) >= 2
            ):

                dl_output_dim = int(
                    probability_shape[-1]
                )

        except Exception:
            dl_output_dim = None

        if dl_output_dim is None:

            raise ValueError(
                "\n"
                "Unable to determine DL output dimension.\n\n"
                f"Dataset: {self.dataset_name}\n"
                f"Global classes: {self.global_classes}\n\n"
                "Expected a Keras model with a readable "
                "output_shape.\n"
            )

        # --------------------------------------------------------------
        # Direct mapping
        # --------------------------------------------------------------

        if (
            dl_output_dim
            == len(self.global_classes)
        ):

            return list(
                range(
                    len(
                        self.global_classes
                    )
                )
            )

        # --------------------------------------------------------------
        # Reduced-class mapping
        # --------------------------------------------------------------

        if (
            dl_output_dim
            < len(self.global_classes)
        ):

            return list(
                range(
                    dl_output_dim
                )
            )

        raise ValueError(
            "\n"
            "DL output has more classes than the global "
            "label encoder.\n\n"
            f"Dataset: {self.dataset_name}\n"
            f"DL output classes: {dl_output_dim}\n"
            f"Global classes: {len(self.global_classes)}\n\n"
            "The trained DL model and label encoder do not "
            "belong to the same training run.\n"
        )

    # ==================================================================
    # DL CLASS INDEX VALIDATION
    # ==================================================================

    def _validate_dl_indices(
        self,
        indices,
    ):
        """
        Validate externally stored DL class mapping.
        """

        if not indices:

            raise ValueError(
                "DL class mapping is empty."
            )

        if len(
            indices
        ) != len(
            set(indices)
        ):

            raise ValueError(
                "DL class mapping contains duplicate indices:\n"
                f"{indices}"
            )

        max_index = (
            len(self.global_classes)
            - 1
        )

        invalid = [
            index
            for index in indices
            if index < 0
            or index > max_index
        ]

        if invalid:

            raise ValueError(
                "DL class mapping contains invalid global "
                "class indices.\n\n"
                f"Invalid indices: {invalid}\n"
                f"Valid range: 0-{max_index}"
            )

    # ==================================================================
    # INPUT VALIDATION
    # ==================================================================

    def _validate_window_columns(
        self,
        window_df,
    ):
        """
        Ensure every trained feature exists in inference window.
        """

        missing_columns = [
            feature
            for feature
            in self.feature_names
            if feature not in window_df.columns
        ]

        if missing_columns:

            raise ValueError(
                "\n"
                "Inference input is missing required features.\n\n"
                f"Dataset: {self.dataset_name}\n\n"
                "Missing features:\n"
                f"{missing_columns}\n\n"
                f"Expected total features: "
                f"{len(self.feature_names)}"
            )

    # ==================================================================
    # PREDICTION
    # ==================================================================

    def predict_window(
        self,
        window_df,
    ):
        """
        Predict one chronological flow window.
        """

        # --------------------------------------------------------------
        # Window size
        # --------------------------------------------------------------

        if len(window_df) != self.window_size:

            raise ValueError(
                "\n"
                "Invalid inference window size.\n\n"
                f"Dataset: {self.dataset_name}\n"
                f"Expected rows: {self.window_size}\n"
                f"Received rows: {len(window_df)}"
            )

        # --------------------------------------------------------------
        # Columns
        # --------------------------------------------------------------

        self._validate_window_columns(
            window_df
        )

        # --------------------------------------------------------------
        # Feature order
        # --------------------------------------------------------------

        try:

            X_window = (
                window_df[
                    self.feature_names
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )

        except Exception as exc:

            raise ValueError(
                "\n"
                "Unable to convert inference features "
                "to numeric float32 values.\n\n"
                f"Dataset: {self.dataset_name}\n"
                "The inference DataFrame must contain the "
                "same numeric/encoded feature representation "
                "used during training."
            ) from exc

        # --------------------------------------------------------------
        # Shape
        # --------------------------------------------------------------

        expected_shape = (
            self.window_size,
            len(self.feature_names),
        )

        if X_window.shape != expected_shape:

            raise ValueError(
                "\n"
                "Unexpected inference feature shape.\n\n"
                f"Expected: {expected_shape}\n"
                f"Received: {X_window.shape}\n"
                f"Dataset: {self.dataset_name}"
            )

        # --------------------------------------------------------------
        # Scale
        # --------------------------------------------------------------

        X_full_scaled = (
            self.scaler.transform(
                X_window
            )
        )

        X_full_scaled = np.asarray(
            X_full_scaled,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # ML branch
        # --------------------------------------------------------------

        X_ml_last = (
            X_full_scaled[-1][
                self._ml_feature_indices
            ]
        )

        X_ml_last = np.asarray(
            X_ml_last,
            dtype=np.float32,
        )

        ml_proba = (
            self.ml_model
            .predict_proba(
                X_ml_last.reshape(
                    1,
                    -1,
                )
            )[0]
        )

        ml_proba = np.asarray(
            ml_proba,
            dtype=np.float64,
        ).reshape(-1)

        # --------------------------------------------------------------
        # Runtime ML probability validation
        # --------------------------------------------------------------

        expected_ml_classes = len(
            self.global_classes
        )

        if len(ml_proba) != expected_ml_classes:

            raise ValueError(
                "\n"
                "============================================================\n"
                "ML probability/class mismatch during inference\n"
                "============================================================\n"
                f"Dataset: {self.dataset_name}\n"
                f"ML artifact: {self.ml_model_path}\n"
                f"ML probabilities: {len(ml_proba)}\n"
                f"Global classes: {expected_ml_classes}\n\n"
                "The loaded ML artifact is incompatible with "
                "the label encoder/meta learner.\n"
                "Do not pad or truncate the probability vector.\n"
                "Use the matching chronological ML artifact.\n"
                "============================================================\n"
            )

        # --------------------------------------------------------------
        # DL branch
        # --------------------------------------------------------------

        dl_input = (
            X_full_scaled
            .reshape(
                1,
                self.window_size,
                len(self.feature_names),
            )
        )

        # --------------------------------------------------------------
        # Final safety check against Keras model input
        # --------------------------------------------------------------

        try:

            model_input_shape = (
                self.dl_model.input_shape
            )

            if isinstance(
                model_input_shape,
                list,
            ):

                model_input_shape = (
                    model_input_shape[0]
                )

            if (
                model_input_shape is not None
                and len(model_input_shape) >= 3
            ):

                expected_dl_shape = (
                    model_input_shape[-2],
                    model_input_shape[-1],
                )

                actual_dl_shape = (
                    dl_input.shape[-2],
                    dl_input.shape[-1],
                )

                if (
                    expected_dl_shape
                    != actual_dl_shape
                ):

                    raise ValueError(
                        "\n"
                        "DL input shape mismatch.\n\n"
                        f"Dataset: {self.dataset_name}\n"
                        f"DL model expects: "
                        f"{expected_dl_shape}\n"
                        f"Inference produced: "
                        f"{actual_dl_shape}\n\n"
                        f"DL model: "
                        f"{self.dl_model_path}\n"
                        f"Feature artifact: "
                        f"{self.feature_names_path}\n\n"
                        "This indicates incompatible trained "
                        "artifacts."
                    )

        except ValueError:
            raise

        except Exception:
            pass

        # --------------------------------------------------------------
        # Keras prediction
        # --------------------------------------------------------------

        dl_output = (
            self.dl_model.predict(
                dl_input,
                verbose=0,
            )
        )

        # --------------------------------------------------------------
        # Extract probabilities + attention
        # --------------------------------------------------------------

        if isinstance(
            dl_output,
            (list, tuple),
        ):

            if len(dl_output) == 0:

                raise ValueError(
                    "DL model returned an empty output."
                )

            dl_proba = np.asarray(
                dl_output[0]
            )[0]

            if len(dl_output) >= 2:

                attn = np.asarray(
                    dl_output[1]
                )[0]

            else:

                attn = np.zeros(
                    self.window_size,
                    dtype=np.float64,
                )

        else:

            dl_proba = np.asarray(
                dl_output
            )[0]

            attn = np.zeros(
                self.window_size,
                dtype=np.float64,
            )

        dl_proba = np.asarray(
            dl_proba,
            dtype=np.float64,
        ).reshape(-1)

        attn = np.asarray(
            attn,
            dtype=np.float64,
        )

        # --------------------------------------------------------------
        # Validate DL class count
        # --------------------------------------------------------------

        expected_dl_classes = len(
            self.dl_class_indices
        )

        if (
            len(dl_proba)
            != expected_dl_classes
        ):

            raise ValueError(
                "\n"
                "DL probability/class mapping mismatch.\n\n"
                f"Dataset: {self.dataset_name}\n"
                f"DL probabilities: {len(dl_proba)}\n"
                f"DL class mapping: "
                f"{self.dl_class_indices}\n"
                f"Expected: {expected_dl_classes}\n\n"
                f"Global classes: {self.global_classes}\n"
                f"DL model: {self.dl_model_path}\n"
            )

        # --------------------------------------------------------------
        # META LEARNER
        #
        # IMPORTANT:
        # Original ML and DL probability vectors are passed
        # unchanged.
        # --------------------------------------------------------------

        meta_features = np.hstack(
            [
                ml_proba,
                dl_proba,
            ]
        ).reshape(
            1,
            -1,
        )

        # --------------------------------------------------------------
        # Validate meta input dimension
        # --------------------------------------------------------------

        meta_expected = getattr(
            self.meta_model,
            "n_features_in_",
            None,
        )

        if meta_expected is not None:

            meta_expected = int(
                meta_expected
            )

            if (
                meta_features.shape[1]
                != meta_expected
            ):

                raise ValueError(
                    "\n"
                    "Meta learner input dimension mismatch.\n\n"
                    f"Dataset: {self.dataset_name}\n"
                    f"Meta learner expects: "
                    f"{meta_expected}\n"
                    f"Received: "
                    f"{meta_features.shape[1]}\n\n"
                    f"ML probability length: "
                    f"{len(ml_proba)}\n"
                    f"DL probability length: "
                    f"{len(dl_proba)}\n\n"
                    "The ML, DL, and meta artifacts are "
                    "not dimensionally compatible."
                )

        # --------------------------------------------------------------
        # Final fused prediction
        # --------------------------------------------------------------

        final_proba = (
            self.meta_model
            .predict_proba(
                meta_features
            )[0]
        )

        final_proba = np.asarray(
            final_proba,
            dtype=np.float64,
        ).reshape(-1)

        # --------------------------------------------------------------
        # Final prediction index
        # --------------------------------------------------------------

        pred_idx = int(
            np.argmax(
                final_proba
            )
        )

        if pred_idx >= len(
            self.label_encoder.classes_
        ):

            raise ValueError(
                "\n"
                "Meta learner predicted an invalid class index.\n\n"
                f"Dataset: {self.dataset_name}\n"
                f"Predicted index: {pred_idx}\n"
                f"Global class count: "
                f"{len(self.label_encoder.classes_)}"
            )

        # --------------------------------------------------------------
        # Final label
        # --------------------------------------------------------------

        pred_label = (
            self.label_encoder
            .inverse_transform(
                [pred_idx]
            )[0]
        )

        # --------------------------------------------------------------
        # Return
        # --------------------------------------------------------------

        return {
            "dataset": self.dataset_name,
            "ml_proba": ml_proba,
            "dl_proba": dl_proba,
            "final_proba": final_proba,
            "pred_idx": pred_idx,
            "pred_label": pred_label,
            "attention": attn,
            "X_ml_last": X_ml_last,
            "X_full_scaled": X_full_scaled,
        }

    # ==================================================================
    # EXPLAINABILITY
    # ==================================================================

    def explain_window(
        self,
        window_df,
        ml_background_X: np.ndarray,
        ml_train_X: np.ndarray,
    ):
        """
        Run complete explainability pipeline:

            SHAP
            LIME
            Attention
            Cross-explainer agreement
        """

        # --------------------------------------------------------------
        # Prediction
        # --------------------------------------------------------------

        result = self.predict_window(
            window_df
        )

        # --------------------------------------------------------------
        # XAI configuration
        # --------------------------------------------------------------

        xai_cfg = self.cfg.get(
            "xai",
            {},
        )

        shap_top_k = int(
            xai_cfg.get(
                "shap_top_k",
                8,
            )
        )

        lime_top_k = int(
            xai_cfg.get(
                "lime_top_k",
                8,
            )
        )

        lime_num_samples = int(
            xai_cfg.get(
                "lime_num_samples",
                500,
            )
        )

        attention_top_k = int(
            xai_cfg.get(
                "attention_top_k",
                8,
            )
        )

        # --------------------------------------------------------------
        # SHAP
        # --------------------------------------------------------------

        shap_top = (
            shap_local_explanation(
                self.ml_model,
                ml_background_X,
                result["X_ml_last"],
                self.selected_ml_features,
                shap_top_k,
                result["pred_idx"],
            )
        )

        # --------------------------------------------------------------
        # LIME
        # --------------------------------------------------------------

        lime_top = (
            lime_local_explanation(
                self.ml_model,
                ml_train_X,
                result["X_ml_last"],
                self.selected_ml_features,
                list(
                    self.label_encoder.classes_
                ),
                lime_top_k,
                lime_num_samples,
                result["pred_idx"],
            )
        )

        # --------------------------------------------------------------
        # Attention
        # --------------------------------------------------------------

        attn_top = (
            attention_explanation(
                result["attention"],
                attention_top_k,
            )
        )

        # --------------------------------------------------------------
        # Agreement
        # --------------------------------------------------------------

        agreement_cfg = self.cfg.get(
            "agreement",
            {},
        )

        trusted_threshold = float(
            agreement_cfg.get(
                "trusted_threshold",
                0.5,
            )
        )

        agreement = (
            cross_explainer_agreement(
                shap_top,
                lime_top,
                result["attention"],
                threshold=trusted_threshold,
            )
        )

        # --------------------------------------------------------------
        # Add XAI results
        # --------------------------------------------------------------

        result.update(
            {
                "shap_top": shap_top,
                "lime_top": lime_top,
                "attn_top": attn_top,
                "agreement": agreement,
            }
        )

        return result
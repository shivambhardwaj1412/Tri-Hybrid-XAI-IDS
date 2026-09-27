"""
Tri-Hybrid-XAI-IDS
Dataset-Agnostic Memory-Safe Preprocessing Pipeline

Supported datasets:
    - CICIDS2017
    - NSL-KDD
    - UNSW-NB15
    - Sample datasets

Dataset selection is controlled ONLY through config.yaml.
"""

from __future__ import annotations

import os
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder
)
from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(message)s]"
)

logger = logging.getLogger(__name__)


# ============================================================
# PROCESSED DATA CONTAINER
# ============================================================

class ProcessedData:
    """
    Common output container used by:
        - train_all.py
        - inference
        - dashboard
    """

    def __init__(
        self,
        X_train=None,
        X_val=None,
        X_test=None,
        y_train=None,
        y_val=None,
        y_test=None,
        scaler=None,
        label_encoder=None,
        feature_names=None,
        timestamps_train=None,
        timestamps_val=None,
        timestamps_test=None,
        dataset=None,
    ):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.scaler = scaler
        self.label_encoder = label_encoder
        self.feature_names = feature_names

        self.timestamps_train = timestamps_train
        self.timestamps_val = timestamps_val
        self.timestamps_test = timestamps_test

        self.dataset = dataset


# ============================================================
# HELPERS
# ============================================================

def ensure_directory(path):
    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


def _read_csv_robust(path):
    """
    Robust CSV reader for normal CSV datasets.
    """

    try:
        return pd.read_csv(
            path,
            encoding="utf-8",
            low_memory=True
        )

    except UnicodeDecodeError:

        logger.warning(
            f"UTF-8 decoding failed for "
            f"{os.path.basename(path)}. "
            f"Retrying with latin-1."
        )

        return pd.read_csv(
            path,
            encoding="latin-1",
            low_memory=True
        )


# ============================================================
# CICIDS2017
# ============================================================

def _load_cicids2017_memory_safe(cfg):
    """
    Memory-safe CICIDS2017 loader.

    Retains only max_raw_rows across all CICIDS CSV files.
    """

    raw_dir = Path(
        cfg["data"]["raw_dir"]
    )

    cicids_dir = raw_dir / "cicids2017"

    if not cicids_dir.exists():

        raise FileNotFoundError(
            f"CICIDS2017 directory not found:\n"
            f"{cicids_dir}"
        )

    csv_files = sorted(
        cicids_dir.glob("*.csv")
    )

    if not csv_files:

        raise FileNotFoundError(
            f"No CICIDS2017 CSV files found in:\n"
            f"{cicids_dir}"
        )

    max_raw_rows = int(
        cfg["data"].get(
            "max_raw_rows",
            120000
        )
    )

    chunk_size = int(
        cfg["data"].get(
            "cicids_chunk_size",
            25000
        )
    )

    seed = int(
        cfg["project"].get(
            "seed",
            42
        )
    )

    logger.info("=" * 60)
    logger.info(
        "CICIDS2017 MEMORY-SAFE LOADING"
    )
    logger.info(
        f"Maximum rows: {max_raw_rows:,}"
    )
    logger.info(
        f"CSV files found: {len(csv_files)}"
    )
    logger.info("=" * 60)

    rows_per_file = max(
        1,
        max_raw_rows // len(csv_files)
    )

    retained_chunks = []

    total_rows = 0

    for csv_path in csv_files:

        if total_rows >= max_raw_rows:
            break

        logger.info(
            f"Reading CICIDS file: "
            f"{csv_path.name}"
        )

        remaining_for_file = rows_per_file

        if csv_path == csv_files[-1]:

            remaining_for_file = min(
                remaining_for_file,
                max_raw_rows - total_rows
            )

        reader = pd.read_csv(
            csv_path,
            encoding="latin-1",
            low_memory=True,
            chunksize=chunk_size
        )

        rows_from_file = 0

        for chunk in reader:

            if rows_from_file >= remaining_for_file:
                break

            remaining = (
                remaining_for_file -
                rows_from_file
            )

            if len(chunk) > remaining:

                chunk = chunk.sample(
                    n=remaining,
                    random_state=seed
                )

            retained_chunks.append(
                chunk
            )

            rows_from_file += len(chunk)
            total_rows += len(chunk)

            if rows_from_file >= remaining_for_file:
                break

            if total_rows >= max_raw_rows:
                break

        logger.info(
            f"Retained {rows_from_file:,} "
            f"rows from {csv_path.name}"
        )

    if not retained_chunks:

        raise RuntimeError(
            "No CICIDS2017 rows were loaded."
        )

    df = pd.concat(
        retained_chunks,
        ignore_index=True
    )

    retained_chunks.clear()

    logger.info(
        f"CICIDS2017 loaded: {df.shape}"
    )

    return df


def adapt_cicids2017(df):

    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    label_candidates = [
        "Label",
        "label",
        "LABEL"
    ]

    label_column = None

    for candidate in label_candidates:

        if candidate in df.columns:

            label_column = candidate
            break

    if label_column is None:

        raise ValueError(
            "CICIDS2017 Label column not found.\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    if label_column != "Label":

        df.rename(
            columns={
                label_column: "Label"
            },
            inplace=True
        )

    return df


# ============================================================
# NSL-KDD
# ============================================================

# Official NSL-KDD structure:
#
# 41 features
# + Label
# + difficulty
# = 43 columns
#
# IMPORTANT:
# Different copies of NSL-KDD may use different delimiters.
# This loader detects the delimiter instead of assuming comma.

NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "Label",
    "difficulty"
]


def _detect_nsl_kdd_delimiter(path):
    """
    Detect the delimiter used by an NSL-KDD file.

    Supported:
        comma ,
        tab \\t
        semicolon ;
        pipe |

    Returns:
        delimiter string
    """

    with open(
        path,
        "r",
        encoding="latin-1",
        errors="replace"
    ) as f:

        first_nonempty_line = None

        for line in f:

            line = line.strip()

            if line:

                first_nonempty_line = line
                break

    if first_nonempty_line is None:

        raise ValueError(
            f"NSL-KDD file is empty:\n{path}"
        )

    delimiter_candidates = [
        ",",
        "\t",
        ";",
        "|"
    ]

    counts = {
        delimiter: first_nonempty_line.count(
            delimiter
        )
        for delimiter in delimiter_candidates
    }

    best_delimiter = max(
        counts,
        key=counts.get
    )

    best_count = counts[
        best_delimiter
    ]

    logger.info(
        f"NSL-KDD delimiter detection: "
        f"{counts}"
    )

    if best_count == 0:

        raise ValueError(
            f"Could not detect a valid delimiter "
            f"in NSL-KDD file:\n{path}\n\n"
            f"First line:\n"
            f"{first_nonempty_line[:500]}"
        )

    logger.info(
        f"Detected NSL-KDD delimiter: "
        f"{repr(best_delimiter)}"
    )

    return best_delimiter


def _read_nsl_kdd_file(path):
    """
    Read an NSL-KDD raw file safely.

    The loader:
      1. Detects the delimiter.
      2. Reads without assuming a header.
      3. Validates 43 columns.
      4. Detects accidental header rows.
      5. Validates duration and Label.
    """

    logger.info(
        f"Reading NSL-KDD file: "
        f"{path.name}"
    )

    delimiter = _detect_nsl_kdd_delimiter(
        path
    )

    # --------------------------------------------------------
    # First read
    # --------------------------------------------------------

    df = pd.read_csv(
        path,
        sep=delimiter,
        header=None,
        encoding="latin-1",
        skipinitialspace=True,
        engine="python"
    )

    logger.info(
        f"Raw NSL-KDD columns detected: "
        f"{df.shape[1]}"
    )

    # --------------------------------------------------------
    # Validate column count BEFORE assigning names
    # --------------------------------------------------------

    if df.shape[1] != 43:

        raise ValueError(
            "\nInvalid NSL-KDD file structure.\n"
            f"File: {path.name}\n"
            f"Expected: 43 columns\n"
            f"Detected: {df.shape[1]} columns\n"
            f"Detected delimiter: "
            f"{repr(delimiter)}\n\n"
            "This usually means the NSL-KDD file "
            "is not in the expected raw format."
        )

    # --------------------------------------------------------
    # Assign official NSL-KDD names
    # --------------------------------------------------------

    df.columns = NSL_KDD_COLUMNS

    # --------------------------------------------------------
    # Strip whitespace
    # --------------------------------------------------------

    string_columns = [
        "protocol_type",
        "service",
        "flag",
        "Label"
    ]

    for col in string_columns:

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Normalize labels
    # --------------------------------------------------------

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.rstrip(".")
    )

    # --------------------------------------------------------
    # Detect accidental header row
    # --------------------------------------------------------

    if len(df) > 0:

        first_label = str(
            df.iloc[0]["Label"]
        ).strip().lower()

        if first_label == "label":

            logger.warning(
                "Header row detected inside "
                "NSL-KDD file. Removing it."
            )

            df = df.iloc[
                1:
            ].reset_index(
                drop=True
            )

    # --------------------------------------------------------
    # Convert duration to numeric
    # --------------------------------------------------------

    df["duration"] = pd.to_numeric(
        df["duration"],
        errors="coerce"
    )

    invalid_duration = int(
        df["duration"].isna().sum()
    )

    if invalid_duration > 0:

        logger.warning(
            f"NSL-KDD duration contains "
            f"{invalid_duration:,} "
            f"non-numeric values."
        )

    # --------------------------------------------------------
    # Validate Label
    # --------------------------------------------------------

    valid_labels = (
        df["Label"].notna()
        & (
            df["Label"]
            .astype(str)
            .str.strip()
            .str.lower()
            != "nan"
        )
        & (
            df["Label"]
            .astype(str)
            .str.strip()
            != ""
        )
    )

    label_count = int(
        valid_labels.sum()
    )

    logger.info(
        f"Valid NSL-KDD labels detected: "
        f"{label_count:,}"
    )

    if label_count == 0:

        raise ValueError(
            "\nNSL-KDD Label column contains "
            "zero valid labels.\n\n"
            "The file was probably parsed with "
            "the wrong delimiter or is not a "
            "standard NSL-KDD file.\n\n"
            f"File: {path}\n"
            f"Delimiter detected: "
            f"{repr(delimiter)}"
        )

    # --------------------------------------------------------
    # Remove difficulty
    # --------------------------------------------------------

    if "difficulty" in df.columns:

        df.drop(
            columns=["difficulty"],
            inplace=True
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    expected_columns = 42

    if df.shape[1] != expected_columns:

        raise ValueError(
            "Unexpected NSL-KDD column count "
            "after removing difficulty.\n"
            f"Expected: {expected_columns}\n"
            f"Found: {df.shape[1]}"
        )

    # --------------------------------------------------------
    # Validate categorical columns
    # --------------------------------------------------------

    required_categorical = [
        "protocol_type",
        "service",
        "flag"
    ]

    missing_categorical = [
        col
        for col in required_categorical
        if col not in df.columns
    ]

    if missing_categorical:

        raise ValueError(
            "NSL-KDD categorical columns missing:\n"
            f"{missing_categorical}"
        )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    logger.info(
        f"NSL-KDD file validated: "
        f"{len(df):,} rows × "
        f"{len(df.columns)} columns"
    )

    logger.info(
        "NSL-KDD categorical columns: "
        f"{required_categorical}"
    )

    logger.info(
        f"NSL-KDD label count: "
        f"{df['Label'].nunique()}"
    )

    logger.info(
        "NSL-KDD sample label distribution:"
    )

    logger.info(
        f"{df['Label'].value_counts().head(15).to_dict()}"
    )

    return df


def load_nsl_kdd(cfg):
    """
    Load NSL-KDD dataset.

    Supports:
        KDDTrain.txt
        KDDTest+.txt
        CSV equivalents

    The files can use comma, tab, semicolon,
    or pipe delimiters.
    """

    raw_dir = Path(
        cfg["data"]["raw_dir"]
    )

    nsl_dir = (
        raw_dir /
        "nsl_kdd"
    )

    if not nsl_dir.exists():

        raise FileNotFoundError(
            f"NSL-KDD directory not found:\n"
            f"{nsl_dir}"
        )

    files = sorted(
        list(
            nsl_dir.glob("*.txt")
        )
        +
        list(
            nsl_dir.glob("*.csv")
        )
    )

    if not files:

        raise FileNotFoundError(
            f"No NSL-KDD files found in:\n"
            f"{nsl_dir}"
        )

    logger.info(
        f"NSL-KDD files found: "
        f"{[p.name for p in files]}"
    )

    frames = []

    for path in files:

        df_part = (
            _read_nsl_kdd_file(
                path
            )
        )

        frames.append(
            df_part
        )

    # --------------------------------------------------------
    # Combine files
    # --------------------------------------------------------

    df = pd.concat(
        frames,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Normalize labels again after concat
    # --------------------------------------------------------

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.rstrip(".")
    )

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    if len(df) == 0:

        raise ValueError(
            "NSL-KDD loader returned zero rows."
        )

    if df["Label"].nunique() == 0:

        raise ValueError(
            "NSL-KDD loader returned zero labels."
        )

    logger.info(
        f"NSL-KDD loaded: "
        f"{df.shape}"
    )

    return df


# ============================================================
# UNSW-NB15
# ============================================================

def load_unsw_nb15(cfg):
    """
    Load UNSW-NB15.

    If attack_cat exists, it is used as the multiclass target.
    Otherwise the binary 'label' column is used.
    """

    raw_dir = Path(
        cfg["data"]["raw_dir"]
    )

    unsw_dir = raw_dir / "unsw_nb15"

    if not unsw_dir.exists():

        raise FileNotFoundError(
            f"UNSW-NB15 directory not found:\n"
            f"{unsw_dir}"
        )

    files = sorted(
        unsw_dir.glob("*.csv")
    )

    if not files:

        raise FileNotFoundError(
            f"No UNSW-NB15 CSV files found in:\n"
            f"{unsw_dir}"
        )

    frames = []

    for path in files:

        logger.info(
            f"Reading UNSW-NB15 file: "
            f"{path.name}"
        )

        frames.append(
            _read_csv_robust(path)
        )

    df = pd.concat(
        frames,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Multiclass target
    # --------------------------------------------------------

    if "attack_cat" in df.columns:

        logger.info(
            "UNSW-NB15 target selected: "
            "attack_cat "
            "(multiclass classification)"
        )

        df["Label"] = (
            df["attack_cat"]
            .fillna("Normal")
            .astype(str)
            .str.strip()
        )

        df.drop(
            columns=["attack_cat"],
            inplace=True
        )

        if "label" in df.columns:

            df.drop(
                columns=["label"],
                inplace=True
            )

    elif "label" in df.columns:

        logger.info(
            "UNSW-NB15 target selected: "
            "label "
            "(binary classification)"
        )

        df.rename(
            columns={
                "label": "Label"
            },
            inplace=True
        )

    else:

        raise ValueError(
            "UNSW-NB15 does not contain either "
            "'attack_cat' or 'label'."
        )

    logger.info(
        f"UNSW-NB15 loaded: {df.shape}"
    )

    return df


# ============================================================
# SAMPLE DATASET
# ============================================================

def load_sample_dataset(cfg):

    sample_dir = Path(
        cfg["data"]["sample_dir"]
    )

    files = list(
        sample_dir.glob("*.csv")
    )

    if not files:

        raise FileNotFoundError(
            f"No sample CSV files found in:\n"
            f"{sample_dir}"
        )

    frames = []

    for path in files:

        logger.info(
            f"Reading sample file: "
            f"{path.name}"
        )

        frames.append(
            _read_csv_robust(path)
        )

    df = pd.concat(
        frames,
        ignore_index=True
    )

    return df


# ============================================================
# DATASET SELECTOR
# ============================================================

def load_dataset(cfg):

    dataset_name = (
        cfg["data"]["dataset"]
        .lower()
        .strip()
    )

    if dataset_name == "cicids2017":

        df = _load_cicids2017_memory_safe(
            cfg
        )

        df = adapt_cicids2017(
            df
        )

    elif dataset_name == "nsl_kdd":

        df = load_nsl_kdd(
            cfg
        )

    elif dataset_name == "unsw_nb15":

        df = load_unsw_nb15(
            cfg
        )

    elif dataset_name == "sample":

        df = load_sample_dataset(
            cfg
        )

    else:

        raise ValueError(
            f"Unsupported dataset: "
            f"{dataset_name}\n"
            "Supported datasets: "
            "cicids2017, nsl_kdd, "
            "unsw_nb15, sample"
        )

    logger.info(
        f"Selected dataset: "
        f"{dataset_name}"
    )

    logger.info(
        f"Loaded dataset shape: "
        f"{df.shape}"
    )

    return df


# ============================================================
# CLEANING
# ============================================================

def clean(df, cfg):

    # Work on a copy so that the original loaded
    # DataFrame is not unexpectedly modified.
    df = df.copy()

    label_column = "Label"

    if label_column not in df.columns:

        raise ValueError(
            f"Required normalized label column "
            f"'{label_column}' not found.\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    dataset_name = (
        cfg["data"]["dataset"]
        .lower()
        .strip()
    )

    # --------------------------------------------------------
    # Dataset-specific columns
    # --------------------------------------------------------

    drop_columns = list(
        cfg["data"].get(
            "drop_columns",
            []
        )
    )

    # NSL-KDD difficulty must never be used as
    # a predictive feature.
    if dataset_name == "nsl_kdd":

        if "difficulty" in df.columns:

            drop_columns.append(
                "difficulty"
            )

    columns_to_drop = list(
        dict.fromkeys(
            [
                col
                for col in drop_columns
                if col in df.columns
                and col != label_column
            ]
        )
    )

    if columns_to_drop:

        df.drop(
            columns=columns_to_drop,
            inplace=True,
            errors="ignore"
        )

        logger.info(
            f"Dropped columns: "
            f"{columns_to_drop}"
        )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    before_duplicates = len(df)

    df.drop_duplicates(
        inplace=True
    )

    removed_duplicates = (
        before_duplicates -
        len(df)
    )

    logger.info(
        f"Removed {removed_duplicates:,} "
        f"duplicate rows"
    )

    # --------------------------------------------------------
    # Clean target
    # --------------------------------------------------------

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.rstrip(".")
    )

    # --------------------------------------------------------
    # Remove invalid/empty labels
    # --------------------------------------------------------

    valid_label_mask = (
        df["Label"].notna()
        & (df["Label"] != "")
        & (
            df["Label"].str.lower()
            != "nan"
        )
        & (
            df["Label"].str.lower()
            != "none"
        )
    )

    removed_invalid_labels = int(
        (~valid_label_mask).sum()
    )

    if removed_invalid_labels > 0:

        logger.warning(
            f"Removing "
            f"{removed_invalid_labels:,} "
            f"rows with invalid labels."
        )

    df = df[
        valid_label_mask
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if len(df) == 0:

        raise ValueError(
            "Cleaning produced ZERO rows.\n"
            "The dataset cannot be trained.\n"
            f"Dataset: {dataset_name}\n"
            f"Columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Numeric invalid values
    # --------------------------------------------------------

    numeric_columns = (
        df.select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    invalid_count = 0

    for col in numeric_columns:

        values = df[col].to_numpy(
            dtype=np.float64,
            copy=False
        )

        invalid_mask = (
            ~np.isfinite(values)
        )

        invalid_count += int(
            invalid_mask.sum()
        )

        if invalid_mask.any():

            df[col] = df[col].replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )

    # --------------------------------------------------------
    # Median imputation
    # --------------------------------------------------------

    if numeric_columns:

        missing_before = int(
            df[numeric_columns]
            .isna()
            .sum()
            .sum()
        )

        for col in numeric_columns:

            if df[col].isna().any():

                median_value = (
                    df[col].median()
                )

                if pd.isna(
                    median_value
                ):
                    median_value = 0.0

                df[col] = (
                    df[col]
                    .fillna(
                        median_value
                    )
                )

        missing_after = int(
            df[numeric_columns]
            .isna()
            .sum()
            .sum()
        )

    else:

        missing_before = 0
        missing_after = 0

    imputed_count = (
        missing_before -
        missing_after
    )

    logger.info(
        f"Invalid numeric values detected: "
        f"{invalid_count:,}"
    )

    logger.info(
        f"Imputed {imputed_count:,} "
        f"missing/invalid numeric values"
    )

    logger.info(
        f"Rows after cleaning: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# ENCODING
# ============================================================

def encode(df, cfg):

    df = df.copy()

    label_column = "Label"

    if label_column not in df.columns:

        raise ValueError(
            "Normalized Label column "
            "not found during encoding."
        )

    # --------------------------------------------------------
    # Timestamp handling
    # --------------------------------------------------------

    timestamp_series = None

    timestamp_candidates = [
        "Timestamp",
        "timestamp",
        "Time",
        "time"
    ]

    for col in timestamp_candidates:

        if col in df.columns:

            timestamp_series = (
                df[col]
                .astype(str)
                .reset_index(
                    drop=True
                )
            )

            df.drop(
                columns=[col],
                inplace=True
            )

            logger.info(
                f"Using '{col}' as "
                "temporal ordering column."
            )

            break

    # --------------------------------------------------------
    # Datasets without timestamps
    # --------------------------------------------------------

    if timestamp_series is None:

        timestamp_series = pd.Series(
            np.arange(
                len(df)
            ),
            name="sequence_order"
        )

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    y_encoder = LabelEncoder()

    y = y_encoder.fit_transform(
        df[label_column]
        .astype(str)
    )

    logger.info(
        f"Detected classes: "
        f"{list(y_encoder.classes_)}"
    )

    # --------------------------------------------------------
    # Remove target
    # --------------------------------------------------------

    feature_df = df.drop(
        columns=[
            label_column
        ]
    )

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    categorical_columns = (
        feature_df
        .select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    if categorical_columns:

        logger.info(
            f"Encoding categorical columns: "
            f"{categorical_columns}"
        )

        feature_df = pd.get_dummies(
            feature_df,
            columns=categorical_columns,
            dummy_na=True
        )

    # --------------------------------------------------------
    # Convert remaining features to numeric
    # --------------------------------------------------------

    for col in feature_df.columns:

        if not pd.api.types.is_numeric_dtype(
            feature_df[col]
        ):

            feature_df[col] = pd.to_numeric(
                feature_df[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Replace infinities
    # --------------------------------------------------------

    feature_df = feature_df.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    # --------------------------------------------------------
    # Fill remaining NaN
    # --------------------------------------------------------

    for col in feature_df.columns:

        if feature_df[col].isna().any():

            median_value = (
                feature_df[col].median()
            )

            if pd.isna(
                median_value
            ):
                median_value = 0.0

            feature_df[col] = (
                feature_df[col]
                .fillna(
                    median_value
                )
            )

    # --------------------------------------------------------
    # Float32
    # --------------------------------------------------------

    feature_df = feature_df.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if len(feature_df) == 0:

        raise ValueError(
            "Encoding produced zero feature rows."
        )

    if len(feature_df) != len(y):

        raise ValueError(
            "Feature/label length mismatch:\n"
            f"features={len(feature_df)}\n"
            f"labels={len(y)}"
        )

    logger.info(
        f"Number of final features: "
        f"{feature_df.shape[1]}"
    )

    return (
        feature_df,
        np.asarray(y),
        timestamp_series,
        y_encoder
    )


# ============================================================
# SAFE STRATIFICATION
# ============================================================

def _get_safe_stratify_target(
    y,
    encoder,
    split_name
):

    class_counts = (
        pd.Series(y)
        .value_counts()
    )

    logger.info(
        f"Class distribution before "
        f"{split_name} split:"
    )

    for class_id, count in (
        class_counts
        .sort_index()
        .items()
    ):

        try:

            class_name = (
                encoder
                .inverse_transform(
                    [int(class_id)]
                )[0]
            )

        except Exception:

            class_name = str(
                class_id
            )

        logger.info(
            f"  Class {class_id} "
            f"({class_name}): "
            f"{count:,}"
        )

    if class_counts.empty:

        return None

    if class_counts.min() < 2:

        rare_ids = (
            class_counts[
                class_counts < 2
            ]
            .index
            .tolist()
        )

        rare_names = []

        for class_id in rare_ids:

            try:

                name = (
                    encoder
                    .inverse_transform(
                        [int(class_id)]
                    )[0]
                )

            except Exception:

                name = str(
                    class_id
                )

            rare_names.append(
                name
            )

        logger.warning(
            f"Stratified {split_name} "
            f"split disabled because "
            f"these classes have <2 "
            f"samples: {rare_names}"
        )

        return None

    return y


# ============================================================
# SCALER
# ============================================================

def create_scaler(cfg):

    scaler_name = (
        cfg["preprocessing"]
        .get(
            "scaler",
            "standard"
        )
        .lower()
    )

    if scaler_name == "minmax":

        return MinMaxScaler()

    return StandardScaler()


# ============================================================
# SPLIT + SCALE
# ============================================================

def split_scale(
    feature_df,
    y,
    timestamps,
    encoder,
    cfg
):

    if len(feature_df) == 0:

        raise ValueError(
            "Cannot split an empty dataset."
        )

    if len(feature_df) != len(y):

        raise ValueError(
            "Feature/label length mismatch "
            "before train/test split:\n"
            f"features={len(feature_df)}\n"
            f"labels={len(y)}"
        )

    test_size = float(
        cfg["data"].get(
            "test_size",
            0.20
        )
    )

    val_size = float(
        cfg["data"].get(
            "val_size",
            0.15
        )
    )

    seed = int(
        cfg["project"].get(
            "seed",
            42
        )
    )

    feature_df = (
        feature_df
        .reset_index(
            drop=True
        )
    )

    y = np.asarray(y)

    ts_series = (
        pd.Series(
            timestamps
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Test split
    # --------------------------------------------------------

    stratify_target = (
        _get_safe_stratify_target(
            y,
            encoder,
            "test"
        )
    )

    (
        X_temp,
        X_test,
        y_temp,
        y_test,
        ts_temp,
        ts_test
    ) = train_test_split(
        feature_df,
        y,
        ts_series,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_target
    )

    # --------------------------------------------------------
    # Validation split
    # --------------------------------------------------------

    val_frac = (
        val_size /
        (1.0 - test_size)
    )

    val_frac = min(
        max(
            val_frac,
            0.01
        ),
        0.90
    )

    temp_stratify = (
        _get_safe_stratify_target(
            y_temp,
            encoder,
            "validation"
        )
    )

    (
        X_train,
        X_val,
        y_train,
        y_val,
        ts_train,
        ts_val
    ) = train_test_split(
        X_temp,
        y_temp,
        ts_temp,
        test_size=val_frac,
        random_state=seed,
        stratify=temp_stratify
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    scaler = create_scaler(
        cfg
    )

    logger.info(
        "Fitting scaler on training data..."
    )

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_val_scaled = (
        scaler.transform(
            X_val
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    X_train_scaled = np.asarray(
        X_train_scaled,
        dtype=np.float32
    )

    X_val_scaled = np.asarray(
        X_val_scaled,
        dtype=np.float32
    )

    X_test_scaled = np.asarray(
        X_test_scaled,
        dtype=np.float32
    )

    logger.info(
        f"Train shape: "
        f"{X_train_scaled.shape}"
    )

    logger.info(
        f"Validation shape: "
        f"{X_val_scaled.shape}"
    )

    logger.info(
        f"Test shape: "
        f"{X_test_scaled.shape}"
    )

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,

        "y_train": np.asarray(
            y_train
        ),

        "y_val": np.asarray(
            y_val
        ),

        "y_test": np.asarray(
            y_test
        ),

        "timestamps_train": ts_train,
        "timestamps_val": ts_val,
        "timestamps_test": ts_test,

        "scaler": scaler
    }


# ============================================================
# SAFE SMOTE-TOMEK
# ============================================================

def apply_smote_tomek(
    X_train,
    y_train,
    cfg
):

    preprocessing_cfg = (
        cfg["preprocessing"]
    )

    enabled = preprocessing_cfg.get(
        "apply_smote_tomek",
        False
    )

    if not enabled:

        logger.info(
            "SMOTE-Tomek disabled."
        )

        return X_train, y_train

    max_rows = int(
        preprocessing_cfg.get(
            "smote_tomek_max_rows",
            8000
        )
    )

    min_class_rows = int(
        preprocessing_cfg.get(
            "smote_tomek_min_class_rows",
            50
        )
    )

    # --------------------------------------------------------
    # Cap input
    # --------------------------------------------------------

    if len(X_train) > max_rows:

        rng = np.random.RandomState(
            cfg["project"].get(
                "seed",
                42
            )
        )

        indices = rng.choice(
            len(X_train),
            size=max_rows,
            replace=False
        )

        X_work = X_train[
            indices
        ]

        y_work = y_train[
            indices
        ]

        logger.info(
            f"SMOTE-Tomek input capped at "
            f"{max_rows:,} rows."
        )

    else:

        X_work = X_train
        y_work = y_train

    class_counts = (
        pd.Series(y_work)
        .value_counts()
    )

    safe_classes = (
        class_counts[
            class_counts >= min_class_rows
        ]
        .index
    )

    rare_classes = (
        class_counts[
            class_counts < min_class_rows
        ]
        .index
    )

    logger.info(
        f"SMOTE-safe classes: "
        f"{list(safe_classes)}"
    )

    logger.info(
        f"Rare classes kept unchanged: "
        f"{list(rare_classes)}"
    )

    if len(safe_classes) < 2:

        logger.warning(
            "Not enough classes for "
            "SMOTE-Tomek. "
            "Original training data retained."
        )

        return X_train, y_train

    safe_mask = np.isin(
        y_work,
        safe_classes
    )

    rare_mask = np.isin(
        y_work,
        rare_classes
    )

    X_safe = X_work[
        safe_mask
    ]

    y_safe = y_work[
        safe_mask
    ]

    X_rare = X_work[
        rare_mask
    ]

    y_rare = y_work[
        rare_mask
    ]

    try:

        smote_tomek = SMOTETomek(
            sampling_strategy=(
                preprocessing_cfg.get(
                    "smote_tomek_sampling_strategy",
                    "auto"
                )
            ),
            random_state=(
                cfg["project"].get(
                    "seed",
                    42
                )
            )
        )

        logger.info(
            "Applying SMOTE-Tomek..."
        )

        (
            X_resampled,
            y_resampled
        ) = smote_tomek.fit_resample(
            X_safe,
            y_safe
        )

        if len(X_rare) > 0:

            X_final = np.concatenate(
                [
                    X_resampled,
                    X_rare
                ],
                axis=0
            )

            y_final = np.concatenate(
                [
                    y_resampled,
                    y_rare
                ],
                axis=0
            )

        else:

            X_final = X_resampled
            y_final = y_resampled

        logger.info(
            f"Training shape after "
            f"SMOTE-Tomek: "
            f"{X_final.shape}"
        )

        return (
            np.asarray(
                X_final,
                dtype=np.float32
            ),
            np.asarray(
                y_final
            )
        )

    except Exception as exc:

        logger.warning(
            f"SMOTE-Tomek failed: {exc}"
        )

        logger.warning(
            "Continuing with original "
            "training data."
        )

        return X_train, y_train


# ============================================================
# SEQUENCE DATASET
# ============================================================

def create_sequence_dataset(
    feature_df,
    timestamps,
    y,
    cfg,
    output_path
):

    sequence_cfg = cfg.get(
        "sequence",
        {}
    )

    window_size = int(
        sequence_cfg.get(
            "window_size",
            10
        )
    )

    stride = int(
        sequence_cfg.get(
            "stride",
            1
        )
    )

    ordered = (
        feature_df
        .reset_index(
            drop=True
        )
        .copy()
    )

    y_array = np.asarray(
        y
    ).reshape(-1)

    if len(ordered) != len(y_array):

        raise ValueError(
            "Sequence dataset error: "
            f"features={len(ordered)}, "
            f"labels={len(y_array)}"
        )

    # --------------------------------------------------------
    # Timestamp / sequence order
    # --------------------------------------------------------

    if timestamps is None:

        ordered.insert(
            0,
            "sequence_order",
            np.arange(
                len(ordered)
            )
        )

    else:

        timestamps_series = (
            pd.Series(
                timestamps
            )
            .reset_index(
                drop=True
            )
        )

        if len(
            timestamps_series
        ) != len(ordered):

            raise ValueError(
                "Sequence dataset error: "
                "feature rows and timestamps "
                "have different lengths."
            )

        ordered.insert(
            0,
            "Timestamp",
            timestamps_series
        )

    # --------------------------------------------------------
    # Add label BEFORE sorting
    # --------------------------------------------------------

    ordered["Label"] = y_array

    # --------------------------------------------------------
    # Timestamp sorting
    # --------------------------------------------------------

    if timestamps is not None:

        try:

            ordered[
                "_parsed_timestamp"
            ] = pd.to_datetime(
                ordered["Timestamp"],
                errors="coerce"
            )

            ordered.sort_values(
                "_parsed_timestamp",
                kind="stable",
                inplace=True
            )

            ordered.drop(
                columns=[
                    "_parsed_timestamp"
                ],
                inplace=True
            )

        except Exception:

            ordered.sort_values(
                "Timestamp",
                kind="stable",
                inplace=True
            )

        ordered.reset_index(
            drop=True,
            inplace=True
        )

    # --------------------------------------------------------
    # Label validation
    # --------------------------------------------------------

    ordered["Label"] = (
        pd.to_numeric(
            ordered["Label"],
            errors="raise"
        )
        .astype(np.int64)
    )

    ordered.attrs[
        "window_size"
    ] = window_size

    ordered.attrs[
        "stride"
    ] = stride

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    try:

        ordered.to_parquet(
            output_path,
            index=False
        )

        logger.info(
            f"Sequence dataset saved: "
            f"{output_path}"
        )

        logger.info(
            f"Sequence dataset shape: "
            f"{ordered.shape}"
        )

    except Exception as exc:

        logger.warning(
            f"Could not save Parquet: "
            f"{exc}"
        )

        csv_path = str(
            Path(output_path)
            .with_suffix(".csv")
        )

        ordered.to_csv(
            csv_path,
            index=False
        )

        logger.info(
            f"Sequence CSV saved: "
            f"{csv_path}"
        )

    return ordered


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(cfg):

    dataset_name = (
        cfg["data"]["dataset"]
        .lower()
        .strip()
    )

    logger.info("=" * 60)

    logger.info(
        f"STARTING PREPROCESSING: "
        f"{dataset_name.upper()}"
    )

    logger.info("=" * 60)

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    processed_dir = Path(
        cfg["data"]["processed_dir"]
    )

    ensure_directory(
        processed_dir
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_dataset(
        cfg
    )

    if len(df) == 0:

        raise ValueError(
            f"Dataset '{dataset_name}' "
            "loaded ZERO rows."
        )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df = clean(
        df,
        cfg
    )

    logger.info(
        f"Cleaned dataset shape: "
        f"{df.shape}"
    )

    if len(df) == 0:

        raise ValueError(
            f"Dataset '{dataset_name}' "
            "contains zero rows after cleaning."
        )

    # --------------------------------------------------------
    # ENCODE
    # --------------------------------------------------------

    (
        feature_df,
        y,
        timestamps,
        y_encoder
    ) = encode(
        df,
        cfg
    )

    # --------------------------------------------------------
    # SPLIT + SCALE
    # --------------------------------------------------------

    split_data = split_scale(
        feature_df,
        y,
        timestamps,
        y_encoder,
        cfg
    )

    # --------------------------------------------------------
    # SMOTE-TOMEK
    # --------------------------------------------------------

    (
        X_train_resampled,
        y_train_resampled
    ) = apply_smote_tomek(
        split_data["X_train"],
        split_data["y_train"],
        cfg
    )

    # --------------------------------------------------------
    # FINAL DATA OBJECT
    # --------------------------------------------------------

    data = {

        "X_train":
            X_train_resampled,

        "X_val":
            split_data["X_val"],

        "X_test":
            split_data["X_test"],

        "y_train":
            y_train_resampled,

        "y_val":
            split_data["y_val"],

        "y_test":
            split_data["y_test"],

        "timestamps_train":
            split_data["timestamps_train"],

        "timestamps_val":
            split_data["timestamps_val"],

        "timestamps_test":
            split_data["timestamps_test"],

        "scaler":
            split_data["scaler"],

        "label_encoder":
            y_encoder,

        "feature_names":
            list(feature_df.columns),

        "dataset":
            dataset_name
    }

    # --------------------------------------------------------
    # SAVE PROCESSED DATA
    # --------------------------------------------------------

    joblib.dump(
        data,
        processed_dir /
        "processed_data.joblib",
        compress=3
    )

    joblib.dump(
        split_data["scaler"],
        processed_dir /
        "scaler.joblib"
    )

    joblib.dump(
        y_encoder,
        processed_dir /
        "label_encoder.joblib"
    )

    joblib.dump(
        list(feature_df.columns),
        processed_dir /
        "feature_names.joblib"
    )

    # --------------------------------------------------------
    # SEQUENCE DATASET
    # --------------------------------------------------------

    sequence_path = (
        processed_dir /
        "sequence_ordered.parquet"
    )

    create_sequence_dataset(
        feature_df,
        timestamps,
        y,
        cfg,
        sequence_path
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    logger.info("=" * 60)

    logger.info(
        "PREPROCESSING COMPLETED"
    )

    logger.info(
        f"Dataset: {dataset_name}"
    )

    logger.info(
        f"Train: "
        f"{data['X_train'].shape}"
    )

    logger.info(
        f"Validation: "
        f"{data['X_val'].shape}"
    )

    logger.info(
        f"Test: "
        f"{data['X_test'].shape}"
    )

    logger.info(
        f"Features: "
        f"{len(data['feature_names'])}"
    )

    logger.info(
        f"Classes: "
        f"{list(y_encoder.classes_)}"
    )

    logger.info("=" * 60)

    return ProcessedData(
        **data
    )


# ============================================================
# CONFIG LOADER
# ============================================================

def main():

    from src.utils.config import load_config

    cfg = load_config()

    run_pipeline(
        cfg
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
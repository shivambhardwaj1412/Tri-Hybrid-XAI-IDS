"""
Turns chronologically-ordered flows into fixed-length sequences for the
DL temporal branch (Stage 2, Branch B).

  flows (ordered by Timestamp) -> sliding window of `window_size` flows
  -> one sequence sample. Label = label of the LAST flow in the window
  (the model is asked: "given the last N flows, what's happening now,
  in temporal context?").

Sequences are built independently within each split (train/val/test) using
the chronological order already established at the preprocessing stage, so
a window never crosses the train/test boundary (no leakage).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_sequences(X: np.ndarray, y: np.ndarray, window_size: int, stride: int = 1):
    """X: (n_samples, n_features) ordered chronologically. Returns
    (n_windows, window_size, n_features) and (n_windows,) labels."""
    n = X.shape[0]
    if n < window_size:
        raise ValueError(f"Not enough samples ({n}) for window_size={window_size}")
    starts = range(0, n - window_size + 1, stride)
    seqs = np.stack([X[s:s + window_size] for s in starts])
    labels = np.array([y[s + window_size - 1] for s in starts])
    return seqs, labels


def build_dl_datasets(seq_df: pd.DataFrame, feature_names: list, label_col: str,
                       scaler, window_size: int, stride: int,
                       test_size: float, val_size: float, seed: int):
    """Builds chronologically-split, windowed sequence datasets straight from
    the timestamp-ordered flow table. Uses per-class splitting (same approach
    as train_meta.chronological_split) to guarantee every class appears in
    every split even on multi-day concatenated datasets like CICIDS2017."""
    df = seq_df.sort_values("Timestamp").reset_index(drop=True)

    # per-class chronological split to avoid class-disappearing on multi-day datasets
    train_parts, val_parts, test_parts = [], [], []
    for _, group in df.groupby(label_col, sort=False):
        group = group.sort_values("Timestamp")
        n = len(group)
        n_test = max(1, int(n * test_size)) if n >= 3 else 0
        n_val = max(1, int(n * val_size)) if n - n_test >= 2 else 0
        n_train = n - n_test - n_val
        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])

    def _concat_sorted(parts):
        return pd.concat(parts).sort_values("Timestamp").reset_index(drop=True)

    train_df = _concat_sorted(train_parts)
    val_df = _concat_sorted(val_parts)
    test_df = _concat_sorted(test_parts)

    def _to_seq(split_df):
        X = scaler.transform(split_df[feature_names].values)
        y = split_df[label_col].values
        return make_sequences(X, y, window_size, stride)

    return _to_seq(train_df), _to_seq(val_df), _to_seq(test_df)

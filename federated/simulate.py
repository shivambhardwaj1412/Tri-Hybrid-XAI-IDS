"""
Stage 6 (optional) — Simulated Federated Learning Demonstration.

IMPORTANT: this is explicitly a SIMULATION. All "nodes" run inside this one
process on locally-partitioned shards of the same machine's data; there is
no real network transport between separate machines. What IS real:

  - Each node's local data shard is never pooled into one global dataset.
  - Only model WEIGHTS (a list of numpy arrays) cross the "node -> server"
    boundary, exactly as FedAvg (McMahan et al., 2017) specifies.
  - The central server never sees a single raw traffic row.

Run: `python -m federated.simulate` (after `python train_all.py`, since it
reuses the trained DL architecture / preprocessing artifacts).
"""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import keras

from src.utils.config import load_config
from src.preprocessing.pipeline import ProcessedData  # noqa: needed for unpickling
from src.dl.model import build_model, AdditiveAttention  # noqa: registers custom layer
from src.dl.sequence import make_sequences
from src.fusion.train_meta import chronological_split
from sklearn.metrics import accuracy_score


def partition_iid(train_df: pd.DataFrame, num_nodes: int, seed: int):
    """Splits the chronological TRAIN split into `num_nodes` disjoint,
    non-overlapping shards (simulating traffic logs siloed at different
    organizations) -- never merged."""
    idx = np.arange(len(train_df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    shards = np.array_split(idx, num_nodes)
    return [train_df.iloc[np.sort(s)].reset_index(drop=True) for s in shards]


def fedavg(weight_list: list[list[np.ndarray]], n_samples_list: list[int]) -> list[np.ndarray]:
    """Weighted average of model weights across nodes (weighted by each
    node's local sample count, as in the original FedAvg paper)."""
    total = sum(n_samples_list)
    avg = [np.zeros_like(w) for w in weight_list[0]]
    for weights, n in zip(weight_list, n_samples_list):
        frac = n / total
        for i, w in enumerate(weights):
            avg[i] += w * frac
    return avg


def run_federated_simulation(cfg: dict, results_dir="results"):
    fed_cfg = cfg["federated"]
    seq_df = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "sequence_ordered.parquet")
    data = joblib.load(Path(cfg["data"]["processed_dir"]) / "processed_data.joblib")
    train_df, val_df, test_df = chronological_split(seq_df, cfg["data"]["test_size"], cfg["data"]["val_size"])

    n_classes = len(data.label_encoder.classes_)
    ws = cfg["sequence"]["window_size"]
    n_features = len(data.feature_names)

    node_shards = partition_iid(train_df, fed_cfg["num_nodes"], cfg["project"]["seed"])
    print(f"Partitioned TRAIN split into {fed_cfg['num_nodes']} disjoint local shards "
          f"(sizes: {[len(s) for s in node_shards]}) -- no raw data pooled.")

    # global model initialization (shared architecture, random init, sent to nodes)
    global_model = build_model(ws, n_features, n_classes, cfg)
    global_weights = global_model.get_weights()

    history = []
    for rnd in range(fed_cfg["rounds"]):
        local_weights, local_n = [], []
        for node_id, shard in enumerate(node_shards):
            X = shard[data.feature_names].values
            y = shard["Label"].values
            try:
                X_seq, y_seq = make_sequences(X, y, ws, stride=1)
            except ValueError:
                continue  # shard too small for one window this round -- skip, don't fabricate

            local_model = build_model(ws, n_features, n_classes, cfg)
            local_model.set_weights(global_weights)  # start from current global weights
            local_model.fit(X_seq, y_seq, epochs=fed_cfg["local_epochs"], batch_size=32, verbose=0)

            local_weights.append(local_model.get_weights())  # ONLY weights leave the node
            local_n.append(len(X_seq))

        if not local_weights:
            print(f"Round {rnd + 1}: no node had enough data for a window; skipping round.")
            continue

        global_weights = fedavg(local_weights, local_n)  # server-side aggregation, no raw data involved
        global_model.set_weights(global_weights)

        # evaluate the newly aggregated global model on the (never-touched-by-any-node) test split
        X_test = test_df[data.feature_names].values
        y_test = test_df["Label"].values
        X_test_seq, y_test_seq = make_sequences(X_test, y_test, ws, stride=1)
        proba, _ = global_model.predict(X_test_seq, verbose=0)
        acc = accuracy_score(y_test_seq, proba.argmax(axis=1))
        history.append({"round": rnd + 1, "global_test_accuracy": float(acc),
                         "participating_nodes": len(local_weights)})
        print(f"Round {rnd + 1}/{fed_cfg['rounds']}: global test accuracy = {acc:.4f} "
              f"({len(local_weights)} nodes participated)")

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(results_dir) / "federated_simulation_history.json", "w") as f:
        json.dump({
            "note": "SIMULATED federated learning: all nodes run in one process on disjoint "
                    "local data shards; only model weights are aggregated (FedAvg), never raw traffic.",
            "num_nodes": fed_cfg["num_nodes"], "rounds": fed_cfg["rounds"],
            "local_epochs": fed_cfg["local_epochs"], "history": history,
        }, f, indent=2)

    Path("models/dl").mkdir(parents=True, exist_ok=True)
    global_model.save("models/dl/dl_model_federated.keras")
    return history


if __name__ == "__main__":
    cfg = load_config()
    run_federated_simulation(cfg)

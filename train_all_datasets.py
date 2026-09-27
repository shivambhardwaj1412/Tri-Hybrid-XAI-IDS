"""
train_all_datasets.py — runs the full Tri-Hybrid XAI-IDS pipeline on ALL
three real benchmark datasets (NSL-KDD, UNSW-NB15, CICIDS2017) back-to-back.

For each dataset:
  - Runs the complete Stage 1 → Stage 3 pipeline
  - Saves models  -> models_by_dataset/<name>/
  - Saves results -> results_by_dataset/<name>/

After all three finish, CICIDS2017 models and results are copied into the
live models/ and results/ folders so the dashboard uses the best dataset.

Usage:
    python train_all_datasets.py

To run only specific datasets:
    python train_all_datasets.py --datasets cicids2017 nsl_kdd
"""
import argparse
import json
import shutil
import time
import yaml
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, ConfusionMatrixDisplay)

from src.utils.config import load_config, subsample_stratified
from src.preprocessing.pipeline import run_pipeline, ProcessedData  # noqa
from src.ml.shap_selection import select_features
from src.ml.ensemble import train_ensemble, save_ensemble
from src.fusion.train_meta import run_fusion

PROJECT_ROOT = Path(__file__).resolve().parent

# Dashboard will use this dataset's models/results after all runs finish
DASHBOARD_DATASET = "cicids2017"

ALL_DATASETS = ["nsl_kdd", "unsw_nb15", "cicids2017"]


def _patch_config(cfg: dict, dataset: str) -> dict:
    """Return a copy of cfg with dataset switched and paths redirected."""
    import copy
    c = copy.deepcopy(cfg)
    c["data"]["dataset"] = dataset
    c["data"]["processed_dir"] = f"data/processed_{dataset}"
    return c


def _save_artifacts(dataset: str, results_dir: Path, models_dir: Path):
    """Copy results/ and models/ into per-dataset archive folders."""
    dst_results = PROJECT_ROOT / "results_by_dataset" / dataset
    dst_models  = PROJECT_ROOT / "models_by_dataset"  / dataset
    if dst_results.exists():
        shutil.rmtree(dst_results)
    if dst_models.exists():
        shutil.rmtree(dst_models)
    shutil.copytree(results_dir, dst_results)
    shutil.copytree(models_dir,  dst_models)
    print(f"  Archived -> results_by_dataset/{dataset}/  models_by_dataset/{dataset}/")


def _activate_dataset(dataset: str):
    """Copy a dataset's archived results/models into the live folders."""
    src_results = PROJECT_ROOT / "results_by_dataset" / dataset
    src_models  = PROJECT_ROOT / "models_by_dataset"  / dataset
    live_results = PROJECT_ROOT / "results"
    live_models  = PROJECT_ROOT / "models"
    if live_results.exists():
        shutil.rmtree(live_results)
    if live_models.exists():
        shutil.rmtree(live_models)
    shutil.copytree(src_results, live_results)
    shutil.copytree(src_models,  live_models)
    print(f"\n  Dashboard activated: {dataset} models/results are now live.")


def run_one_dataset(dataset: str, base_cfg: dict) -> dict:
    """Run the full pipeline for one dataset. Returns summary metrics dict."""
    print("\n" + "=" * 70)
    print(f"  DATASET: {dataset.upper()}")
    print("=" * 70)
    t0 = time.time()

    cfg = _patch_config(base_cfg, dataset)

    results_dir = PROJECT_ROOT / "results"
    models_dir  = PROJECT_ROOT / "models"

    # ── Stage 1: Preprocessing ──────────────────────────────────────────────
    print(f"\n[1/3] Preprocessing...")
    data = run_pipeline(cfg)
    print(f"      Train {data.X_train.shape}  Val {data.X_val.shape}  "
          f"Test {data.X_test.shape}  Classes: {len(data.label_encoder.classes_)}")

    # ── Stage 2a: SHAP feature selection + ML ensemble ──────────────────────
    print(f"\n[2/3] Branch A — SHAP feature selection + ML ensemble...")
    max_n = cfg["ml_branch"].get("max_training_samples")
    X_tr, y_tr = subsample_stratified(data.X_train, data.y_train, max_n, cfg["project"]["seed"])

    selected_features, _ = select_features(
        X_tr, y_tr, cfg,
        results_dir=str(results_dir / "shap")
    )
    print(f"      Selected {len(selected_features)} / {len(data.feature_names)} features via SHAP")

    n_classes = len(data.label_encoder.classes_)
    ml_model = train_ensemble(X_tr[selected_features].values, y_tr, cfg, n_classes)
    ml_preds = ml_model.predict(data.X_test[selected_features].values)
    ml_iid_acc = float(accuracy_score(data.y_test, ml_preds))
    ml_iid_f1  = float(f1_score(data.y_test, ml_preds, average="macro"))
    print(f"      Branch A (i.i.d.) accuracy: {ml_iid_acc:.4f}  macro-F1: {ml_iid_f1:.4f}")

    save_ensemble(ml_model, str(models_dir / "ml/ml_ensemble.joblib"))
    (models_dir / "ml").mkdir(parents=True, exist_ok=True)
    with open(models_dir / "ml/selected_features.json", "w") as f:
        json.dump(selected_features, f, indent=2)

    # ── Stage 2b + 3: DL branch + meta-learner fusion ───────────────────────
    print(f"\n[3/3] Branch B (DL) + Meta-learner fusion...")
    seq_df = pd.read_parquet(
        PROJECT_ROOT / cfg["data"]["processed_dir"] / "sequence_ordered.parquet"
    )
    fusion = run_fusion(cfg, seq_df, data.feature_names, selected_features,
                        data.scaler, n_classes, results_dir=str(results_dir))
    m = fusion["metrics"]
    print(f"      ML-only: {m['branch_a_ml_only_acc']:.4f}  "
          f"DL-only: {m['branch_b_dl_only_acc']:.4f}  "
          f"Fused: {m['fused_meta_acc']:.4f}  "
          f"macro-F1: {m['fused_meta_f1_macro']:.4f}")

    # ── Artifacts ────────────────────────────────────────────────────────────
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)
    (results_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)

    # metrics.csv
    pd.DataFrame([
        {"model": "Branch A only (ML, chronological split)",
         "accuracy": m["branch_a_ml_only_acc"]},
        {"model": "Branch B only (DL, chronological split)",
         "accuracy": m["branch_b_dl_only_acc"]},
        {"model": "Proposed Tri-Hybrid (fused)",
         "accuracy": m["fused_meta_acc"],
         "macro_f1": m["fused_meta_f1_macro"]},
        {"model": "Branch A only (i.i.d. split)",
         "accuracy": ml_iid_acc, "macro_f1": ml_iid_f1},
    ]).to_csv(results_dir / "metrics.csv", index=False)

    # confusion matrix
    cm = confusion_matrix(fusion["y_test_aligned"], fusion["final_preds"],
                          labels=np.arange(n_classes))
    disp = ConfusionMatrixDisplay(cm, display_labels=data.label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(max(6, n_classes * 0.5), max(6, n_classes * 0.5)))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=90)
    plt.title(f"Confusion Matrix — Fused ({dataset.upper()})")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrices/fused_test_confusion_matrix.png", dpi=140)
    plt.close()

    # branch comparison bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    bar_labels = ["Branch A\n(ML only)", "Branch B\n(DL only)", "Proposed\n(Fused)"]
    accs = [m["branch_a_ml_only_acc"], m["branch_b_dl_only_acc"], m["fused_meta_acc"]]
    bars = ax.bar(bar_labels, accs, color=["#4C72B0", "#DD8452", "#55A868"])
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.005, f"{a:.3f}", ha="center")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test Accuracy (chronological split)")
    ax.set_title(f"Branch A vs B vs Fused — {dataset.upper()}")
    plt.tight_layout()
    plt.savefig(results_dir / "plots/branch_comparison.png", dpi=140)
    plt.close()

    runtime = round(time.time() - t0, 1)
    summary = {
        "dataset_used": dataset,
        "selected_ml_features": selected_features,
        "n_classes": n_classes,
        "classes": list(data.label_encoder.classes_),
        **m,
        "branch_a_iid_split_accuracy": ml_iid_acc,
        "branch_a_iid_split_macro_f1": ml_iid_f1,
        "runtime_seconds": runtime,
    }
    with open(results_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Done in {runtime}s")
    _save_artifacts(dataset, results_dir, models_dir)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        choices=ALL_DATASETS,
                        help="Which datasets to train (default: all three)")
    args = parser.parse_args()

    base_cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    all_summaries = {}
    overall_start = time.time()

    for dataset in args.datasets:
        try:
            summary = run_one_dataset(dataset, base_cfg)
            all_summaries[dataset] = summary
        except Exception as e:
            print(f"\n  ERROR on {dataset}: {e}")
            import traceback; traceback.print_exc()
            print(f"  Skipping {dataset} and continuing...\n")

    # ── Print cross-dataset comparison table ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  CROSS-DATASET RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Dataset':<15} {'Classes':>7} {'ML-only':>9} {'DL-only':>9} {'Fused':>9} {'macro-F1':>10} {'Time(s)':>8}")
    print("  " + "-" * 65)
    for ds, s in all_summaries.items():
        print(f"  {ds:<15} {s['n_classes']:>7} "
              f"{s['branch_a_ml_only_acc']:>9.4f} "
              f"{s['branch_b_dl_only_acc']:>9.4f} "
              f"{s['fused_meta_acc']:>9.4f} "
              f"{s['fused_meta_f1_macro']:>10.4f} "
              f"{s['runtime_seconds']:>8.1f}")

    # ── Activate dashboard dataset ────────────────────────────────────────────
    dashboard_ds = DASHBOARD_DATASET if DASHBOARD_DATASET in all_summaries else list(all_summaries.keys())[-1]
    _activate_dataset(dashboard_ds)

    # Update config.yaml to reflect the active dataset
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg_text = f.read()
    import re
    cfg_text = re.sub(r'(dataset:\s*")[^"]*(")', rf'\g<1>{dashboard_ds}\g<2>', cfg_text)
    with open(cfg_path, "w") as f:
        f.write(cfg_text)
    print(f"  config.yaml updated: dataset = \"{dashboard_ds}\"")

    print(f"\n  Total time: {round(time.time() - overall_start, 1)}s")
    print("  Run 'streamlit run dashboard/app.py' to launch the dashboard.")
    print("=" * 70)


if __name__ == "__main__":
    main()

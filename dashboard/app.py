"""
Dataset-aware Tri-Hybrid XAI-IDS Streamlit dashboard.

Run:
    python -m streamlit run dashboard/app.py

Supports:
    CICIDS2017 -> data/processed_cicids2017, models/cicids2017, results/cicids2017
    NSL-KDD    -> data/processed_nsl_kdd,    models/nsl_kdd,    results/nsl_kdd
    UNSW-NB15  -> data/processed_unsw_nb15,  models/unsw_nb15,  results/unsw_nb15
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.preprocessing.pipeline import ProcessedData  # noqa: F401
from src.fusion.train_meta import chronological_split
from dashboard.export_report import ALL_SECTIONS, generate_pdf, generate_png_single

st.set_page_config(
    page_title="Tri-Hybrid XAI-IDS — Analyst Dashboard",
    layout="wide",
    page_icon="🛡️",
)

BASE_CFG = load_config(str(PROJECT_ROOT / "config.yaml"))

DATASETS = {
    "CICIDS2017": "cicids2017",
    "NSL-KDD": "nsl_kdd",
    "UNSW-NB15": "unsw_nb15",
}


def make_cfg(dataset):
    cfg = copy.deepcopy(BASE_CFG)
    cfg.setdefault("data", {})
    cfg.setdefault("paths", {})
    cfg["data"]["dataset"] = dataset
    cfg["data"]["processed_dir"] = f"data/processed_{dataset}"
    cfg["paths"]["models_dir"] = "models"
    cfg["paths"]["results_dir"] = "results"
    return cfg


def dirs(dataset):
    return {
        "processed": PROJECT_ROOT / f"data/processed_{dataset}",
        "models": PROJECT_ROOT / f"models/{dataset}",
        "results": PROJECT_ROOT / f"results/{dataset}",
    }


def artifact_status(dataset):
    d = dirs(dataset)
    missing = []

    if not (d["processed"] / "sequence_ordered.parquet").exists():
        missing.append(f"data/processed_{dataset}/sequence_ordered.parquet")

    if not any((d["models"] / x).exists() for x in (
        "ml/ml_ensemble_chrono.joblib",
        "ml/ml_ensemble.joblib",
    )):
        missing.append(
            f"models/{dataset}/ml/ml_ensemble_chrono.joblib "
            "OR ml_ensemble.joblib"
        )

    if not any((d["models"] / x).exists() for x in (
        "dl/dl_model_chrono.keras",
        "dl/dl_model.keras",
        "dl/dl_model.h5",
    )):
        missing.append(f"models/{dataset}/dl/dl_model_chrono.keras")

    if not (d["models"] / "meta/meta_learner.joblib").exists():
        missing.append(f"models/{dataset}/meta/meta_learner.joblib")

    if not (d["results"] / "experiment_summary.json").exists():
        missing.append(f"results/{dataset}/experiment_summary.json")

    return not missing, missing


@st.cache_resource(show_spinner="Loading trained models and test data...")
def load_all(dataset):
    from src.inference.pipeline import TriHybridPipeline

    cfg = make_cfg(dataset)
    d = dirs(dataset)

    pipe = TriHybridPipeline(
        cfg,
        models_dir=str(PROJECT_ROOT / "models"),
        processed_dir=str(d["processed"]),
    )

    seq = pd.read_parquet(d["processed"] / "sequence_ordered.parquet")
    train, val, test = chronological_split(
        seq,
        cfg["data"]["test_size"],
        cfg["data"]["val_size"],
    )

    with open(d["results"] / "experiment_summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    return cfg, d, pipe, train, val, test, summary


# --------------------------------------------------------------------------
# Sidebar / dataset selector
# --------------------------------------------------------------------------

st.sidebar.title("🛡️ Tri-Hybrid XAI-IDS")

configured = str(BASE_CFG.get("data", {}).get("dataset", "nsl_kdd")).lower()
if configured not in DATASETS.values():
    configured = "nsl_kdd"

labels = list(DATASETS)
default_index = list(DATASETS.values()).index(configured)

dataset_label = st.sidebar.selectbox(
    "Select dataset",
    labels,
    index=default_index,
)
DATASET = DATASETS[dataset_label]

if st.sidebar.button("🔄 Reload selected dataset"):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.write("**Active paths**")
st.sidebar.code(
    f"data/processed_{DATASET}\n"
    f"models/{DATASET}\n"
    f"results/{DATASET}"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Live Traffic Feed",
        "Predict a Flow",
        "Explainability",
        "Alerts",
        "Model Performance",
        "Export Report",
    ],
)

ready, missing = artifact_status(DATASET)

if not ready:
    st.error(f"**{dataset_label} is not ready for the dashboard.**")
    st.write("Missing required artifacts:")
    for item in missing:
        st.code(item)
    st.info(
        "Train this dataset first, then select it again. "
        "Other trained datasets can still be viewed from the selector."
    )
    st.stop()

try:
    cfg, D, pipe, train_df, val_df, test_df, summary = load_all(DATASET)
except Exception as exc:
    st.error(f"Could not load the trained {dataset_label} artifacts.")
    st.exception(exc)
    st.stop()

ws = int(cfg["sequence"]["window_size"])
n_windows = max(0, len(test_df) - ws + 1)

if n_windows < 1:
    st.error("The selected dataset has no complete test windows.")
    st.stop()

normal_label = next(
    (
        x for x in pipe.label_encoder.classes_
        if str(x).upper() in ("BENIGN", "NORMAL")
    ),
    pipe.label_encoder.classes_[0],
)


def window(idx):
    return test_df.iloc[int(idx):int(idx) + ws]


def timestamp(df):
    if "Timestamp" not in df.columns:
        return "N/A"
    value = df["Timestamp"].iloc[-1]
    if pd.isna(value):
        return "N/A"
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)


def true_label(df):
    if "Label" not in df.columns:
        return "N/A"
    try:
        return pipe.label_encoder.inverse_transform(
            [int(df["Label"].values[-1])]
        )[0]
    except Exception:
        return str(df["Label"].values[-1])


def confidence(result):
    return float(np.max(np.asarray(result["final_proba"], dtype=float)))


def probability_df(result):
    classes = list(pipe.label_encoder.classes_)
    ml = np.zeros(len(classes))
    dl = np.zeros(len(classes))
    fused = np.zeros(len(classes))

    ml_p = np.asarray(result["ml_proba"], dtype=float)
    dl_p = np.asarray(result["dl_proba"], dtype=float)
    final_p = np.asarray(result["final_proba"], dtype=float)

    ml_classes = getattr(
        pipe.ml_model, "classes_", np.arange(len(ml_p))
    )
    for pos, cls in enumerate(np.asarray(ml_classes).reshape(-1)):
        cls = int(cls)
        if pos < len(ml_p) and 0 <= cls < len(ml):
            ml[cls] = ml_p[pos]

    meta_classes = getattr(
        pipe.meta_model, "classes_", np.arange(len(final_p))
    )
    for pos, cls in enumerate(np.asarray(meta_classes).reshape(-1)):
        cls = int(cls)
        if pos < len(final_p) and 0 <= cls < len(fused):
            fused[cls] = final_p[pos]

    mapping = getattr(pipe, "dl_class_indices", None)
    if mapping is None:
        mapping = list(range(len(dl_p)))

    for pos, cls in enumerate(mapping):
        cls = int(cls)
        if pos < len(dl_p) and 0 <= cls < len(dl):
            dl[cls] = dl_p[pos]

    return pd.DataFrame({
        "class": classes,
        "ML branch": ml,
        "DL branch": dl,
        "Fused (final)": fused,
    }).set_index("class")


def explanation(idx):
    wdf = window(idx)
    X = train_df[pipe.feature_names].to_numpy(dtype=np.float32)
    Xs = pipe.scaler.transform(X).astype(np.float32)
    ml_X = Xs[:, pipe._ml_feature_indices]
    return pipe.explain_window(wdf, ml_X[:200], ml_X), wdf


# --------------------------------------------------------------------------
# Main header
# --------------------------------------------------------------------------

st.title(f"Tri-Hybrid XAI-IDS — {dataset_label}")
st.caption(
    "Dataset-specific dashboard using real trained artifacts and held-out data."
)


# ==========================================================================
# HOME
# ==========================================================================

if page == "Home":
    st.header("Security Analyst Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test windows available", n_windows)

    for col, key, label in (
        (c2, "fused_meta_acc", "Fused test accuracy"),
        (c3, "branch_a_ml_only_acc", "Branch A (ML)"),
        (c4, "branch_b_dl_only_acc", "Branch B (DL)"),
    ):
        value = summary.get(key)
        col.metric(
            label,
            f"{float(value) * 100:.2f}%" if value is not None else "N/A",
        )

    st.subheader("Dataset & classes")
    left, right = st.columns(2)

    with left:
        st.write("**Dataset:**", summary.get("dataset_used", dataset_label))
        classes = summary.get(
            "classes",
            list(map(str, pipe.label_encoder.classes_)),
        )
        st.write("**Number of classes:**", len(classes))
        st.write("**Classes:**", ", ".join(map(str, classes)))
        st.write("**Sequence window:**", ws)

    with right:
        counts = test_df["Label"].value_counts()
        try:
            counts.index = pipe.label_encoder.inverse_transform(
                counts.index.astype(int)
            )
        except Exception:
            pass
        st.bar_chart(counts)

    plot = D["results"] / "plots/branch_comparison.png"
    st.subheader("Branch A vs Branch B vs Fused")
    if plot.exists():
        st.image(str(plot))
    else:
        st.info("Branch comparison plot is not available.")


# ==========================================================================
# LIVE TRAFFIC FEED
# ==========================================================================

elif page == "Live Traffic Feed":
    st.header("Live / Simulated Traffic Feed")
    st.caption("Real held-out test windows in chronological order.")

    n_show = st.slider(
        "Number of flows to show",
        5,
        min(50, n_windows),
        min(15, n_windows),
    )
    max_start = max(0, n_windows - n_show)
    start = st.number_input(
        "Start index in test split",
        0,
        max_start,
        0,
        step=1,
    )

    rows = []
    progress = st.progress(0.0, text="Scoring simulated traffic...")

    for i in range(n_show):
        idx = int(start) + i
        wdf = window(idx)
        result = pipe.predict_window(wdf)
        conf = confidence(result)
        pred = result["pred_label"]

        rows.append({
            "idx": idx,
            "Timestamp": timestamp(wdf),
            "Prediction": pred,
            "Confidence": round(conf, 3),
            "Risk": (
                "🔴 High" if pred != normal_label and conf > 0.7
                else "🟡 Medium" if pred != normal_label
                else "🟢 Normal"
            ),
        })
        progress.progress((i + 1) / n_show)

    progress.empty()
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ==========================================================================
# PREDICT A FLOW
# ==========================================================================

elif page == "Predict a Flow":
    st.header("Prediction — Full Pipeline")

    idx = st.number_input(
        f"Test-split window index (0 – {n_windows - 1})",
        0,
        n_windows - 1,
        min(int(st.session_state.get("last_idx", 0)), n_windows - 1),
        step=1,
    )

    if st.button("Run prediction", type="primary"):
        with st.spinner("Running ML → DL → meta-fusion..."):
            wdf = window(idx)
            result = pipe.predict_window(wdf)
            actual = true_label(wdf)

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted class", result["pred_label"])
        c2.metric("True label", actual)
        c3.metric("Confidence", f"{confidence(result):.3f}")

        st.info(f"**Timestamp:** {timestamp(wdf)}")

        st.subheader("Branch probabilities vs fused decision")
        st.bar_chart(probability_df(result))

        st.write({
            "dataset": dataset_label,
            "window_index": int(idx),
            "prediction": result["pred_label"],
            "ground_truth": actual,
            "correct": result["pred_label"] == actual,
        })

        st.session_state["last_idx"] = int(idx)


# ==========================================================================
# EXPLAINABILITY
# ==========================================================================

elif page == "Explainability":
    st.header("Multi-Method Explainability")

    default_idx = min(
        max(int(st.session_state.get("last_idx", 0)), 0),
        n_windows - 1,
    )

    idx = st.number_input(
        f"Test-split window index (0 – {n_windows - 1})",
        0,
        n_windows - 1,
        default_idx,
        step=1,
    )

    if st.button("Explain this prediction", type="primary"):
        with st.spinner("Computing SHAP + LIME + attention + agreement..."):
            result, wdf = explanation(idx)

        st.subheader(
            f"Prediction: **{result['pred_label']}** "
            f"(confidence {confidence(result):.2f})"
        )
        st.info(f"**Timestamp:** {timestamp(wdf)}")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**SHAP — Branch A**")
            st.bar_chart(
                pd.Series(result.get("shap_top", {})).sort_values()
            )

        with c2:
            st.markdown("**LIME — Branch A**")
            st.bar_chart(
                pd.Series(result.get("lime_top", {})).sort_values()
            )

        with c3:
            st.markdown("**Attention — Branch B**")
            st.bar_chart(
                pd.Series(result.get("attn_top", {}))
            )

        ag = result["agreement"]
        st.subheader("Cross-Explainer Agreement Check")

        st.markdown(
            f"### {'🟢 TRUSTED' if ag.get('status') == 'TRUSTED' else '🔴 REVIEW REQUIRED'} "
            f"(score: {ag.get('agreement_score', 'N/A')})"
        )

        c1, c2 = st.columns(2)
        c1.metric(
            "SHAP↔LIME feature Jaccard overlap",
            ag.get("feature_agreement_jaccard", "N/A"),
        )
        c2.metric(
            "Attention recency alignment",
            ag.get("recency_alignment", "N/A"),
        )

        if ag.get("reason"):
            st.info(ag["reason"])

        st.session_state["last_expl"] = {"result": result}


# ==========================================================================
# ALERTS
# ==========================================================================

elif page == "Alerts":
    st.header("Alerts — Non-Normal Predictions")

    max_scan = min(300, n_windows)
    min_scan = min(20, max_scan)

    n_scan = st.slider(
        "How many test windows to scan",
        min_scan,
        max_scan,
        min(60, max_scan),
    )

    alerts = []
    progress = st.progress(0.0, text="Scanning for alerts...")

    for i in range(n_scan):
        wdf = window(i)
        result = pipe.predict_window(wdf)

        if result["pred_label"] != normal_label:
            conf = confidence(result)
            alerts.append({
                "idx": i,
                "Timestamp": timestamp(wdf),
                "Attack type": result["pred_label"],
                "Confidence": round(conf, 3),
                "Severity": "High" if conf > 0.8 else "Medium",
            })

        progress.progress((i + 1) / n_scan)

    progress.empty()

    if alerts:
        st.dataframe(
            pd.DataFrame(alerts),
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("No non-normal predictions in the scanned range.")


# ==========================================================================
# MODEL PERFORMANCE
# ==========================================================================

elif page == "Model Performance":
    st.header("Model Performance — Real Experimental Results")

    metrics = D["results"] / "metrics.csv"
    if metrics.exists():
        st.dataframe(
            pd.read_csv(metrics),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("metrics.csv is not available.")

    cm = (
        D["results"]
        / "confusion_matrices"
        / "fused_test_confusion_matrix.png"
    )
    if cm.exists():
        st.subheader("Fused Test Confusion Matrix")
        st.image(str(cm))

    shap_plot = D["results"] / "shap" / "shap_global_importance.png"
    if shap_plot.exists():
        st.subheader("Global SHAP Feature Importance")
        st.image(str(shap_plot))

    st.subheader("Experiment Summary")
    st.dataframe(
        pd.DataFrame({
            "Metric": list(summary.keys()),
            "Value": [str(v) for v in summary.values()],
        }),
        width="stretch",
        hide_index=True,
    )


# ==========================================================================
# EXPORT REPORT
# ==========================================================================

elif page == "Export Report":
    st.header("Export Dashboard Report")
    st.caption(f"Selected dataset: **{dataset_label}**")

    key = f"export_selected_{DATASET}"

    if key not in st.session_state:
        st.session_state[key] = {s: False for s in ALL_SECTIONS}

    c1, c2 = st.columns(2)
    if c1.button("☑ Select All"):
        st.session_state[key] = {s: True for s in ALL_SECTIONS}
        st.rerun()

    if c2.button("☐ Clear All"):
        st.session_state[key] = {s: False for s in ALL_SECTIONS}
        st.rerun()

    selected_map = st.session_state[key]
    cols = st.columns(2)

    for i, section in enumerate(ALL_SECTIONS):
        with cols[i % 2]:
            selected_map[section] = st.checkbox(
                section,
                value=selected_map.get(section, False),
                key=f"{DATASET}_section_{section}",
            )

    selected = [s for s in ALL_SECTIONS if selected_map.get(s)]

    fmt = st.radio(
        "Export format",
        ["PDF", "PNG", "PDF + PNG"],
        horizontal=True,
    )

    if st.button("📑 Export Complete Dashboard"):
        st.session_state[key] = {s: True for s in ALL_SECTIONS}
        st.rerun()

    if not selected:
        st.info("Select at least one section to generate a report.")
    else:
        st.write("**Dataset:**", dataset_label)
        st.write("**Selected sections:**", ", ".join(selected))
        st.write("**Format:**", fmt)

    if st.button(
        "Generate Report",
        type="primary",
        disabled=not selected,
    ):
        ctx = {
            "summary": summary,
            "results_dir": D["results"],
            "label_encoder": pipe.label_encoder,
            "dataset": DATASET,
            "dataset_label": dataset_label,
        }

        feed = []
        for i in range(min(20, n_windows)):
            wdf = window(i)
            result = pipe.predict_window(wdf)
            conf = confidence(result)
            pred = result["pred_label"]
            feed.append({
                "idx": i,
                "Timestamp": timestamp(wdf),
                "Prediction": pred,
                "Confidence": round(conf, 3),
                "Risk": (
                    "High" if pred != normal_label and conf > 0.7
                    else "Medium" if pred != normal_label
                    else "Normal"
                ),
            })
        ctx["feed_rows"] = feed

        last_idx = st.session_state.get("last_idx")
        if last_idx is not None and 0 <= int(last_idx) < n_windows:
            wdf = window(last_idx)
            result = pipe.predict_window(wdf)
            ctx["pred_state"] = {
                "idx": int(last_idx),
                "timestamp": timestamp(wdf),
                "pred_label": result["pred_label"],
                "true_label": true_label(wdf),
                "match": result["pred_label"] == true_label(wdf),
                "confidence": confidence(result),
                "classes": list(pipe.label_encoder.classes_),
                "ml_proba": result["ml_proba"].tolist(),
                "dl_proba": result["dl_proba"].tolist(),
                "final_proba": result["final_proba"].tolist(),
            }

        if st.session_state.get("last_expl"):
            ctx["expl_state"] = st.session_state["last_expl"]

        alerts = []
        for i in range(min(60, n_windows)):
            wdf = window(i)
            result = pipe.predict_window(wdf)
            if result["pred_label"] != normal_label:
                conf = confidence(result)
                alerts.append({
                    "idx": i,
                    "Timestamp": timestamp(wdf),
                    "Attack type": result["pred_label"],
                    "Confidence": round(conf, 3),
                    "Severity": "High" if conf > 0.8 else "Medium",
                })
        ctx["alert_rows"] = alerts

        with st.spinner("Generating report..."):
            pdf = (
                generate_pdf(selected, ctx)
                if fmt in ("PDF", "PDF + PNG")
                else None
            )
            png = (
                generate_png_single(selected, ctx)
                if fmt in ("PNG", "PDF + PNG")
                else None
            )

        st.success(f"{dataset_label} report generated successfully.")

        if pdf:
            st.download_button(
                "⬇️ Download PDF",
                pdf,
                file_name=f"TriHybrid_XAI_IDS_{DATASET}_Report.pdf",
                mime="application/pdf",
            )

        if png:
            st.download_button(
                "⬇️ Download PNG",
                png,
                file_name=f"TriHybrid_XAI_IDS_{DATASET}_Report.png",
                mime="image/png",
            )

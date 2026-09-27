"""
dashboard/export_report.py — PDF and PNG report generation for Tri-Hybrid XAI-IDS.
"""
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image

# ── colour palette ──────────────────────────────────────────────────────────
C_DARK   = "#0d1b2a"
C_BLUE   = "#1e88e5"
C_LIGHT  = "#e3f2fd"
C_WHITE  = "#ffffff"
C_GREY   = "#f5f5f5"
C_TEXT   = "#212121"
C_SUB    = "#546e7a"
C_GREEN  = "#2e7d32"
C_RED    = "#c62828"

NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── helpers ──────────────────────────────────────────────────────────────────
def _fig(w=11, h=8.5):
    fig = plt.figure(figsize=(w, h), facecolor=C_WHITE)
    return fig


def _close(fig):
    plt.close(fig)


def _img_to_fig(img_path, title="", w=11, h=7):
    """Wrap an existing PNG into a matplotlib figure."""
    fig = _fig(w, h)
    ax = fig.add_axes([0, 0.05, 1, 0.9])
    ax.axis("off")
    try:
        img = plt.imread(str(img_path))
        ax.imshow(img, aspect="auto")
    except Exception:
        ax.text(0.5, 0.5, f"Image not available:\n{img_path}",
                ha="center", va="center", color=C_RED, fontsize=11,
                transform=ax.transAxes)
    if title:
        fig.text(0.5, 0.97, title, ha="center", va="top",
                 fontsize=13, fontweight="bold", color=C_DARK)
    return fig


def _table_fig(df, title="", col_widths=None, w=11, h=None):
    """Render a DataFrame as a styled matplotlib table figure."""
    rows = len(df)
    h = h or max(2.5, min(rows * 0.38 + 1.5, 9))
    fig = _fig(w, h)
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.85])
    ax.axis("off")
    if title:
        fig.text(0.5, 0.97, title, ha="center", va="top",
                 fontsize=13, fontweight="bold", color=C_DARK)
    if df.empty:
        ax.text(0.5, 0.5, "No data available.", ha="center", va="center",
                color=C_SUB, fontsize=11, transform=ax.transAxes)
        return fig

    cols = list(df.columns)
    cell_text = df.astype(str).values.tolist()
    n_cols = len(cols)
    cw = col_widths or [1 / n_cols] * n_cols

    tbl = ax.table(cellText=cell_text, colLabels=cols,
                   cellLoc="center", loc="center",
                   colWidths=cw)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(C_DARK)
            cell.set_text_props(color=C_WHITE, fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor(C_GREY)
        else:
            cell.set_facecolor(C_WHITE)
        cell.set_edgecolor("#cccccc")
    return fig


def _cover_fig(dataset, selected):
    fig = _fig(11, 8.5)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C_DARK)
    ax.axis("off")

    # header band
    ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.72), 0.9, 0.22,
                 boxstyle="round,pad=0.01", facecolor=C_BLUE,
                 edgecolor="none", transform=ax.transAxes))

    ax.text(0.5, 0.90, "Tri-Hybrid XAI-IDS",
            ha="center", va="center", fontsize=28, fontweight="bold",
            color=C_WHITE, transform=ax.transAxes)
    ax.text(0.5, 0.80, "Explainable AI-Based Intrusion Detection System",
            ha="center", va="center", fontsize=14, color=C_LIGHT,
            transform=ax.transAxes)

    ax.text(0.5, 0.67, f"Dataset: {dataset.upper()}",
            ha="center", va="center", fontsize=12, color=C_LIGHT,
            transform=ax.transAxes)
    ax.text(0.5, 0.62, f"Generated: {NOW}",
            ha="center", va="center", fontsize=11, color="#90caf9",
            transform=ax.transAxes)

    ax.text(0.5, 0.54, "Sections included in this report:",
            ha="center", va="center", fontsize=12, color=C_LIGHT,
            fontweight="bold", transform=ax.transAxes)

    for i, s in enumerate(selected):
        ax.text(0.5, 0.48 - i * 0.055, f"✓  {s}",
                ha="center", va="center", fontsize=10, color="#a5d6a7",
                transform=ax.transAxes)

    ax.text(0.5, 0.04,
            "C.V. Raman Global University · Dept. of CSE · 7th Sem Major Project",
            ha="center", va="center", fontsize=9, color="#78909c",
            transform=ax.transAxes)
    return fig


def _metrics_bar_fig(summary):
    labels = ["Branch A\n(ML only)", "Branch B\n(DL only)", "Fused\n(Proposed)"]
    vals = [
        summary.get("branch_a_ml_only_acc", 0),
        summary.get("branch_b_dl_only_acc", 0),
        summary.get("fused_meta_acc", 0),
    ]
    colors = [C_BLUE, "#ff7043", C_GREEN]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=C_WHITE)
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors,
                  width=0.45, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{v*100:.2f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=C_DARK)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Branch A vs Branch B vs Fused — Accuracy Comparison",
                 fontsize=12, fontweight="bold", color=C_DARK, pad=10)
    ax.set_facecolor(C_GREY)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _bar_fig(data_dict, title, xlabel="Value", color=C_BLUE):
    keys = list(data_dict.keys())
    vals = list(data_dict.values())
    fig, ax = plt.subplots(figsize=(9, max(3, len(keys) * 0.45 + 1)),
                           facecolor=C_WHITE)
    colors = [C_GREEN if v >= 0 else C_RED for v in vals]
    ax.barh(keys, vals, color=colors, edgecolor="white")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color=C_DARK)
    ax.set_facecolor(C_GREY)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _text_fig(text, title="", monospace=True, w=11, h=6):
    fig = _fig(w, h)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.88])
    ax.axis("off")
    if title:
        fig.text(0.5, 0.97, title, ha="center", va="top",
                 fontsize=13, fontweight="bold", color=C_DARK)
    font = "monospace" if monospace else "sans-serif"
    ax.text(0.01, 0.98, text, va="top", ha="left",
            fontsize=8, family=font, color=C_TEXT,
            transform=ax.transAxes, wrap=True,
            bbox=dict(facecolor=C_GREY, edgecolor="#cccccc",
                      boxstyle="round,pad=0.5"))
    return fig


def _footer(fig, page_num):
    fig.text(0.5, 0.01,
             f"Tri-Hybrid XAI-IDS  ·  Generated {NOW}  ·  Page {page_num}",
             ha="center", va="bottom", fontsize=7, color=C_SUB)


# ── section builders ─────────────────────────────────────────────────────────
def build_overview(summary, results_dir):
    figs = []
    # metrics summary table
    rows = [
        ["Dataset", summary.get("dataset_used", "—")],
        ["Classes", ", ".join(summary.get("classes", []))],
        ["Number of classes", str(summary.get("n_classes", "—"))],
        ["Fused accuracy", f"{summary.get('fused_meta_acc', 0)*100:.2f}%"],
        ["Fused macro-F1", f"{summary.get('fused_meta_f1_macro', 0)*100:.2f}%"],
        ["Branch A (ML) accuracy", f"{summary.get('branch_a_ml_only_acc', 0)*100:.2f}%"],
        ["Branch B (DL) accuracy", f"{summary.get('branch_b_dl_only_acc', 0)*100:.2f}%"],
        ["Runtime (s)", str(summary.get("runtime_seconds", "—"))],
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    figs.append(_table_fig(df, "System Overview — Key Metrics", col_widths=[0.5, 0.5]))
    figs.append(_metrics_bar_fig(summary))
    return figs


def build_branch_comparison(results_dir):
    img = results_dir / "plots" / "branch_comparison.png"
    return [_img_to_fig(img, "Branch A vs Branch B vs Fused — Comparison Chart")]


def build_live_feed(rows):
    if not rows:
        fig = _fig(11, 3)
        fig.text(0.5, 0.5, "No live feed data available for export.",
                 ha="center", va="center", color=C_SUB, fontsize=12)
        return [fig]
    df = pd.DataFrame(rows)[["idx", "Timestamp", "Prediction", "Confidence", "Risk"]]
    return [_table_fig(df, "Live Traffic Feed — Scored Windows")]


def build_prediction(pred_state):
    if not pred_state:
        fig = _fig(11, 3)
        fig.text(0.5, 0.5,
                 "No prediction has been generated for export.\nRun a prediction first on the 'Predict a Flow' page.",
                 ha="center", va="center", color=C_SUB, fontsize=12)
        return [fig]

    figs = []
    rows = [
        ["Window index", str(pred_state.get("idx", "—"))],
        ["Predicted class", pred_state.get("pred_label", "—")],
        ["True label", pred_state.get("true_label", "—")],
        ["Match", "✓ Correct" if pred_state.get("match") else "✗ Mismatch"],
        ["Confidence", f"{pred_state.get('confidence', 0)*100:.1f}%"],
    ]
    df = pd.DataFrame(rows, columns=["Field", "Value"])
    figs.append(_table_fig(df, "Single Flow Prediction", col_widths=[0.4, 0.6]))

    # probability chart
    classes = pred_state.get("classes", [])
    ml_p = pred_state.get("ml_proba", [])
    dl_p = pred_state.get("dl_proba", [])
    fu_p = pred_state.get("final_proba", [])
    if classes and ml_p:
        x = np.arange(len(classes))
        w = 0.25
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=C_WHITE)
        ax.bar(x - w, np.array(ml_p)*100, w, label="ML Branch", color=C_BLUE)
        ax.bar(x,     np.array(dl_p)*100, w, label="DL Branch", color="#ff7043")
        ax.bar(x + w, np.array(fu_p)*100, w, label="Fused",     color=C_GREEN)
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=20, ha="right")
        ax.set_ylabel("Probability (%)"); ax.set_ylim(0, 110)
        ax.set_title("Branch Probabilities vs Fused Decision",
                     fontsize=12, fontweight="bold", color=C_DARK)
        ax.legend(); ax.set_facecolor(C_GREY)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        figs.append(fig)
    return figs


def build_explainability(expl_state):
    if not expl_state:
        fig = _fig(11, 3)
        fig.text(0.5, 0.5,
                 "No explainability results available.\nRun 'Explain this prediction' first.",
                 ha="center", va="center", color=C_SUB, fontsize=12)
        return [fig]

    figs = []
    result = expl_state.get("result", {})

    # header
    rows = [
        ["Predicted class", result.get("pred_label", "—")],
        ["Confidence", f"{float(np.max(result.get('final_proba', [0]))):.3f}"],
    ]
    figs.append(_table_fig(pd.DataFrame(rows, columns=["Field", "Value"]),
                           "Explainability — Prediction Summary", col_widths=[0.4, 0.6]))

    if result.get("shap_top"):
        figs.append(_bar_fig(result["shap_top"],
                             "SHAP Feature Attribution (Branch A)", "SHAP value"))
    if result.get("lime_top"):
        figs.append(_bar_fig(result["lime_top"],
                             "LIME Local Explanation (Branch A)", "Coefficient"))
    if result.get("attn_top"):
        figs.append(_bar_fig(result["attn_top"],
                             "Attention Weights (Branch B — per time-step)", "Weight"))
    return figs


def build_agreement(expl_state):
    if not expl_state:
        fig = _fig(11, 3)
        fig.text(0.5, 0.5, "No agreement data available.",
                 ha="center", va="center", color=C_SUB, fontsize=12)
        return [fig]

    ag = expl_state.get("result", {}).get("agreement", {})
    rows = [
        ["Status", ag.get("status", "—")],
        ["Agreement score", str(ag.get("agreement_score", "—"))],
        ["SHAP↔LIME Jaccard overlap", str(ag.get("feature_agreement_jaccard", "—"))],
        ["Attention recency alignment", str(ag.get("recency_alignment", "—"))],
        ["Reason", ag.get("reason", "—") or "—"],
    ]
    df = pd.DataFrame(rows, columns=["Field", "Value"])
    return [_table_fig(df, "Cross-Explainer Agreement Check", col_widths=[0.45, 0.55])]


def build_alerts(alert_rows):
    if not alert_rows:
        fig = _fig(11, 3)
        fig.text(0.5, 0.5, "No alerts found in the scanned range.",
                 ha="center", va="center", color=C_SUB, fontsize=12)
        return [fig]
    df = pd.DataFrame(alert_rows)
    return [_table_fig(df, "Alerts — Non-Normal Predictions")]


def build_model_performance(results_dir):
    figs = []
    csv_path = results_dir / "metrics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        figs.append(_table_fig(df, "Model Performance Metrics"))
    else:
        fig = _fig(11, 2)
        fig.text(0.5, 0.5, "metrics.csv not found.",
                 ha="center", va="center", color=C_RED, fontsize=11)
        figs.append(fig)
    return figs


def build_confusion_matrix(results_dir):
    img = results_dir / "confusion_matrices" / "fused_test_confusion_matrix.png"
    return [_img_to_fig(img, "Fused Model — Confusion Matrix")]


def build_shap_importance(results_dir):
    figs = []
    img = results_dir / "shap" / "shap_global_importance.png"
    figs.append(_img_to_fig(img, "Global SHAP Feature Importance"))
    csv_path = results_dir / "shap" / "feature_importance.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path).head(20)
        figs.append(_table_fig(df, "Top Feature Importances (SHAP)"))
    return figs


def build_fusion_metrics(results_dir):
    fpath = results_dir / "fusion_metrics.json"
    if not fpath.exists():
        fig = _fig(11, 2)
        fig.text(0.5, 0.5, "fusion_metrics.json not found.",
                 ha="center", va="center", color=C_RED, fontsize=11)
        return [fig]
    with open(fpath) as f:
        fm = json.load(f)
    rows = [[k, f"{v*100:.4f}%" if isinstance(v, float) else str(v)]
            for k, v in fm.items()]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    return [_table_fig(df, "Fusion Metrics", col_widths=[0.6, 0.4])]


def build_classification_report(results_dir, label_encoder=None):
    fpath = results_dir / "fusion_classification_report.txt"
    if not fpath.exists():
        fig = _fig(11, 2)
        fig.text(0.5, 0.5, "fusion_classification_report.txt not found.",
                 ha="center", va="center", color=C_RED, fontsize=11)
        return [fig]
    text = fpath.read_text()
    # try to parse into a table
    figs = []
    try:
        lines = [l for l in text.strip().splitlines() if l.strip()]
        data_rows = []
        for line in lines:
            parts = line.split()
            if len(parts) == 5 and parts[0].lstrip("-").isdigit():
                cls = parts[0]
                if label_encoder is not None:
                    try:
                        cls = label_encoder.inverse_transform([int(cls)])[0]
                    except Exception:
                        pass
                data_rows.append([cls] + parts[1:])
            elif len(parts) >= 5 and parts[0] in ("macro", "weighted"):
                data_rows.append([f"{parts[0]} {parts[1]}"] + parts[2:])
            elif parts[0] == "accuracy":
                data_rows.append(["accuracy", "", "", parts[1], parts[2]])
        if data_rows:
            df = pd.DataFrame(data_rows,
                              columns=["Class", "Precision", "Recall", "F1-score", "Support"])
            figs.append(_table_fig(df, "Classification Report — Fused Model"))
        else:
            raise ValueError("parse failed")
    except Exception:
        figs.append(_text_fig(text, "Classification Report — Fused Model"))
    return figs


def build_experiment_summary(summary):
    skip = {"NOTE"}
    rows = []
    for k, v in summary.items():
        if k in skip:
            continue
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        elif isinstance(v, float):
            v = f"{v:.6f}"
        rows.append([k, str(v)])
    df = pd.DataFrame(rows, columns=["Key", "Value"])
    return [_table_fig(df, "Experiment Summary", col_widths=[0.45, 0.55])]


# ── main generators ───────────────────────────────────────────────────────────
SECTION_BUILDERS = {
    "Home / Overview":               lambda ctx: build_overview(ctx["summary"], ctx["results_dir"]),
    "Branch A vs B vs Fused":        lambda ctx: build_branch_comparison(ctx["results_dir"]),
    "Live Traffic Feed":             lambda ctx: build_live_feed(ctx.get("feed_rows", [])),
    "Single Flow Prediction":        lambda ctx: build_prediction(ctx.get("pred_state")),
    "Explainability":                lambda ctx: build_explainability(ctx.get("expl_state")),
    "Cross-Explainer Agreement":     lambda ctx: build_agreement(ctx.get("expl_state")),
    "Alerts":                        lambda ctx: build_alerts(ctx.get("alert_rows", [])),
    "Model Performance":             lambda ctx: build_model_performance(ctx["results_dir"]),
    "Confusion Matrix":              lambda ctx: build_confusion_matrix(ctx["results_dir"]),
    "SHAP Feature Importance":       lambda ctx: build_shap_importance(ctx["results_dir"]),
    "Fusion Metrics":                lambda ctx: build_fusion_metrics(ctx["results_dir"]),
    "Classification Report":         lambda ctx: build_classification_report(
                                         ctx["results_dir"], ctx.get("label_encoder")),
    "Experiment Summary":            lambda ctx: build_experiment_summary(ctx["summary"]),
}

ALL_SECTIONS = list(SECTION_BUILDERS.keys())


def generate_pdf(selected_sections, ctx) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        page_num = 1
        # cover
        fig = _cover_fig(ctx["summary"].get("dataset_used", "unknown"), selected_sections)
        _footer(fig, page_num)
        pdf.savefig(fig, bbox_inches="tight")
        _close(fig)
        page_num += 1

        for sec in selected_sections:
            builder = SECTION_BUILDERS.get(sec)
            if not builder:
                continue
            figs = builder(ctx)
            for fig in figs:
                _footer(fig, page_num)
                pdf.savefig(fig, bbox_inches="tight")
                _close(fig)
                page_num += 1

        # metadata
        d = pdf.infodict()
        d["Title"] = "Tri-Hybrid XAI-IDS Report"
        d["Author"] = "Tri-Hybrid XAI-IDS Dashboard"
        d["Subject"] = f"Dataset: {ctx['summary'].get('dataset_used','')}"
        d["CreationDate"] = datetime.now()

    buf.seek(0)
    return buf.read()


def generate_png_single(selected_sections, ctx) -> bytes:
    """Render every section figure and stitch them vertically into one PNG."""
    DPI = 150
    raw_images: list[Image.Image] = []

    def _fig_to_pil(fig) -> Image.Image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
        _close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # cover
    page_num = 1
    fig = _cover_fig(ctx["summary"].get("dataset_used", "unknown"), selected_sections)
    _footer(fig, page_num)
    raw_images.append(_fig_to_pil(fig))
    page_num += 1

    for sec in selected_sections:
        builder = SECTION_BUILDERS.get(sec)
        if not builder:
            continue
        for fig in builder(ctx):
            _footer(fig, page_num)
            raw_images.append(_fig_to_pil(fig))
            page_num += 1

    # unify width — scale each image to the widest one
    max_w = max(img.width for img in raw_images)
    resized = []
    for img in raw_images:
        if img.width != max_w:
            scale = max_w / img.width
            new_h = int(img.height * scale)
            img = img.resize((max_w, new_h), Image.LANCZOS)
        resized.append(img.convert("RGB"))

    total_h = sum(img.height for img in resized)
    canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for img in resized:
        canvas.paste(img, (0, y))
        y += img.height

    out = io.BytesIO()
    canvas.save(out, format="PNG", optimize=False)
    out.seek(0)
    return out.read()

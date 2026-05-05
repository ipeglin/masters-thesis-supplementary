"""
KNN Probabilistic Forecasting Results — Reliability, ROC/PR, and Uncertainty.

Reads the probabilistic classification reports written by
`crates/08classification/src/eval.rs`. Each report contains:

* hard-decision metrics at the legacy 0.5 threshold and at the Youden-optimal
  threshold,
* probabilistic metrics for both raw KNN vote-share and Platt-scaled outputs
  (Brier, log loss, AUC-ROC, AUC-PR, ECE, calibration bins, threshold sweep),
* per-sample test/val predictions with `p1_raw` and `p1_calibrated`.

This script produces, per (roi_fingerprint, run, k, distance metric):

1. Reliability diagram grid — analysis × source. Diagonal = perfect
   calibration; raw vs Platt-calibrated drawn together.
2. ROC and PR curves — analysis × source, raw vs calibrated overlay, with the
   Youden-optimal operating point marked.
3. Subject-rank uncertainty plot — every test sample ordered by
   `p1_calibrated`, true label encoded as colour, with the [0.4, 0.6]
   "uncertainty band" shaded so subjects sitting near the decision boundary
   are visually obvious. Reflects the supervisor's framing of clinical
   ambiguity in spectrum disorders.
4. Probabilistic summary table — Brier / log loss / AUC-ROC / AUC-PR / ECE
   per analysis × source, raw vs calibrated, dumped as CSV.

Usage: run as a module so `lib.*` and `plotting.plot_config` resolve relative
to the repo root. Adjust `RESULTS_DIR` to point at the run's output.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lib.fs.get_script_name import get_name
from lib.fs.project_config import REPO_ROOT
from plotting import plot_config

# Directory where the Rust pipeline writes classification reports.
# ANALYSIS_RESULTS_DIR = Path("/Users/ipeglin/Documents/masters_thesis/classifier_results/")
ANALYSIS_RESULTS_DIR = Path("/Volumes/work/classifier_results")  # IDUN network mount

# Match the conventions in `knn_results_cmp.py`.
ANALYSIS_ORDER = [
    "Baseline Resized",
    "Baseline Chunked",
    "Baseline Averaged",
    "Task Per Block Resized",
    "Task Per Block",
    "Task Averaged Resized",
    "Task Averaged",
    "Task Concat",
]

ANALYSIS_SUBGROUPS = {
    "baseline": [a for a in ANALYSIS_ORDER if "Baseline" in a],
    "task": [a for a in ANALYSIS_ORDER if "Task" in a],
}

UNNAMED_ROI_SELECTION = "_unnamed"
NETWORK_SUFFIX_SEP = "__net-"

SHOW_FIGURES = False
PLOT_MEAN_ANALYSES = True
UNCERTAINTY_BAND = (0.4, 0.6)
N_RELIABILITY_BINS = 10

# ---------------------------------------------------------------------------
# Formatting / parsing helpers (mirror knn_results_cmp.py)
# ---------------------------------------------------------------------------

def format_label(text):
    if not text:
        return text
    if text.lower() == "hht_roi":
        return "HHT ROI"
    if text.lower() in ("cwt", "hht", "ts"):
        return text.upper()
    return text.replace("_", " ").title()


def slugify(text):
    if not text:
        return "unspecified"
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "unspecified"


def parse_roi_dir(dir_name: str):
    if NETWORK_SUFFIX_SEP not in dir_name:
        return dir_name, ""
    base, networks_part = dir_name.split(NETWORK_SUFFIX_SEP, 1)
    networks_label = "+".join(networks_part.split("_")) if networks_part else ""
    return base, networks_label


def _roi_title(roi_name: str, networks_label: str) -> str:
    roi_name_clean = roi_name.replace("_", " ")
    return f"{roi_name_clean} [{networks_label}]" if networks_label else roi_name_clean


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reports(directory: Path) -> pd.DataFrame:
    """Return a row-per-report dataframe with raw + calibrated metrics columns
    plus the embedded per-sample prediction lists (used for ROC/PR/uncertainty
    plots that need full scores).
    """
    base = Path(directory).resolve()
    records = []
    for filepath in base.rglob("*classification.json"):
        with open(filepath, "r") as f:
            data = json.load(f)

        run_match = re.search(r"run-(\d+)", filepath.stem)
        run_idx = int(run_match.group(1)) if run_match else 0

        parent = filepath.parent.resolve()
        if parent == base:
            dir_selection = UNNAMED_ROI_SELECTION
        else:
            dir_selection = parent.relative_to(base).parts[0]
        roi_name, networks_label = parse_roi_dir(dir_selection)
        roi_fingerprint = data.get("roi_selection_fingerprint") or dir_selection

        raw_analysis = data.get("analysis", "")
        is_mean_analysis = raw_analysis.endswith("_mean")
        clean_analysis = format_label(raw_analysis.replace("_mean", ""))

        test = data.get("test", {})
        val = data.get("val", {})
        record = {
            "run": run_idx,
            "analysis": clean_analysis,
            "is_mean": is_mean_analysis,
            "source": format_label(data.get("source")),
            "k": data.get("num_neighbors"),
            "metric": format_label(data.get("metric")),
            "roi_selection": dir_selection,
            "roi_name": roi_name,
            "roi_networks": networks_label,
            "roi_fingerprint": roi_fingerprint,
            "platt_a": data.get("platt_a"),
            "platt_b": data.get("platt_b"),
            "test_predictions": data.get("test_predictions", []),
            "val_predictions": data.get("val_predictions", []),
            "test_at_0_5": test.get("at_0_5", {}),
            "test_at_youden": test.get("at_youden", {}),
            "test_raw": test.get("raw", {}),
            "test_calibrated": test.get("calibrated", {}),
            "val_raw": val.get("raw", {}),
            "val_calibrated": val.get("calibrated", {}),
        }
        records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    in_df = df["analysis"].unique()
    missing = [x for x in in_df if x not in ANALYSIS_ORDER]
    df["analysis"] = pd.Categorical(
        df["analysis"], categories=list(ANALYSIS_ORDER) + missing, ordered=True
    )
    df = df.sort_values(["analysis", "source"])
    return df


# ---------------------------------------------------------------------------
# Curve helpers (hand-rolled — avoids a sklearn dep on top of the repo's stack)
# ---------------------------------------------------------------------------

def roc_curve(y_true: np.ndarray, scores: np.ndarray):
    """Return (fpr, tpr) sweeping every distinct score, descending."""
    pos = float(np.sum(y_true == 1))
    neg = float(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    scores_sorted = scores[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    distinct = np.r_[np.where(np.diff(scores_sorted))[0], len(scores_sorted) - 1]
    tpr = np.r_[0.0, tps[distinct] / pos, 1.0]
    fpr = np.r_[0.0, fps[distinct] / neg, 1.0]
    return fpr, tpr


def pr_curve(y_true: np.ndarray, scores: np.ndarray):
    pos = float(np.sum(y_true == 1))
    if pos == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 1.0])
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    scores_sorted = scores[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    distinct = np.r_[np.where(np.diff(scores_sorted))[0], len(scores_sorted) - 1]
    recall = tps[distinct] / pos
    precision = tps[distinct] / np.maximum(tps[distinct] + fps[distinct], 1e-9)
    # Prepend (0, first precision) so the curve starts at recall=0.
    recall = np.r_[0.0, recall]
    precision = np.r_[precision[0], precision]
    return recall, precision


def auc_trapezoidal(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reliability_grid(group_df: pd.DataFrame, save_dir: Path,
                          run_val: int, k_val: int, metric_val: str,
                          roi_selection: str, roi_name: str, networks_label: str,
                          filename_suffix: str = "") -> None:
    analyses = list(group_df["analysis"].cat.categories)
    sources = sorted(group_df["source"].unique())
    if not sources or not analyses:
        return
    n_rows = len(analyses)
    n_cols = len(sources)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows),
                             sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    active_grid = np.zeros((n_rows, n_cols), dtype=bool)
    for r, analysis in enumerate(analyses):
        for c, source in enumerate(sources):
            if not group_df[(group_df["analysis"] == analysis) & (group_df["source"] == source)].empty:
                active_grid[r, c] = True

    for r, analysis in enumerate(analyses):
        for c, source in enumerate(sources):
            ax = axes[r, c]
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
            if not active_grid[r, c]:
                ax.set_visible(False)
                continue
            row = group_df[(group_df["analysis"] == analysis) & (group_df["source"] == source)].iloc[0]
            for variant, color, lbl in [
                ("test_raw", "tab:orange", "raw"),
                ("test_calibrated", "tab:blue", "Platt"),
            ]:
                bins = row[variant].get("calibration_bins", [])
                xs = [b["mean_pred"] for b in bins if b["count"] > 0]
                ys = [b["frac_positive"] for b in bins if b["count"] > 0]
                sizes = np.array([b["count"] for b in bins if b["count"] > 0], dtype=float)
                if not xs:
                    continue
                sizes = 25 + 175 * sizes / sizes.max()
                ece = row[variant].get("expected_calibration_error", float("nan"))
                ax.scatter(xs, ys, s=sizes, color=color, alpha=0.6,
                           label=f"{lbl} (ECE={ece:.3f})")
                ax.plot(xs, ys, color=color, alpha=0.4)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f"{analysis} | {source}", fontsize=10)
            
            is_bottom = (r == np.where(active_grid[:, c])[0][-1])
            is_left = (c == np.where(active_grid[r, :])[0][0])
            
            if is_bottom:
                ax.set_xlabel(r"Predicted $P(\mathrm{anhedonic})$")
                ax.tick_params(labelbottom=True)
            if is_left:
                ax.set_ylabel("Empirical fraction anhedonic")
                ax.tick_params(labelleft=True)
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Reliability - ROI={_roi_title(roi_name, networks_label)} | "
        f"Run {run_val:02d} | K={k_val} | {metric_val}",
        y=1.02,
    )
    fig.tight_layout()
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    filename = (
        f"knn_reliability_roi-{slugify(roi_selection)}_run{run_val:02d}_k{k_val}"
        f"_{slugify(metric_val)}{suffix_str}.pdf"
    )
    fig.savefig(save_dir / filename)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_roc_pr_grid(group_df: pd.DataFrame, save_dir: Path,
                     run_val: int, k_val: int, metric_val: str,
                     roi_selection: str, roi_name: str, networks_label: str,
                     filename_suffix: str = "") -> None:
    analyses = list(group_df["analysis"].cat.categories)
    sources = sorted(group_df["source"].unique())
    if not sources or not analyses:
        return
    n_rows = len(analyses)
    n_cols = len(sources)
    fig, axes = plt.subplots(n_rows, 2 * n_cols,
                             figsize=(4.0 * 2 * n_cols, 3.6 * n_rows),
                             squeeze=False)
    
    active_grid = np.zeros((n_rows, n_cols), dtype=bool)
    for r, analysis in enumerate(analyses):
        for c, source in enumerate(sources):
            row = group_df[(group_df["analysis"] == analysis) & (group_df["source"] == source)]
            if not row.empty and row.iloc[0].get("test_predictions"):
                active_grid[r, c] = True

    fig.suptitle(
        f"ROC & PR - ROI={_roi_title(roi_name, networks_label)} | "
        f"Run {run_val:02d} | K={k_val} | {metric_val}",
        y=1.01,
    )

    for r, analysis in enumerate(analyses):
        for c, source in enumerate(sources):
            ax_roc = axes[r, 2 * c]
            ax_pr = axes[r, 2 * c + 1]
            row = group_df[(group_df["analysis"] == analysis) & (group_df["source"] == source)]
            if row.empty:
                ax_roc.set_visible(False); ax_pr.set_visible(False)
                continue
            preds = row.iloc[0]["test_predictions"]
            if not preds:
                ax_roc.set_visible(False); ax_pr.set_visible(False)
                continue
            y = np.array([p["y_true"] for p in preds])
            for score_key, color, lbl in [
                ("p1_raw", "tab:orange", "raw"),
                ("p1_calibrated", "tab:blue", "Platt"),
            ]:
                s = np.array([p[score_key] for p in preds], dtype=float)
                fpr, tpr = roc_curve(y, s)
                rec, prec = pr_curve(y, s)
                auc_r = auc_trapezoidal(fpr, tpr)
                auc_p = auc_trapezoidal(rec, prec)
                ax_roc.plot(fpr, tpr, color=color, linewidth=1.5,
                            label=f"{lbl} AUC={auc_r:.3f}")
                ax_pr.plot(rec, prec, color=color, linewidth=1.5,
                           label=f"{lbl} AUC={auc_p:.3f}")

            youden = row.iloc[0]["test_at_youden"].get("threshold")
            t05_acc = row.iloc[0]["test_at_0_5"].get("accuracy")
            if youden is not None:
                s = np.array([p["p1_calibrated"] for p in preds], dtype=float)
                
                # Youden point
                pred_y = (s >= youden).astype(int)
                tp_y = int(np.sum((pred_y == 1) & (y == 1)))
                fp_y = int(np.sum((pred_y == 1) & (y == 0)))
                fn_y = int(np.sum((pred_y == 0) & (y == 1)))
                tn_y = int(np.sum((pred_y == 0) & (y == 0)))
                pos_y = max(tp_y + fn_y, 1); neg_y = max(tn_y + fp_y, 1)
                ax_roc.scatter([fp_y / neg_y], [tp_y / pos_y], color="green", s=40, zorder=5,
                               label=f"Youden t={youden:.2f}")
                pred_pos_y = max(tp_y + fp_y, 1)
                ax_pr.scatter([tp_y / pos_y], [tp_y / pred_pos_y], color="green", s=40, zorder=5,
                              label=f"Youden t={youden:.2f}")

                # 0.5 point
                pred_05 = (s >= 0.5).astype(int)
                tp_05 = int(np.sum((pred_05 == 1) & (y == 1)))
                fp_05 = int(np.sum((pred_05 == 1) & (y == 0)))
                fn_05 = int(np.sum((pred_05 == 0) & (y == 1)))
                tn_05 = int(np.sum((pred_05 == 0) & (y == 0)))
                pos_05 = max(tp_05 + fn_05, 1); neg_05 = max(tn_05 + fp_05, 1)
                ax_roc.scatter([fp_05 / neg_05], [tp_05 / pos_05], color="purple", marker="x", s=40, zorder=5,
                               label=f"t=0.5 (acc={t05_acc:.3f})")
                pred_pos_05 = max(tp_05 + fp_05, 1)
                ax_pr.scatter([tp_05 / pos_05], [tp_05 / pred_pos_05], color="purple", marker="x", s=40, zorder=5,
                              label=f"t=0.5 (acc={t05_acc:.3f})")

            ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.5)
            ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
            ax_roc.set_title(f"ROC | {analysis} | {source}", fontsize=10)
            ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.05)
            ax_pr.set_title(f"PR | {analysis} | {source}", fontsize=10)
            
            is_bottom = (r == np.where(active_grid[:, c])[0][-1])
            is_left = (c == np.where(active_grid[r, :])[0][0])
            
            if is_bottom:
                ax_roc.set_xlabel("False positive rate")
                ax_pr.set_xlabel("Recall")
            if is_left:
                ax_roc.set_ylabel("True positive rate")
                ax_pr.set_ylabel("Precision")
            
            ax_roc.legend(loc="lower right", fontsize=8)
            ax_pr.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    filename = (
        f"knn_roc_pr_roi-{slugify(roi_selection)}_run{run_val:02d}_k{k_val}"
        f"_{slugify(metric_val)}{suffix_str}.pdf"
    )
    fig.savefig(save_dir / filename)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_uncertainty_grid(group_df: pd.DataFrame, save_dir: Path,
                          run_val: int, k_val: int, metric_val: str,
                          roi_selection: str, roi_name: str, networks_label: str,
                          filename_suffix: str = "") -> None:
    analyses = list(group_df["analysis"].cat.categories)
    sources = sorted(group_df["source"].unique())
    if not sources or not analyses:
        return
    n_rows = len(analyses)
    n_cols = len(sources)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 3.0 * n_rows),
                             squeeze=False)
                             
    active_grid = np.zeros((n_rows, n_cols), dtype=bool)
    for r, analysis in enumerate(analyses):
        for c, source in enumerate(sources):
            row = group_df[(group_df["analysis"] == analysis) & (group_df["source"] == source)]
            if not row.empty and row.iloc[0].get("test_predictions"):
                active_grid[r, c] = True

    fig.suptitle(
        f"Per-sample uncertainty - ROI={_roi_title(roi_name, networks_label)} | "
        f"Run {run_val:02d} | K={k_val} | {metric_val}",
        y=1.01,
    )

    for r, analysis in enumerate(analyses):
        for c, source in enumerate(sources):
            ax = axes[r, c]
            row = group_df[(group_df["analysis"] == analysis) & (group_df["source"] == source)]
            if row.empty:
                ax.set_visible(False); continue
            preds = row.iloc[0]["test_predictions"]
            if not preds:
                ax.set_visible(False); continue
            df_p = pd.DataFrame(preds).sort_values("p1_calibrated").reset_index(drop=True)
            colors = ["tab:blue" if y == 0 else "tab:red" for y in df_p["y_true"]]
            ax.scatter(range(len(df_p)), df_p["p1_calibrated"], c=colors, s=14, alpha=0.8)
            ax.axhspan(*UNCERTAINTY_BAND, color="gray", alpha=0.15,
                       label=f"uncertainty band {UNCERTAINTY_BAND}")
            ax.axhline(0.5, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
            youden = row.iloc[0]["test_at_youden"].get("threshold")
            if youden is not None:
                ax.axhline(youden, color="green", linestyle="-.", linewidth=1.0,
                           alpha=0.7, label=f"Youden t={youden:.2f}")
            in_band = df_p[(df_p["p1_calibrated"] >= UNCERTAINTY_BAND[0]) &
                           (df_p["p1_calibrated"] <= UNCERTAINTY_BAND[1])]
            ax.set_title(
                f"{analysis} | {source} | n uncertain: {len(in_band)} / {len(df_p)}",
                fontsize=10,
            )
            ax.set_ylim(0, 1)
            
            is_bottom = (r == np.where(active_grid[:, c])[0][-1])
            is_left = (c == np.where(active_grid[r, :])[0][0])
            
            if is_bottom:
                ax.set_xlabel("Samples ranked by P(anhedonic)")
            if is_left:
                ax.set_ylabel(r"Calibrated $P(\mathrm{anhedonic})$")
            ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    filename = (
        f"knn_uncertainty_roi-{slugify(roi_selection)}_run{run_val:02d}_k{k_val}"
        f"_{slugify(metric_val)}{suffix_str}.pdf"
    )
    fig.savefig(save_dir / filename)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_confusion_grid(group_df: pd.DataFrame, save_dir: Path,
                        run_val: int, k_val: int, metric_val: str,
                        roi_selection: str, roi_name: str, networks_label: str,
                        filename_suffix: str = "") -> None:
    analyses = list(group_df["analysis"].cat.categories)
    sources = sorted(group_df["source"].unique())
    if not sources or not analyses:
        return
    n_rows = len(analyses)
    n_cols = len(sources)
    fig, axes = plt.subplots(n_rows, 2 * n_cols,
                             figsize=(6.0 * 2 * n_cols, 4.5 * n_rows),
                             squeeze=False)
                             
    active_grid = np.zeros((n_rows, n_cols), dtype=bool)
    for r, analysis in enumerate(analyses):
        for c, source in enumerate(sources):
            row = group_df[(group_df["analysis"] == analysis) & (group_df["source"] == source)]
            if not row.empty and row.iloc[0].get("test_predictions"):
                active_grid[r, c] = True

    fig.suptitle(
        f"Confusion Matrices - ROI={_roi_title(roi_name, networks_label)} | "
        f"Run {run_val:02d} | K={k_val} | {metric_val}",
        y=1.01,
    )

    for r, analysis in enumerate(analyses):
        for c, source in enumerate(sources):
            ax_05 = axes[r, 2 * c]
            ax_youden = axes[r, 2 * c + 1]
            row = group_df[(group_df["analysis"] == analysis) & (group_df["source"] == source)]
            if row.empty:
                ax_05.set_visible(False)
                ax_youden.set_visible(False)
                continue
            
            preds = row.iloc[0]["test_predictions"]
            if not preds:
                ax_05.set_visible(False)
                ax_youden.set_visible(False)
                continue
                
            y_true = np.array([p["y_true"] for p in preds])
            p1_calibrated = np.array([p["p1_calibrated"] for p in preds])
            
            youden = row.iloc[0]["test_at_youden"].get("threshold", 0.5)
            
            for threshold, ax, title_prefix in [(0.5, ax_05, "t=0.5"), (youden, ax_youden, f"t={youden:.2f}")]:
                y_pred = (p1_calibrated >= threshold).astype(int)
                
                tp = int(np.sum((y_pred == 1) & (y_true == 1)))
                fp = int(np.sum((y_pred == 1) & (y_true == 0)))
                fn = int(np.sum((y_pred == 0) & (y_true == 1)))
                tn = int(np.sum((y_pred == 0) & (y_true == 0)))
                
                cm = np.array([[tn, fp], [fn, tp]])
                
                acc = (tp + tn) / max(1, len(y_true))
                sens = tp / max(1, tp + fn)
                spec = tn / max(1, tn + fp)
                
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"], ax=ax)
                
                ax.set_title(f"{analysis} | {source}\n{title_prefix} | Acc: {acc:.2f} | Sens: {sens:.2f} | Spec: {spec:.2f}",
                             fontsize=10)
                
                if r == np.where(active_grid[:, c])[0][-1]:
                    ax.set_xlabel("Predicted")
                if c == 0 and ax == ax_05:
                    ax.set_ylabel("True")
                else:
                    ax.set_ylabel("")

    fig.tight_layout()
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    filename = (
        f"knn_cm_roi-{slugify(roi_selection)}_run{run_val:02d}_k{k_val}"
        f"_{slugify(metric_val)}{suffix_str}.pdf"
    )
    fig.savefig(save_dir / filename)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "run": r["run"],
            "analysis": r["analysis"],
            "source": r["source"],
            "k": r["k"],
            "metric": r["metric"],
            "roi_selection": r["roi_selection"],
            "test_acc_0_5": r["test_at_0_5"].get("accuracy"),
            "test_acc_youden": r["test_at_youden"].get("accuracy"),
            "youden_threshold": r["test_at_youden"].get("threshold"),
            "brier_raw": r["test_raw"].get("brier"),
            "brier_calibrated": r["test_calibrated"].get("brier"),
            "log_loss_raw": r["test_raw"].get("log_loss"),
            "log_loss_calibrated": r["test_calibrated"].get("log_loss"),
            "auc_roc_raw": r["test_raw"].get("auc_roc"),
            "auc_roc_calibrated": r["test_calibrated"].get("auc_roc"),
            "auc_pr_raw": r["test_raw"].get("auc_pr"),
            "auc_pr_calibrated": r["test_calibrated"].get("auc_pr"),
            "ece_raw": r["test_raw"].get("expected_calibration_error"),
            "ece_calibrated": r["test_calibrated"].get("expected_calibration_error"),
            "platt_a": r["platt_a"],
            "platt_b": r["platt_b"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device_src = 'local' if '/Users/ipeglin' in str(ANALYSIS_RESULTS_DIR) else 'IDUN'
    figs_out_dir = plot_config.get_figs_output_dir() / device_src
    csv_out_dir = REPO_ROOT / "confounds" / device_src
    
    figs_out_dir.mkdir(parents=True, exist_ok=True)
    csv_out_dir.mkdir(parents=True, exist_ok=True)
    

    if not ANALYSIS_RESULTS_DIR.exists():
        print(f"Error: Results directory {ANALYSIS_RESULTS_DIR} not found.")
        sys.exit(1)

    print("Loading probabilistic classification reports...")
    df = load_reports(ANALYSIS_RESULTS_DIR)
    if df.empty:
        print("No probabilistic JSON reports found.")
        sys.exit(0)

    summary = build_summary_table(df)
    summary_path = csv_out_dir / f"{get_name()}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"summary written: {summary_path}")

    for fingerprint, fp_df in df.groupby("roi_fingerprint"):
        roi_selection = fp_df["roi_selection"].iloc[0]
        roi_name = fp_df["roi_name"].iloc[0]
        networks_label = fp_df["roi_networks"].iloc[0]
        print(f"\n=== ROI selection: {_roi_title(roi_name, networks_label)} ({fingerprint}) ===")

        for is_mean_val, track_df in fp_df.groupby('is_mean'):
            if is_mean_val and not PLOT_MEAN_ANALYSES:
                continue

            track_label = "Mean Vectors" if is_mean_val else "Per-ROI Vectors"
            print(f"  --- Track: {track_label} ---")
            
            track_roi_name = f"{roi_name}_mean" if is_mean_val else roi_name
            track_roi_sel = f"{roi_selection}_mean" if is_mean_val else roi_selection

            for (run_val, k_val, metric_val), group_df in track_df.groupby(["run", "k", "metric"]):
                group_df = group_df.copy()
                group_df["analysis"] = group_df["analysis"].cat.remove_unused_categories()
                print(f"  run={run_val:02d}, K={k_val}, Metric={metric_val}")

                # 1. Plot all analyses together
                all_dir = figs_out_dir / track_roi_sel / "all"
                all_dir.mkdir(parents=True, exist_ok=True)
                plot_reliability_grid(group_df, all_dir, run_val, k_val, metric_val,
                                      track_roi_sel, track_roi_name, networks_label)
                plot_roc_pr_grid(group_df, all_dir, run_val, k_val, metric_val,
                                 track_roi_sel, track_roi_name, networks_label)
                plot_uncertainty_grid(group_df, all_dir, run_val, k_val, metric_val,
                                      track_roi_sel, track_roi_name, networks_label)
                plot_confusion_grid(group_df, all_dir, run_val, k_val, metric_val,
                                    track_roi_sel, track_roi_name, networks_label)

                # 2. Plot subgroups (e.g. baseline vs task) for smaller figures
                for group_key, target_analyses in ANALYSIS_SUBGROUPS.items():
                    sub_df = group_df[group_df["analysis"].isin(target_analyses)].copy()
                    if sub_df.empty:
                        continue
                    sub_df["analysis"] = sub_df["analysis"].cat.remove_unused_categories()
                    
                    sub_dir = figs_out_dir / track_roi_sel / group_key
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    plot_reliability_grid(sub_df, sub_dir, run_val, k_val, metric_val,
                                          track_roi_sel, track_roi_name, networks_label, group_key)
                    plot_roc_pr_grid(sub_df, sub_dir, run_val, k_val, metric_val,
                                     track_roi_sel, track_roi_name, networks_label, group_key)
                    plot_uncertainty_grid(sub_df, sub_dir, run_val, k_val, metric_val,
                                          track_roi_sel, track_roi_name, networks_label, group_key)
                    plot_confusion_grid(sub_df, sub_dir, run_val, k_val, metric_val,
                                        track_roi_sel, track_roi_name, networks_label, group_key)

    print(f"\nSuccess! Figures saved in: {figs_out_dir}")

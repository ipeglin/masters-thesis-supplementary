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

ANALYSIS_RESULTS_DIR = Path("/Users/ipeglin/Documents/masters_thesis/classifier_results/")
ANALYSIS_RESULTS_DIR = Path("/Volumes/work/classifier_results")  # IDUN network mount

ANALYSIS_ORDER = [
    "Resting Resized",
    "Resting Chunked",
    "Resting Averaged",
    "Task Per Block Resized",
    "Task Per Block",
    "Task Averaged Resized",
    "Task Averaged",
    "Task Concat",
]

ANALYSIS_SUBGROUPS = {
    "resting": [a for a in ANALYSIS_ORDER if "Resting" in a],
    "task": [a for a in ANALYSIS_ORDER if "Task" in a],
}

SOURCE_LABEL = {
    "cwt":     "CWT",
    "hht":     "HHT",
    "hht_roi": "HHT-S",
    "ts":      "Time Series",
}

ANALYSIS_LABEL = {
    "Resting Resized":        "Resize",
    "Resting Chunked":        "Chunk Avg.",
    "Resting Averaged":       "Avg.",
    "Task Per Block Resized": "Block Avg. Resize",
    "Task Per Block":         "Block Avg.",
    "Task Averaged Resized":  "Block Avg. Resize",
    "Task Averaged":          "Block Avg.",
    "Task Concat":            "Concat.",
}

CLASSIFIER_LABEL = {
    "random_forest": "RF",
    "knn":           "KNN",
    "rf":            "RF",
}

ANALYSIS_NAME_MAP = {
    "Baseline Resized":  "Resting Resized",
    "Baseline Chunked":  "Resting Chunked",
    "Baseline Averaged": "Resting Averaged",
}

UNNAMED_ROI_SELECTION = "_unnamed"
NETWORK_SUFFIX_SEP = "__net-"

SHOW_FIGURES = False
PLOT_MEAN_ANALYSES = True
PLOT_HOLDOUT_FIGURES = False
PLOT_COMBINED_KFOLD_FIGURES = True
UNCERTAINTY_BAND = (0.4, 0.6)
N_RELIABILITY_BINS = 10

TABLE_THRESHOLD_CONFIGS = [
    ("holdout_at_youden", "Youden"),
    ("holdout_at_f1",     "F1-opt"),
    ("holdout_at_spec90", r"Spec$\geq$90%"),
]
TABLE_METRICS = [
    ("threshold",        "Threshold",   False),
    ("precision",        "Precision",   True),
    ("sensitivity",      "Recall",      True),
    ("specificity",      "Specificity", True),
    ("balanced_accuracy","Bal. Acc.",   True),
    ("f1",               "F1",          True),
]


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


def _config_dir_name(run_val: int, k_val: int, metric_val: str, reducer_name, reduced_dim) -> str:
    parts = [f"run{run_val:02d}", f"k{k_val}", slugify(metric_val)]
    if reducer_name is not None and reduced_dim is not None:
        parts.append(f"{reducer_name.lower()}{int(reduced_dim)}")
    return "_".join(parts)


def _reducer_label(reducer_name, reduced_dim) -> str:
    if reducer_name is not None and reduced_dim is not None:
        return f" | {reducer_name.upper()}={int(reduced_dim)}"
    return ""


def _reducer_slug(reducer_name, reduced_dim) -> str:
    if reducer_name is not None and reduced_dim is not None:
        return f"_{reducer_name.lower()}{int(reduced_dim)}"
    return ""


# ---------------------------------------------------------------------------
# BootstrapCi helpers
# ---------------------------------------------------------------------------

def _ci_pt(d, key):
    return d.get(key, {}).get("point") if isinstance(d.get(key), dict) else None


def _ci_lo(d, key):
    return d.get(key, {}).get("lo_95") if isinstance(d.get(key), dict) else None


def _ci_hi(d, key):
    return d.get(key, {}).get("hi_95") if isinstance(d.get(key), dict) else None


def _thresh_metric(d: dict, key: str):
    if not isinstance(d, dict):
        return None
    v = d.get(key)
    if v is not None:
        return float(v)
    if key == "f1":
        s = d.get("sensitivity")
        p = d.get("precision") or d.get("ppv")
        if s is not None and p is not None and (float(s) + float(p)) > 0:
            return 2.0 * float(s) * float(p) / (float(s) + float(p))
    if key == "balanced_accuracy":
        s = d.get("sensitivity")
        sp = d.get("specificity")
        if s is not None and sp is not None:
            return (float(s) + float(sp)) / 2.0
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reports(directory: Path, classifier_filter: str = "knn") -> pd.DataFrame:
    base = Path(directory).resolve()
    records = []
    for filepath in base.rglob("*classification.json"):
        # skip k-fold reports — they have a different schema
        if "kfold_classification" in filepath.stem:
            continue
        with open(filepath, "r") as f:
            data = json.load(f)

        if classifier_filter and data.get("classifier") != classifier_filter:
            continue

        run_match = re.search(r"run-(\d+)", filepath.stem)
        run_idx = int(run_match.group(1)) if run_match else 0

        parent = filepath.parent.resolve()
        try:
            rel_parts = parent.relative_to(base).parts
        except ValueError:
            rel_parts = tuple()

        is_subject_stratified = "subject_stratified" in rel_parts
        filtered_parts = [p for p in rel_parts if p != "subject_stratified"]

        if not filtered_parts:
            dir_selection = UNNAMED_ROI_SELECTION
        else:
            dir_selection = filtered_parts[0]

        roi_name, networks_label = parse_roi_dir(dir_selection) or (dir_selection, "")

        roi_fingerprint = data.get("roi_selection_fingerprint") or dir_selection
        if is_subject_stratified:
            roi_fingerprint += "_subject_stratified"
            dir_selection += "_subject_stratified"
            roi_name += " (Subject Stratified)"

        raw_analysis = data.get("analysis", "")
        is_mean_analysis = raw_analysis.endswith("_mean")
        clean_analysis = ANALYSIS_NAME_MAP.get(
            format_label(raw_analysis.replace("_mean", "")),
            format_label(raw_analysis.replace("_mean", ""))
        )

        reducer_name = data.get("reducer_name")
        reduced_dim = data.get("reduced_dim")

        holdout = data.get("holdout", {})
        record = {
            "run": run_idx,
            "analysis": clean_analysis,
            "is_mean": is_mean_analysis,
            "is_subject_stratified": is_subject_stratified,
            "source": format_label(data.get("source")),
            "classifier": data.get("classifier", "knn"),
            "k": data.get("num_neighbors"),
            "metric": format_label(data.get("metric")),
            "reducer_name": reducer_name,
            "reduced_dim": reduced_dim,
            "roi_selection": dir_selection,
            "roi_name": roi_name,
            "roi_networks": networks_label,
            "roi_fingerprint": roi_fingerprint,
            "holdout_predictions": data.get("holdout_predictions", []),
            "holdout_at_0_5": holdout.get("at_0_5", {}),
            "holdout_at_youden": holdout.get("at_youden", {}),
            "holdout_at_f1": holdout.get("at_f1", {}),
            "holdout_at_spec90": holdout.get("at_spec90", {}),
            "holdout_probabilistic": holdout.get("probabilistic", {}),
            "validation_predictions": data.get("validation_predictions", []),
            "validation_at_0_5": data.get("validation", {}).get("at_0_5", {}),
            "validation_at_youden": data.get("validation", {}).get("at_youden", {}),
            "validation_at_f1": data.get("validation", {}).get("at_f1", {}),
            "validation_at_spec90": data.get("validation", {}).get("at_spec90", {}),
            "validation_probabilistic": data.get("validation", {}).get("probabilistic", {}),
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


def load_kfold_reports(directory: Path, classifier_filter: str = "knn") -> pd.DataFrame:
    base = Path(directory).resolve()
    records = []
    for filepath in base.rglob("*kfold_classification.json"):
        with open(filepath, "r") as f:
            data = json.load(f)

        if classifier_filter:
            allowed = (classifier_filter,) if isinstance(classifier_filter, str) else tuple(classifier_filter)
            if data.get("classifier") not in allowed:
                continue

        run_match = re.search(r"run-(\d+)", filepath.stem)
        run_idx = int(run_match.group(1)) if run_match else 0

        # derive a stable key without the run suffix
        file_key = re.sub(r"_run-\d+", "", filepath.stem)

        parent = filepath.parent.resolve()
        try:
            rel_parts = parent.relative_to(base).parts
        except ValueError:
            rel_parts = tuple()

        is_subject_stratified = "subject_stratified" in rel_parts
        filtered_parts = [p for p in rel_parts if p != "subject_stratified"]

        if not filtered_parts:
            dir_selection = UNNAMED_ROI_SELECTION
        else:
            dir_selection = filtered_parts[0]

        roi_name, networks_label = parse_roi_dir(dir_selection) or (dir_selection, "")

        roi_fingerprint = data.get("roi_selection_fingerprint") or dir_selection
        if is_subject_stratified:
            roi_fingerprint += "_subject_stratified"
            dir_selection += "_subject_stratified"
            roi_name += " (Subject Stratified)"

        raw_analysis = data.get("analysis", "")
        is_mean_analysis = raw_analysis.endswith("_mean")
        clean_analysis = ANALYSIS_NAME_MAP.get(
            format_label(raw_analysis.replace("_mean", "")),
            format_label(raw_analysis.replace("_mean", ""))
        )

        k_folds = data.get("k_folds", 0)

        for fold_entry in data.get("folds", []):
            holdout = fold_entry.get("holdout", {})
            prob = holdout.get("probabilistic", {})
            records.append({
                "file_key": file_key,
                "run": run_idx,
                "analysis": clean_analysis,
                "is_mean": is_mean_analysis,
                "is_subject_stratified": is_subject_stratified,
                "source": format_label(data.get("source")),
                "classifier": data.get("classifier", "knn"),
                "k": data.get("num_neighbors"),
                "n_trees": data.get("n_trees"),
                "reducer_name": data.get("reducer_name"),
                "reduced_dim": data.get("reduced_dim"),
                "roi_selection": dir_selection,
                "roi_name": roi_name,
                "roi_networks": networks_label,
                "roi_fingerprint": roi_fingerprint,
                "k_folds": k_folds,
                "fold": fold_entry.get("fold"),
                "n_train": fold_entry.get("n_train"),
                "n_calibration": fold_entry.get("n_calibration"),
                "n_holdout": fold_entry.get("n_holdout"),
                "auc_roc_pt": _ci_pt(prob, "auc_roc"),
                "auc_roc_lo": _ci_lo(prob, "auc_roc"),
                "auc_roc_hi": _ci_hi(prob, "auc_roc"),
                "auc_pr_pt": _ci_pt(prob, "auc_pr"),
                "brier_pt": _ci_pt(prob, "brier"),
                "log_loss_pt": _ci_pt(prob, "log_loss"),
                "bss_pt": _ci_pt(prob, "brier_skill_score"),
                "auc_roc_perm_pvalue": prob.get("auc_roc_perm_pvalue"),
                "calibration_slope": prob.get("calibration_slope"),
                "calibration_intercept": prob.get("calibration_intercept"),
                "ece": prob.get("expected_calibration_error"),
                "calibration_bins": prob.get("calibration_bins", []),
                "holdout_at_0_5": holdout.get("at_0_5", {}),
                "holdout_at_youden": holdout.get("at_youden", {}),
                "holdout_at_f1": holdout.get("at_f1", {}),
                "holdout_at_spec90": holdout.get("at_spec90", {}),
                "fold_predictions": fold_entry.get("validation_predictions", []),
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    in_df = df["analysis"].unique()
    missing = [x for x in in_df if x not in ANALYSIS_ORDER]
    df["analysis"] = pd.Categorical(
        df["analysis"], categories=list(ANALYSIS_ORDER) + missing, ordered=True
    )
    df = df.sort_values(["analysis", "source", "fold"])
    return df


# ---------------------------------------------------------------------------
# Curve helpers (hand-rolled — avoids sklearn)
# ---------------------------------------------------------------------------

def roc_curve(y_true: np.ndarray, scores: np.ndarray):
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
    recall = np.r_[0.0, recall]
    precision = np.r_[precision[0], precision]
    return recall, precision


def auc_trapezoidal(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


# ---------------------------------------------------------------------------
# Layout helper
# ---------------------------------------------------------------------------

def _layout(group_df):
    analyses = list(group_df["analysis"].cat.categories)
    sources = sorted(group_df["source"].unique())
    stack_sources = len(analyses) == 1
    n_rows = len(sources) if stack_sources else len(analyses)
    n_cols = len(analyses) if stack_sources else len(sources)
    active_grid = np.zeros((n_rows, n_cols), dtype=bool)
    for idx_a, analysis in enumerate(analyses):
        for idx_s, source in enumerate(sources):
            r = idx_s if stack_sources else idx_a
            c = idx_a if stack_sources else idx_s
            sub = group_df[(group_df["analysis"] == analysis) & (group_df["source"] == source)]
            if not sub.empty and sub.iloc[0].get("holdout_predictions"):
                active_grid[r, c] = True
    return analyses, sources, stack_sources, n_rows, n_cols, active_grid


# ---------------------------------------------------------------------------
# Plots — split reports
# ---------------------------------------------------------------------------

OPERATING_POINTS = [
    ("holdout_at_0_5",    "Fixed",    "purple",      "x"),
    ("holdout_at_youden", "Youden",   "forestgreen", "s"),
    ("holdout_at_f1",     "F1-opt",   "darkorange",  "D"),
    ("holdout_at_spec90", r"Spec$\geq$90%", "crimson",     "^"),
]


def plot_roc_pr_grid(group_df: pd.DataFrame, save_dir: Path,
                     run_val: int, k_val: int, metric_val: str,
                     roi_selection: str, roi_name: str, networks_label: str,
                     filename_suffix: str = "",
                     reducer_name=None, reduced_dim=None) -> None:
    analyses, sources, stack_sources, n_rows, n_cols, active_grid = _layout(group_df)
    if not sources or not analyses:
        return

    fig, axes = plt.subplots(n_rows, 3 * n_cols,
                             figsize=(4.2 * 3 * n_cols, 3.8 * n_rows),
                             squeeze=False)

    reducer_str = _reducer_label(reducer_name, reduced_dim)
    reducer_slug_str = _reducer_slug(reducer_name, reduced_dim)
    fig.suptitle(
        f"ROC, PR \\& Reliability  -  ROI={_roi_title(roi_name, networks_label)} | "
        f"Run {run_val:02d} | K={k_val} | {metric_val}{reducer_str}",
        y=1.01,
    )

    for idx_a, analysis in enumerate(analyses):
        for idx_s, source in enumerate(sources):
            r = idx_s if stack_sources else idx_a
            c = idx_a if stack_sources else idx_s
            ax_roc = axes[r, 3 * c]
            ax_pr  = axes[r, 3 * c + 1]
            ax_rel = axes[r, 3 * c + 2]

            row = group_df[(group_df["analysis"] == analysis) & (group_df["source"] == source)]
            if row.empty:
                ax_roc.set_visible(False); ax_pr.set_visible(False); ax_rel.set_visible(False)
                continue
            row0 = row.iloc[0]
            # use validation predictions for ROC/PR — thresholds were tuned here
            preds = row0.get("validation_predictions") or row0["holdout_predictions"]
            if not preds:
                ax_roc.set_visible(False); ax_pr.set_visible(False); ax_rel.set_visible(False)
                continue

            using_validation = bool(row0.get("validation_predictions"))
            y = np.array([p["y_true"] for p in preds])
            prob_src = row0["validation_probabilistic"] if using_validation else row0["holdout_probabilistic"]
            prob = prob_src if isinstance(prob_src, dict) and prob_src else row0["holdout_probabilistic"]

            s = np.array([p["p1"] for p in preds], dtype=float)
            fpr, tpr = roc_curve(y, s)
            rec, prec = pr_curve(y, s)

            auc_roc_ci = prob.get("auc_roc", {})
            auc_pr_ci = prob.get("auc_pr", {})
            pt_roc = auc_roc_ci.get("point", auc_trapezoidal(fpr, tpr)) if isinstance(auc_roc_ci, dict) else auc_trapezoidal(fpr, tpr)
            lo_roc = auc_roc_ci.get("lo_95", float("nan")) if isinstance(auc_roc_ci, dict) else float("nan")
            hi_roc = auc_roc_ci.get("hi_95", float("nan")) if isinstance(auc_roc_ci, dict) else float("nan")
            pt_pr  = auc_pr_ci.get("point", auc_trapezoidal(rec, prec)) if isinstance(auc_pr_ci, dict) else auc_trapezoidal(rec, prec)
            lo_pr  = auc_pr_ci.get("lo_95", float("nan")) if isinstance(auc_pr_ci, dict) else float("nan")
            hi_pr  = auc_pr_ci.get("hi_95", float("nan")) if isinstance(auc_pr_ci, dict) else float("nan")
            pval   = prob.get("auc_roc_perm_pvalue", float("nan"))
            split_label = "Val" if using_validation else "Holdout"

            ax_roc.plot(fpr, tpr, color=plot_config.RESULTS_PALETTE[0], linewidth=1.5,
                        label=f"{split_label} AUC={pt_roc:.3f} [{lo_roc:.3f}, {hi_roc:.3f}]  p={pval:.3f}")
            ax_pr.plot(rec, prec, color=plot_config.RESULTS_PALETTE[0], linewidth=1.5,
                       label=f"{split_label} AUC={pt_pr:.3f} [{lo_pr:.3f}, {hi_pr:.3f}]")

            def _op(threshold):
                pred = (s >= threshold).astype(int)
                tp = int(np.sum((pred == 1) & (y == 1)))
                fp = int(np.sum((pred == 1) & (y == 0)))
                fn = int(np.sum((pred == 0) & (y == 1)))
                tn = int(np.sum((pred == 0) & (y == 0)))
                tpr_ = tp / max(tp + fn, 1)
                fpr_ = fp / max(tn + fp, 1)
                prec_ = tp / max(tp + fp, 1)
                return tpr_, fpr_, prec_

            # operating point keys mapped to validation equivalents when available
            val_op_map = {
                "holdout_at_0_5":    "validation_at_0_5",
                "holdout_at_youden": "validation_at_youden",
                "holdout_at_f1":     "validation_at_f1",
                "holdout_at_spec90": "validation_at_spec90",
            }
            for rec_key, label, color, marker in OPERATING_POINTS:
                op_key = val_op_map.get(rec_key, rec_key) if using_validation else rec_key
                t = row0.get(op_key, {}).get("threshold") if isinstance(row0.get(op_key), dict) else None
                if t is None:
                    t = row0[rec_key].get("threshold")
                if t is None:
                    continue
                tpr_, fpr_, prec_ = _op(t)
                ax_roc.scatter([fpr_], [tpr_], color=color, marker=marker, s=50,
                               zorder=5, label=f"{label} t={t:.3f}")
                ax_pr.scatter([tpr_], [prec_], color=color, marker=marker, s=50,
                              zorder=5, label=f"{label} t={t:.3f}")

            ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.4)
            ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
            ax_roc.set_title(f"ROC [{split_label}] | {analysis} | {source}", fontsize=9)

            ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.05)
            ax_pr.set_title(f"PR [{split_label}] | {analysis} | {source}", fontsize=9)

            # reliability panel
            ax_rel.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect")
            bins = prob.get("calibration_bins", [])
            xs = [b["mean_pred"] for b in bins if b.get("count", 0) > 0 and b.get("mean_pred") is not None]
            ys = [b["frac_positive"] for b in bins if b.get("count", 0) > 0 and b.get("mean_pred") is not None]
            sizes_arr = np.array([b["count"] for b in bins if b.get("count", 0) > 0 and b.get("mean_pred") is not None], dtype=float)
            brier_pt = _ci_pt(prob, "brier")
            bss_pt   = _ci_pt(prob, "brier_skill_score")
            slope    = prob.get("calibration_slope", float("nan"))
            intercept = prob.get("calibration_intercept", float("nan"))
            ece      = prob.get("expected_calibration_error", float("nan"))
            if xs:
                sizes_arr = 25 + 175 * sizes_arr / sizes_arr.max()
                brier_str = f"{brier_pt:.3f}" if brier_pt is not None else "nan"
                bss_str   = f"{bss_pt:.3f}" if bss_pt is not None else "nan"
                ax_rel.scatter(xs, ys, s=sizes_arr, color=plot_config.RESULTS_PALETTE[0], alpha=0.7,
                               label=f"Brier={brier_str}  BSS={bss_str}  slope={slope:.3f}  ECE={ece:.3f}")
                ax_rel.plot(xs, ys, color=plot_config.RESULTS_PALETTE[0], alpha=0.35, linewidth=1.0)
            ax_rel.set_xlim(0, 1); ax_rel.set_ylim(0, 1)
            ax_rel.set_title(f"Reliability | {analysis} | {source}", fontsize=9)

            is_bottom = (r == np.where(active_grid[:, c])[0][-1])
            is_left   = (c == np.where(active_grid[r, :])[0][0])
            if is_bottom:
                ax_roc.set_xlabel("False positive rate")
                ax_pr.set_xlabel("Recall")
                ax_rel.set_xlabel(r"Mean predicted $P(\mathrm{anhedonic})$")
                for ax in (ax_roc, ax_pr, ax_rel):
                    ax.tick_params(labelbottom=True)
            if is_left:
                ax_roc.set_ylabel("True positive rate")
                ax_pr.set_ylabel("Precision")
                ax_rel.set_ylabel("Fraction anhedonic")

            ax_roc.legend(loc="lower right", fontsize=7)
            ax_pr.legend(loc="lower left",  fontsize=7)
            ax_rel.legend(loc="upper left", fontsize=7)

    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    filename = (
        f"knn_roc_pr_reliability_roi-{slugify(roi_selection)}_run{run_val:02d}_k{k_val}"
        f"_{slugify(metric_val)}{reducer_slug_str}{suffix_str}.pdf"
    )
    fig.savefig(save_dir / filename)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_uncertainty_grid(group_df: pd.DataFrame, save_dir: Path,
                          run_val: int, k_val: int, metric_val: str,
                          roi_selection: str, roi_name: str, networks_label: str,
                          filename_suffix: str = "",
                          reducer_name=None, reduced_dim=None) -> None:
    analyses, sources, stack_sources, n_rows, n_cols, active_grid = _layout(group_df)
    if not sources or not analyses:
        return

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 3.0 * n_rows),
                             squeeze=False)

    reducer_str = _reducer_label(reducer_name, reduced_dim)
    reducer_slug_str = _reducer_slug(reducer_name, reduced_dim)
    fig.suptitle(
        f"Per-sample uncertainty - ROI={_roi_title(roi_name, networks_label)} | "
        f"Run {run_val:02d} | K={k_val} | {metric_val}{reducer_str}",
        y=1.01,
    )

    for idx_a, analysis in enumerate(analyses):
        for idx_s, source in enumerate(sources):
            r = idx_s if stack_sources else idx_a
            c = idx_a if stack_sources else idx_s
            ax = axes[r, c]
            row = group_df[(group_df["analysis"] == analysis) & (group_df["source"] == source)]
            if row.empty:
                ax.set_visible(False); continue
            preds = row.iloc[0]["holdout_predictions"]
            if not preds:
                ax.set_visible(False); continue

            df_p = pd.DataFrame(preds).sort_values("p1").reset_index(drop=True)
            colors = [plot_config.cohort_color(y) for y in df_p["y_true"]]
            ax.scatter(range(len(df_p)), df_p["p1"], c=colors, s=14, alpha=0.8)
            ax.axhspan(*UNCERTAINTY_BAND, color="gray", alpha=0.15,
                       label=f"uncertainty band {UNCERTAINTY_BAND}")
            ax.axhline(0.5, color="black", linestyle="--", linewidth=0.6, alpha=0.5,
                       label="t=0.50")

            row0 = row.iloc[0]
            for rec_key, label, color, ls in [
                ("holdout_at_youden", "Youden",   "forestgreen", "-."),
                ("holdout_at_f1",     "F1-opt",   "darkorange",  ":"),
                ("holdout_at_spec90", r"Spec$\geq$90%", "crimson",    (0, (3, 1, 1, 1))),
            ]:
                t = row0[rec_key].get("threshold")
                if t is not None:
                    ax.axhline(t, color=color, linestyle=ls, linewidth=1.0,
                               alpha=0.8, label=f"{label} t={t:.3f}")

            in_band = df_p[(df_p["p1"] >= UNCERTAINTY_BAND[0]) &
                           (df_p["p1"] <= UNCERTAINTY_BAND[1])]
            ax.set_title(
                f"{analysis} | {source} | n uncertain: {len(in_band)} / {len(df_p)}",
                fontsize=10,
            )
            ax.set_ylim(0, 1)

            is_bottom = (r == np.where(active_grid[:, c])[0][-1])
            is_left   = (c == np.where(active_grid[r, :])[0][0])
            if is_bottom:
                ax.set_xlabel("Samples ranked by P(anhedonic)")
            if is_left:
                ax.set_ylabel(r"$P(\mathrm{anhedonic})$")
            ax.legend(loc="upper left", fontsize=8)

    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    filename = (
        f"knn_uncertainty_roi-{slugify(roi_selection)}_run{run_val:02d}_k{k_val}"
        f"_{slugify(metric_val)}{reducer_slug_str}{suffix_str}.pdf"
    )
    fig.savefig(save_dir / filename)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_confusion_grid(group_df: pd.DataFrame, save_dir: Path,
                        run_val: int, k_val: int, metric_val: str,
                        roi_selection: str, roi_name: str, networks_label: str,
                        filename_suffix: str = "",
                        reducer_name=None, reduced_dim=None) -> None:
    analyses, sources, stack_sources, n_rows, n_cols, active_grid = _layout(group_df)
    if not sources or not analyses:
        return

    fig, axes = plt.subplots(n_rows, 2 * n_cols,
                             figsize=(6.0 * 2 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    reducer_str = _reducer_label(reducer_name, reduced_dim)
    reducer_slug_str = _reducer_slug(reducer_name, reduced_dim)
    fig.suptitle(
        f"Confusion Matrices - ROI={_roi_title(roi_name, networks_label)} | "
        f"Run {run_val:02d} | K={k_val} | {metric_val}{reducer_str}",
        y=1.01,
    )

    for idx_a, analysis in enumerate(analyses):
        for idx_s, source in enumerate(sources):
            r = idx_s if stack_sources else idx_a
            c = idx_a if stack_sources else idx_s
            ax_05 = axes[r, 2 * c]
            ax_youden = axes[r, 2 * c + 1]
            row = group_df[(group_df["analysis"] == analysis) & (group_df["source"] == source)]
            if row.empty:
                ax_05.set_visible(False); ax_youden.set_visible(False); continue
            preds = row.iloc[0]["holdout_predictions"]
            if not preds:
                ax_05.set_visible(False); ax_youden.set_visible(False); continue

            y_true = np.array([p["y_true"] for p in preds])
            p1 = np.array([p["p1"] for p in preds])
            youden = row.iloc[0]["holdout_at_youden"].get("threshold", 0.5)

            for threshold, ax, title_prefix in [
                (0.5,    ax_05,     "t=0.5"),
                (youden, ax_youden, f"t={youden:.3f}"),
            ]:
                y_pred = (p1 >= threshold).astype(int)
                tp = int(np.sum((y_pred == 1) & (y_true == 1)))
                fp = int(np.sum((y_pred == 1) & (y_true == 0)))
                fn = int(np.sum((y_pred == 0) & (y_true == 1)))
                tn = int(np.sum((y_pred == 0) & (y_true == 0)))
                cm = np.array([[tn, fp], [fn, tp]])
                acc  = (tp + tn) / max(1, len(y_true))
                sens = tp / max(1, tp + fn)
                spec = tn / max(1, tn + fp)
                sns.heatmap(cm, annot=True, fmt="d", cmap=plot_config.CMAP_CONF, cbar=False,
                            xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"], ax=ax)
                ax.set_title(
                    f"{analysis} | {source}\n"
                    f"{title_prefix} | Acc: {acc:.3f} | Sens: {sens:.3f} | Spec: {spec:.3f}",
                    fontsize=10,
                )
                if r == np.where(active_grid[:, c])[0][-1]:
                    ax.set_xlabel("Predicted")
                if c == 0 and ax == ax_05:
                    ax.set_ylabel("True")
                else:
                    ax.set_ylabel("")

    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    filename = (
        f"knn_cm_roi-{slugify(roi_selection)}_run{run_val:02d}_k{k_val}"
        f"_{slugify(metric_val)}{reducer_slug_str}{suffix_str}.pdf"
    )
    fig.savefig(save_dir / filename)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_holdout_threshold_table(
    group_df: pd.DataFrame,
    save_dir: Path,
    run_val: int,
    param_label: str,
    param_slug: str,
    roi_selection: str,
    roi_name: str,
    networks_label: str,
    filename_prefix: str,
    filename_suffix: str = "",
    reducer_name=None,
    reduced_dim=None,
) -> None:
    analyses = list(group_df["analysis"].cat.categories)
    sources = sorted(group_df["source"].unique())

    rows_data = []
    for analysis in analyses:
        for source in sources:
            sub = group_df[(group_df["analysis"] == analysis) & (group_df["source"] == source)]
            if sub.empty:
                continue
            row0 = sub.iloc[0]
            entry = {"label": f"{analysis} | {source}"}
            has_data = False
            for thresh_key, _ in TABLE_THRESHOLD_CONFIGS:
                d = row0.get(thresh_key) or {}
                for metric_key, _, _ in TABLE_METRICS:
                    v = _thresh_metric(d, metric_key)
                    entry[(thresh_key, metric_key)] = v
                    if v is not None:
                        has_data = True
            if has_data:
                rows_data.append(entry)

    if not rows_data:
        return

    n_rows = len(rows_data)
    n_metrics = len(TABLE_METRICS)
    metric_cols = [(mi, mk, hb) for mi, (mk, _lbl, hb) in enumerate(TABLE_METRICS) if mi > 0]
    header_colors = ["#DDDDDD", "#EEEEEE", "#F5F5F5"]
    highlight_bg = "#F0E442"

    reducer_str = _reducer_label(reducer_name, reduced_dim)
    reducer_slug_str = _reducer_slug(reducer_name, reduced_dim)

    fig, axes = plt.subplots(1, len(TABLE_THRESHOLD_CONFIGS),
                             figsize=(6.5 * len(TABLE_THRESHOLD_CONFIGS),
                                      max(3.5, 0.55 * n_rows + 2.5)))
    fig.suptitle(
        f"Hold-out threshold performance  |  ROI={_roi_title(roi_name, networks_label)} | "
        f"Run {run_val:02d} | {param_label}{reducer_str}",
        fontsize=10, y=1.02,
    )

    row_labels = [r["label"] for r in rows_data]

    for ti, (thresh_key, thresh_label) in enumerate(TABLE_THRESHOLD_CONFIGS):
        ax = axes[ti]
        ax.axis("off")
        ax.set_title(thresh_label, fontsize=11, fontweight="bold", pad=8)

        col_labels = [lbl for _, lbl, _ in TABLE_METRICS]
        cell_texts = [
            [
                f"{r[(thresh_key, mk)]:.3f}" if r[(thresh_key, mk)] is not None else "|"
                for mk, _, _ in TABLE_METRICS
            ]
            for r in rows_data
        ]

        best_rows = {}
        for mi, metric_key, higher_better in metric_cols:
            vals = [(ri, r[(thresh_key, metric_key)])
                    for ri, r in enumerate(rows_data)
                    if r[(thresh_key, metric_key)] is not None]
            if vals:
                best_rows[mi] = max(vals, key=lambda x: x[1] if higher_better else -x[1])[0]

        tbl = ax.table(
            cellText=cell_texts,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            rowLoc="right",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.5)
        tbl.auto_set_column_width(list(range(n_metrics)))

        for ci in range(n_metrics):
            tbl[0, ci].set_facecolor(header_colors[ti])
            tbl[0, ci].set_text_props(fontweight="bold")

        for mi, _mk, _ in metric_cols:
            if mi in best_rows:
                ri = best_rows[mi]
                tbl[ri + 1, mi].set_facecolor(highlight_bg)
                tbl[ri + 1, mi].set_text_props(fontweight="bold")

    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    filename = (
        f"{filename_prefix}_holdout_table_roi-{slugify(roi_selection)}"
        f"_run{run_val:02d}_{slugify(param_slug)}{reducer_slug_str}{suffix_str}.pdf"
    )
    fig.savefig(save_dir / filename, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        prob = r["holdout_probabilistic"]
        rows.append({
            "run": r["run"],
            "analysis": r["analysis"],
            "is_subject_stratified": r["is_subject_stratified"],
            "source": r["source"],
            "classifier": r.get("classifier", "knn"),
            "k": r["k"],
            "metric": r["metric"],
            "roi_selection": r["roi_selection"],
            "reducer_name": r["reducer_name"],
            "reduced_dim": r["reduced_dim"],
            "holdout_acc_0_5": r["holdout_at_0_5"].get("accuracy"),
            "holdout_acc_youden": r["holdout_at_youden"].get("accuracy"),
            "holdout_acc_f1": r["holdout_at_f1"].get("accuracy"),
            "holdout_acc_spec90": r["holdout_at_spec90"].get("accuracy"),
            "holdout_sens_spec90": r["holdout_at_spec90"].get("sensitivity"),
            "holdout_spec_spec90": r["holdout_at_spec90"].get("specificity"),
            "youden_threshold": r["holdout_at_youden"].get("threshold"),
            "f1_threshold": r["holdout_at_f1"].get("threshold"),
            "spec90_threshold": r["holdout_at_spec90"].get("threshold"),
            "brier": _ci_pt(prob, "brier"),
            "brier_lo_95": _ci_lo(prob, "brier"),
            "brier_hi_95": _ci_hi(prob, "brier"),
            "bss": _ci_pt(prob, "brier_skill_score"),
            "auc_roc": _ci_pt(prob, "auc_roc"),
            "auc_roc_lo_95": _ci_lo(prob, "auc_roc"),
            "auc_roc_hi_95": _ci_hi(prob, "auc_roc"),
            "auc_pr": _ci_pt(prob, "auc_pr"),
            "auc_pr_lo_95": _ci_lo(prob, "auc_pr"),
            "auc_pr_hi_95": _ci_hi(prob, "auc_pr"),
            "auc_roc_perm_pvalue": prob.get("auc_roc_perm_pvalue"),
            "calibration_slope": prob.get("calibration_slope"),
            "calibration_intercept": prob.get("calibration_intercept"),
            "ece": prob.get("expected_calibration_error"),
            "log_loss": _ci_pt(prob, "log_loss"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots — k-fold reports
# ---------------------------------------------------------------------------

def plot_kfold_metrics_grid(kfold_df: pd.DataFrame, save_dir: Path,
                             roi_selection: str, roi_name: str, networks_label: str,
                             filename_suffix: str = "",
                             filename_prefix: str = "knn") -> None:
    analyses = list(kfold_df["analysis"].cat.categories)
    sources = sorted(kfold_df["source"].unique())
    if not sources or not analyses:
        return

    stack_sources = len(analyses) == 1
    n_rows = len(sources) if stack_sources else len(analyses)
    n_cols = len(analyses) if stack_sources else len(sources)

    fig, axes = plt.subplots(3 * n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.0 * 3 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"{filename_prefix.upper()} K-fold metrics | ROI={_roi_title(roi_name, networks_label)}",
        y=1.01,
    )

    for idx_a, analysis in enumerate(analyses):
        for idx_s, source in enumerate(sources):
            r = idx_s if stack_sources else idx_a
            c = idx_a if stack_sources else idx_s
            ax_roc   = axes[3 * r,     c]
            ax_pr    = axes[3 * r + 1, c]
            ax_brier = axes[3 * r + 2, c]

            sub = kfold_df[(kfold_df["analysis"] == analysis) & (kfold_df["source"] == source)]
            if sub.empty:
                ax_roc.set_visible(False); ax_pr.set_visible(False); ax_brier.set_visible(False)
                continue

            for ax, col, ylabel, invert in [
                (ax_roc,   "auc_roc_pt", "AUC-ROC", False),
                (ax_pr,    "auc_pr_pt",  "AUC-PR",  False),
                (ax_brier, "brier_pt",   "Brier",   True),
            ]:
                vals = sub[col].dropna().values
                folds = sub["fold"].values[:len(vals)]
                ax.scatter(folds, vals, color=plot_config.RESULTS_PALETTE[0], s=30, zorder=3)
                mean_v = float(np.mean(vals))
                std_v  = float(np.std(vals))
                ax.axhline(mean_v, color=plot_config.RESULTS_PALETTE[4], linewidth=1.2,
                           label=f"mean={mean_v:.3f} std={std_v:.3f}")
                ax.axhspan(mean_v - std_v, mean_v + std_v, color=plot_config.RESULTS_PALETTE[4], alpha=0.12)
                ax.set_title(f"{ylabel} | {analysis} | {source}", fontsize=9)
                ax.legend(loc="best", fontsize=7)
                ax.set_ylabel(ylabel)
            ax_brier.set_xlabel("Fold")

    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    filename = f"{filename_prefix}_kfold_metrics_roi-{slugify(roi_selection)}{suffix_str}.pdf"
    fig.savefig(save_dir / filename)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_kfold_reliability_grid(kfold_df: pd.DataFrame, save_dir: Path,
                                 roi_selection: str, roi_name: str, networks_label: str,
                                 filename_suffix: str = "",
                                 filename_prefix: str = "knn") -> None:
    analyses = list(kfold_df["analysis"].cat.categories)
    sources = sorted(kfold_df["source"].unique())
    if not sources or not analyses:
        return

    stack_sources = len(analyses) == 1
    n_rows = len(sources) if stack_sources else len(analyses)
    n_cols = len(analyses) if stack_sources else len(sources)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.8 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    fig.suptitle(
        f"{filename_prefix.upper()} K-fold reliability | ROI={_roi_title(roi_name, networks_label)}",
        y=1.01,
    )

    for idx_a, analysis in enumerate(analyses):
        for idx_s, source in enumerate(sources):
            r = idx_s if stack_sources else idx_a
            c = idx_a if stack_sources else idx_s
            ax = axes[r, c]
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)

            sub = kfold_df[(kfold_df["analysis"] == analysis) & (kfold_df["source"] == source)]
            if sub.empty:
                ax.set_visible(False); continue

            # average calibration bins across folds bin-by-bin
            all_bins = [row for row in sub["calibration_bins"] if row]
            if all_bins:
                max_len = max(len(b) for b in all_bins)
                mean_pred_sum = np.zeros(max_len)
                frac_pos_sum  = np.zeros(max_len)
                count_n       = np.zeros(max_len)
                weight_sum    = np.zeros(max_len)
                for fold_bins in all_bins:
                    for i, b in enumerate(fold_bins):
                        if b.get("count", 0) > 0 and b.get("mean_pred") is not None:
                            mean_pred_sum[i] += b["mean_pred"] * b["count"]
                            frac_pos_sum[i]  += b["frac_positive"] * b["count"]
                            weight_sum[i]    += b["count"]
                            count_n[i]       += 1
                xs, ys, sz = [], [], []
                for i in range(max_len):
                    if weight_sum[i] > 0:
                        xs.append(mean_pred_sum[i] / weight_sum[i])
                        ys.append(frac_pos_sum[i] / weight_sum[i])
                        sz.append(weight_sum[i])
                if xs:
                    sz_arr = np.array(sz, dtype=float)
                    sz_arr = 25 + 175 * sz_arr / sz_arr.max()
                    mean_brier = float(sub["brier_pt"].mean())
                    mean_bss   = float(sub["bss_pt"].mean())
                    mean_slope = float(sub["calibration_slope"].mean())
                    mean_inter = float(sub["calibration_intercept"].mean())
                    ax.scatter(xs, ys, s=sz_arr, color=plot_config.RESULTS_PALETTE[0], alpha=0.7,
                               label=(f"Brier={mean_brier:.3f}  BSS={mean_bss:.3f}\n"
                                      f"slope={mean_slope:.3f}  intercept={mean_inter:.3f}"))
                    ax.plot(xs, ys, color=plot_config.RESULTS_PALETTE[0], alpha=0.35, linewidth=1.0)

            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f"{analysis} | {source}", fontsize=10)
            ax.legend(loc="upper left", fontsize=8)

            if r == n_rows - 1:
                ax.set_xlabel(r"Mean predicted $P(\mathrm{anhedonic})$")
            if c == 0:
                ax.set_ylabel("Empirical fraction anhedonic")

    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    filename = f"{filename_prefix}_kfold_reliability_roi-{slugify(roi_selection)}{suffix_str}.pdf"
    fig.savefig(save_dir / filename)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_kfold_roc_pr_grid(kfold_df: pd.DataFrame, save_dir: Path,
                            roi_selection: str, roi_name: str, networks_label: str,
                            filename_prefix: str, filename_suffix: str = "") -> None:
    analyses = list(kfold_df["analysis"].cat.categories)
    sources = sorted(kfold_df["source"].unique())
    if not sources or not analyses:
        return

    stack_sources = len(analyses) == 1
    n_rows = len(sources) if stack_sources else len(analyses)
    n_cols = len(analyses) if stack_sources else len(sources)

    fig, axes = plt.subplots(n_rows, 2 * n_cols,
                             figsize=(4.5 * 2 * n_cols, 3.8 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"{filename_prefix.upper()} K-fold validation ROC \\& PR | ROI={_roi_title(roi_name, networks_label)}",
        y=1.04,
    )

    op_specs = [
        ("holdout_at_youden", "Youden",   "forestgreen", "s"),
        ("holdout_at_f1",     "F1-opt",   "darkorange",  "D"),
        ("holdout_at_spec90", r"Spec$\geq$90%", "crimson",     "^"),
    ]

    for idx_a, analysis in enumerate(analyses):
        for idx_s, source in enumerate(sources):
            r = idx_s if stack_sources else idx_a
            c = idx_a if stack_sources else idx_s
            ax_roc = axes[r, 2 * c]
            ax_pr  = axes[r, 2 * c + 1]

            sub = kfold_df[(kfold_df["analysis"] == analysis) & (kfold_df["source"] == source)]
            if sub.empty:
                ax_roc.set_visible(False); ax_pr.set_visible(False)
                continue

            all_fpr, all_tpr, all_rec, all_prec = [], [], [], []
            for _, fold_row in sub.iterrows():
                preds = fold_row.get("fold_predictions") or []
                if preds:
                    y = np.array([p["y_true"] for p in preds])
                    s = np.array([p["p1"] for p in preds], dtype=float)
                    fpr_, tpr_ = roc_curve(y, s)
                    rec_, prec_ = pr_curve(y, s)
                    ax_roc.plot(fpr_, tpr_, color=plot_config.RESULTS_PALETTE[0], linewidth=0.7, alpha=0.3)
                    ax_pr.plot(rec_, prec_, color=plot_config.RESULTS_PALETTE[0], linewidth=0.7, alpha=0.3)
                    all_fpr.append(fpr_); all_tpr.append(tpr_)
                    all_rec.append(rec_); all_prec.append(prec_)

            mean_auc_roc = float(sub["auc_roc_pt"].dropna().mean()) if sub["auc_roc_pt"].notna().any() else float("nan")
            mean_auc_pr  = float(sub["auc_pr_pt"].dropna().mean()) if sub["auc_pr_pt"].notna().any() else float("nan")

            if all_fpr:
                fpr_grid = np.linspace(0, 1, 200)
                mean_tpr = np.array([np.interp(fpr_grid, fp, tp)
                                      for fp, tp in zip(all_fpr, all_tpr)]).mean(axis=0)
                ax_roc.plot(fpr_grid, mean_tpr, color=plot_config.RESULTS_PALETTE[0], linewidth=2.0,
                            label=f"Mean AUC={mean_auc_roc:.3f}")

                rec_grid = np.linspace(0, 1, 200)
                mean_prec = np.array([np.interp(rec_grid, rc, pr)
                                       for rc, pr in zip(all_rec, all_prec)]).mean(axis=0)
                ax_pr.plot(rec_grid, mean_prec, color=plot_config.RESULTS_PALETTE[0], linewidth=2.0,
                           label=f"Mean AUC={mean_auc_pr:.3f}")
            else:
                std_roc = float(sub["auc_roc_pt"].dropna().std()) if sub["auc_roc_pt"].notna().sum() > 1 else float("nan")
                std_pr  = float(sub["auc_pr_pt"].dropna().std()) if sub["auc_pr_pt"].notna().sum() > 1 else float("nan")
                ax_roc.text(0.5, 0.5, f"AUC={mean_auc_roc:.3f} $\\pm$ {std_roc:.3f}",
                            ha="center", va="center", fontsize=11, transform=ax_roc.transAxes,
                            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
                ax_pr.text(0.5, 0.5, f"AUC={mean_auc_pr:.3f} $\\pm$ {std_pr:.3f}",
                           ha="center", va="center", fontsize=11, transform=ax_pr.transAxes,
                           bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

            for col, label, color, marker in op_specs:
                sens_vals, spec_vals, prec_vals = [], [], []
                for _, fold_row in sub.iterrows():
                    d = fold_row.get(col) or {}
                    sv = _thresh_metric(d, "sensitivity")
                    spv = _thresh_metric(d, "specificity")
                    pv = _thresh_metric(d, "precision") or _thresh_metric(d, "ppv")
                    if sv is not None and spv is not None:
                        sens_vals.append(sv); spec_vals.append(spv)
                    if pv is not None:
                        prec_vals.append(pv)
                if sens_vals:
                    ax_roc.scatter([1.0 - float(np.mean(spec_vals))], [float(np.mean(sens_vals))],
                                   color=color, marker=marker, s=60, zorder=5, label=label)
                if sens_vals and len(prec_vals) == len(sens_vals):
                    ax_pr.scatter([float(np.mean(sens_vals))], [float(np.mean(prec_vals))],
                                  color=color, marker=marker, s=60, zorder=5, label=label)

            ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.4)
            ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
            ax_roc.set_title(f"ROC | {analysis} | {source}", fontsize=9)
            ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.05)
            ax_pr.set_title(f"PR | {analysis} | {source}", fontsize=9)

            ax_roc.set_xlabel("False positive rate")
            ax_pr.set_xlabel("Recall")
            if c == 0:
                ax_roc.set_ylabel("True positive rate")
                ax_pr.set_ylabel("Precision")

            ax_roc.legend(loc="lower right", fontsize=7)
            ax_pr.legend(loc="lower left", fontsize=7)

    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    filename = f"{filename_prefix}_kfold_roc_pr_roi-{slugify(roi_selection)}{suffix_str}.pdf"
    fig.savefig(save_dir / filename)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


_KFOLD_HARD_THRESH_CONFIGS = [
    ("holdout_at_0_5",    "t=0.50"),
    ("holdout_at_youden", "Youden"),
    ("holdout_at_f1",     "F1-opt"),
    ("holdout_at_spec90", r"Spec$\geq$90%"),
]
_KFOLD_HARD_METRICS = [
    "threshold", "sensitivity", "precision", "specificity", "f1", "balanced_accuracy",
]

_KFOLD_METRIC_LABELS = {
    "threshold":        "Thr.",
    "sensitivity":      "Recall",
    "precision":        "Precision",
    "specificity":      "Specificity",
    "f1":               "F1",
    "balanced_accuracy":"Bal. Acc.",
}

CRITERION_LABELS = {
    "holdout_at_0_5":    "t = 0.50",
    "holdout_at_youden": "Youden",
    "holdout_at_f1":     "F1-opt",
    "holdout_at_spec90": r"Spec.$~\geq 90\%$",
}


def _fmt(mean, std) -> str:
    return _fmt_paren(mean, std, 3)


def build_kfold_summary_table(kfold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["auc_roc_pt", "auc_pr_pt", "brier_pt", "log_loss_pt",
                   "bss_pt", "auc_roc_perm_pvalue", "calibration_slope",
                   "calibration_intercept", "ece"]
    group_keys = ["file_key", "run", "analysis", "is_mean", "is_subject_stratified",
                  "source", "classifier", "k", "reducer_name", "reduced_dim",
                  "roi_fingerprint", "roi_selection", "roi_name", "roi_networks", "k_folds"]
    rows = []
    for keys, grp in kfold_df.groupby(group_keys, observed=True, dropna=False):
        row = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))
        for col in metric_cols:
            vals = grp[col].dropna()
            row[f"mean_{col}"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"std_{col}"]  = float(vals.std())  if len(vals) > 1 else float("nan")
        # hard metrics per threshold aggregated across folds
        for thresh_key, _ in _KFOLD_HARD_THRESH_CONFIGS:
            for metric_key in _KFOLD_HARD_METRICS:
                fold_vals = [
                    _thresh_metric(r, metric_key)
                    for r in grp[thresh_key]
                    if isinstance(r, dict)
                ]
                fold_vals = [v for v in fold_vals if v is not None]
                col_prefix = f"{thresh_key}_{metric_key}"
                row[f"mean_{col_prefix}"] = float(np.mean(fold_vals)) if fold_vals else float("nan")
                row[f"std_{col_prefix}"]  = float(np.std(fold_vals))  if len(fold_vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregate helpers (pooled across all configs)
# ---------------------------------------------------------------------------

def build_pooled_long_table(kfold_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (fold × threshold × metric) with classifier/source/group context."""
    display_metrics = ["sensitivity", "precision", "specificity", "f1"]
    resting_set = set(ANALYSIS_SUBGROUPS["resting"])
    task_set    = set(ANALYSIS_SUBGROUPS["task"])

    rows = []
    for _, row in kfold_df.iterrows():
        ana = str(row.get("analysis", ""))
        group = "resting" if ana in resting_set else ("task" if ana in task_set else "unknown")

        raw_clf = str(row.get("classifier", ""))
        clf = "random_forest" if raw_clf in ("rf", "random_forest") else raw_clf

        src     = str(row.get("source", ""))
        roi_fp  = str(row.get("roi_fingerprint", ""))
        n_h     = row.get("n_holdout")
        weight  = float(n_h) if n_h is not None else float("nan")

        for thresh_key, thresh_label in _KFOLD_HARD_THRESH_CONFIGS:
            thresh_dict = row.get(thresh_key)
            for metric_key in display_metrics:
                val = _thresh_metric(thresh_dict, metric_key)
                rows.append({
                    "classifier":       clf,
                    "source":           src,
                    "roi_fingerprint":  roi_fp,
                    "analysis":         ana,
                    "analysis_group":   group,
                    "n_holdout":        weight,
                    "threshold":        thresh_key,
                    "threshold_label":  thresh_label,
                    "metric":           metric_key,
                    "value":            float(val) if val is not None else float("nan"),
                })
    return pd.DataFrame(rows)


def plot_aggregate_panel(
    df_long: pd.DataFrame,
    axis_col: str,
    axis_order: list,
    axis_labels: dict,
    title: str,
    save_path: Path,
) -> None:
    """4-row (thresholds) × 4-col (metrics) grid of pooled-mean bars with std errorbars."""
    from lib.pooled_stats import pooled_mean_std

    display_metrics = ["sensitivity", "precision", "specificity", "f1"]
    metric_titles   = [_KFOLD_METRIC_LABELS.get(m, m) for m in display_metrics]
    colors          = plot_config.RESULTS_PALETTE

    if axis_col not in df_long.columns:
        print(f"[aggregate] column '{axis_col}' not in long table, skipping {save_path.name}")
        return

    df = df_long[df_long[axis_col].isin(axis_order)].copy()

    fig, axes = plt.subplots(
        len(_KFOLD_HARD_THRESH_CONFIGS), len(display_metrics),
        figsize=(4.2 * len(display_metrics), 3.8 * len(_KFOLD_HARD_THRESH_CONFIGS)),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=10)

    for row_idx, (thresh_key, thresh_label) in enumerate(_KFOLD_HARD_THRESH_CONFIGS):
        sub_thresh = df[df["threshold"] == thresh_key]
        for col_idx, metric_key in enumerate(display_metrics):
            ax = axes[row_idx][col_idx]
            sub = sub_thresh[sub_thresh["metric"] == metric_key]

            for bar_idx, level in enumerate(axis_order):
                grp = sub[sub[axis_col] == level]
                mean, std, n = pooled_mean_std(grp["value"].to_numpy(), grp["n_holdout"].to_numpy())
                color = colors[bar_idx % len(colors)]
                bar_h = mean if np.isfinite(mean) else 0.0
                ax.bar(bar_idx, bar_h, color=color, alpha=0.85, width=0.6)
                if np.isfinite(mean) and np.isfinite(std):
                    ax.errorbar(bar_idx, mean, yerr=std, fmt="none",
                                color="black", capsize=4, linewidth=1)
                if np.isfinite(mean):
                    top = min((mean + std + 0.04 if np.isfinite(std) else mean + 0.04), 1.01)
                    ax.text(bar_idx, top, f"n={n}", ha="center", va="bottom", fontsize=6)

            ax.set_xticks(range(len(axis_order)))
            ax.set_xticklabels([axis_labels.get(lv, lv) for lv in axis_order], fontsize=8)
            ax.set_ylim(0, 1.12)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            if row_idx == 0:
                ax.set_title(metric_titles[col_idx], fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(thresh_label, fontsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    print(f"Saved: {save_path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_kfold_threshold_table(
    kfold_df: pd.DataFrame,
    save_dir: Path,
    roi_selection: str,
    roi_name: str,
    networks_label: str,
    filename_prefix: str,
    filename_suffix: str = "",
) -> None:
    analyses = list(kfold_df["analysis"].cat.categories)
    sources = sorted(kfold_df["source"].unique())

    rows_data = []
    for analysis in analyses:
        for source in sources:
            sub = kfold_df[(kfold_df["analysis"] == analysis) & (kfold_df["source"] == source)]
            if sub.empty:
                continue
            entry = {"label": f"{analysis} | {source}"}
            has_data = False
            for thresh_key, _ in _KFOLD_HARD_THRESH_CONFIGS:
                for metric_key in _KFOLD_HARD_METRICS:
                    fold_vals = [
                        _thresh_metric(r, metric_key)
                        for r in sub[thresh_key]
                        if isinstance(r, dict)
                    ]
                    fold_vals = [v for v in fold_vals if v is not None]
                    mean_v = float(np.mean(fold_vals)) if fold_vals else None
                    std_v  = float(np.std(fold_vals))  if len(fold_vals) > 1 else None
                    entry[(thresh_key, metric_key, "mean")] = mean_v
                    entry[(thresh_key, metric_key, "std")]  = std_v
                    if mean_v is not None:
                        has_data = True
            if has_data:
                rows_data.append(entry)

    if not rows_data:
        return

    display_metrics = [m for m in _KFOLD_HARD_METRICS if m != "threshold"]
    col_labels = [_KFOLD_METRIC_LABELS.get(m, m.replace("_", " ").title()) for m in _KFOLD_HARD_METRICS]
    header_colors = ["#DDDDDD", "#EEEEEE", "#F5F5F5", "#FAFAFA"]
    highlight_bg  = "#F0E442"
    n_rows = len(rows_data)

    fig, axes = plt.subplots(
        len(_KFOLD_HARD_THRESH_CONFIGS), 1,
        figsize=(9.0, len(_KFOLD_HARD_THRESH_CONFIGS) * max(1.75, 0.3 * n_rows + 1.25)),
        constrained_layout=False,
    )
    fig.subplots_adjust(hspace=0.0, top=0.94)
    if len(_KFOLD_HARD_THRESH_CONFIGS) == 1:
        axes = [axes]
    fig.suptitle(
        rf"{filename_prefix.upper()} K-fold holdout performance (mean $\pm$ SD across folds)"
        f"\nROI={_roi_title(roi_name, networks_label)}",
        fontsize=10, y=0.98,
    )

    row_labels = [r["label"] for r in rows_data]

    for ti, (thresh_key, thresh_label) in enumerate(_KFOLD_HARD_THRESH_CONFIGS):
        ax = axes[ti]
        ax.axis("off")
        ax.set_title(thresh_label, fontsize=11, fontweight="bold", pad=2)

        all_display = ["threshold"] + display_metrics
        cell_texts = []
        for r in rows_data:
            cells = []
            for metric_key in all_display:
                mean_v = r[(thresh_key, metric_key, "mean")]
                std_v  = r[(thresh_key, metric_key, "std")]
                if mean_v is None:
                    cells.append("|")
                elif std_v is not None:
                    cells.append(f"\\shortstack{{{mean_v:.3f} \\\\ $\\pm${std_v:.3f}}}")
                else:
                    cells.append(f"{mean_v:.3f}")
            cell_texts.append(cells)

        # highlight best mean for each non-threshold metric
        best_rows = {}
        for mi, metric_key in enumerate(all_display):
            if metric_key == "threshold":
                continue
            vals = [(ri, r[(thresh_key, metric_key, "mean")])
                    for ri, r in enumerate(rows_data)
                    if r[(thresh_key, metric_key, "mean")] is not None]
            if vals:
                best_rows[mi] = max(vals, key=lambda x: x[1])[0]

        tbl = ax.table(
            cellText=cell_texts,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            rowLoc="right",
            loc="upper center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.0, 1.8)
        tbl.auto_set_column_width(list(range(len(col_labels))))

        for ci in range(len(col_labels)):
            tbl[0, ci].set_facecolor(header_colors[ti % len(header_colors)])
            tbl[0, ci].set_text_props(fontweight="bold")

        for mi in best_rows:
            tbl[best_rows[mi] + 1, mi].set_facecolor(highlight_bg)
            tbl[best_rows[mi] + 1, mi].set_text_props(fontweight="bold")

    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    filename = (
        f"{filename_prefix}_kfold_threshold_table_roi-{slugify(roi_selection)}{suffix_str}.pdf"
    )
    fig.savefig(save_dir / filename, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def write_latex_kfold_table(
    kfold_summary: pd.DataFrame,
    out_path: Path,
    caption: str,
    label: str,
    threshold_key: str = "holdout_at_spec90",
) -> None:
    criterion = CRITERION_LABELS.get(threshold_key, threshold_key)
    sort_col = f"mean_{threshold_key}_f1"
    df = kfold_summary.copy()
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last")

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        f"  \\caption{{{caption} ({criterion})}}",
        f"  \\label{{{label}}}",
        r"  \begin{tabularx}{\linewidth}{l l R R R R R R}",
        r"    \toprule",
        (r"    \textbf{Formatting} & \textbf{Clf.}"
         r" & \textbf{Thr.} & \textbf{Rec.} & \textbf{Spec.}"
         r" & \textbf{Prec.} & \textbf{F1} & \textbf{AUC} \\"),
        r"    \midrule",
    ]
    for _, row in df.iterrows():
        cols = [
            format_label(str(row.get("analysis", ""))),
            str(row.get("classifier", "")),
            _fmt(row.get(f"mean_{threshold_key}_threshold"),  row.get(f"std_{threshold_key}_threshold")),
            _fmt(row.get(f"mean_{threshold_key}_sensitivity"), row.get(f"std_{threshold_key}_sensitivity")),
            _fmt(row.get(f"mean_{threshold_key}_specificity"), row.get(f"std_{threshold_key}_specificity")),
            _fmt(row.get(f"mean_{threshold_key}_precision"),   row.get(f"std_{threshold_key}_precision")),
            _fmt(row.get(f"mean_{threshold_key}_f1"),          row.get(f"std_{threshold_key}_f1")),
            _fmt(row.get("mean_auc_roc_pt"),                   row.get("std_auc_roc_pt")),
        ]
        lines.append("    " + " & ".join(cols) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabularx}",
        r"\end{table}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Top-N selection + formatting helpers
# ---------------------------------------------------------------------------

def _select_top_rows(
    df: pd.DataFrame,
    roi_fingerprint: str,
    setting: str,
    top_n: int,
    include_mean: bool = False,
    threshold_key: str = "holdout_at_spec90",
) -> pd.DataFrame:
    sub = df[df["roi_fingerprint"] == roi_fingerprint].copy()
    if setting in ANALYSIS_SUBGROUPS:
        sub = sub[sub["analysis"].isin(ANALYSIS_SUBGROUPS[setting])]
    if not include_mean:
        sub = sub[sub["is_mean"] == False]  # noqa: E712
    sub = sub.dropna(subset=[f"mean_{threshold_key}_sensitivity"])
    return sub.sort_values(
        [f"mean_{threshold_key}_sensitivity", f"mean_{threshold_key}_f1"],
        ascending=False,
    ).head(top_n)


def _strip0(s: str) -> str:
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def _fmt_pm(mean, std, decimals: int = 3) -> str:
    return _fmt_paren(mean, std, decimals)


def _fmt_paren(mean, std, decimals: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    fmt = f"{{:.{decimals}f}}"
    m = _strip0(fmt.format(mean))
    if std is not None and not (isinstance(std, float) and np.isnan(std)):
        return f"{m} ({_strip0(fmt.format(std))})"
    return m


def _fmt_scalar(val, decimals: int = 3) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    s = _strip0(f"{val:.{decimals}f}")
    return f"${s}$" if val < 0 else s


def _latex_table_preamble(caption: str, label: str, col_spec: str) -> list[str]:
    return [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        r"  \begin{threeparttable}",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        f"    \\begin{{tabularx}}{{\\linewidth}}{{{col_spec}}}",
        r"      \toprule",
    ]


def _latex_table_closing(notes: list[str] | None) -> list[str]:
    lines = [r"      \bottomrule", r"    \end{tabularx}"]
    if notes:
        lines += [r"    \begin{tablenotes}", r"      \small"]
        lines += [f"      {n}" for n in notes]
        lines += [r"    \end{tablenotes}"]
    lines += [r"  \end{threeparttable}", r"\end{table}"]
    return lines


def write_latex_validation_threshold_table(
    kfold_summary: pd.DataFrame,
    out_path: Path,
    groups: list[tuple[str, str, str]],
    caption: str,
    label: str,
    top_n: int = 4,
    notes: list[str] | None = None,
) -> None:
    n_cols = 6
    lines = _latex_table_preamble(caption, label, "X X r r r r")
    lines.append(
        r"      \textbf{Formatting} & \textbf{Clf.}"
        r" & \textbf{AUC-ROC} & \textbf{AUC-PR} & \textbf{Thr.\ (S90)} & \textbf{Thr.\ (F1)} \\"
    )
    lines.append(r"      \midrule")
    for i, (group_label, roi_fp, setting) in enumerate(groups):
        lines.append(
            f"      \\multicolumn{{{n_cols}}}{{l}}{{{group_label}}} \\\\ \\addlinespace[2pt]"
        )
        rows = _select_top_rows(kfold_summary, roi_fp, setting, top_n)
        for _, row in rows.iterrows():
            ana = ANALYSIS_LABEL.get(str(row.get("analysis", "")), str(row.get("analysis", "")))
            clf = CLASSIFIER_LABEL.get(str(row.get("classifier", "")), str(row.get("classifier", "")))
            cols = [
                ana, clf,
                _fmt_scalar(row.get("mean_auc_roc_pt")),
                _fmt_scalar(row.get("mean_auc_pr_pt")),
                _fmt_pm(row.get("mean_holdout_at_spec90_threshold"), row.get("std_holdout_at_spec90_threshold")),
                _fmt_pm(row.get("mean_holdout_at_f1_threshold"), row.get("std_holdout_at_f1_threshold")),
            ]
            lines.append("      " + " & ".join(cols) + r" \\")
        if i < len(groups) - 1:
            lines.append(r"      \midrule")
    lines += _latex_table_closing(notes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def write_latex_validation_reliability_table(
    kfold_summary: pd.DataFrame,
    out_path: Path,
    groups: list[tuple[str, str, str]],
    caption: str,
    label: str,
    top_n: int = 4,
    notes: list[str] | None = None,
) -> None:
    n_cols = 6
    lines = _latex_table_preamble(caption, label, "X X r r r r")
    lines.append(
        r"      \textbf{Formatting} & \textbf{Clf.}"
        r" & \textbf{Brier} & \textbf{BSS} & \textbf{Cal.\ slope} & \textbf{Int.} \\"
    )
    lines.append(r"      \midrule")
    for i, (group_label, roi_fp, setting) in enumerate(groups):
        lines.append(
            f"      \\multicolumn{{{n_cols}}}{{l}}{{{group_label}}} \\\\ \\addlinespace[2pt]"
        )
        rows = _select_top_rows(kfold_summary, roi_fp, setting, top_n)
        for _, row in rows.iterrows():
            ana = ANALYSIS_LABEL.get(str(row.get("analysis", "")), str(row.get("analysis", "")))
            clf = CLASSIFIER_LABEL.get(str(row.get("classifier", "")), str(row.get("classifier", "")))
            cols = [
                ana, clf,
                _fmt_scalar(row.get("mean_brier_pt")),
                _fmt_scalar(row.get("mean_bss_pt")),
                _fmt_scalar(row.get("mean_calibration_slope")),
                _fmt_scalar(row.get("mean_calibration_intercept")),
            ]
            lines.append("      " + " & ".join(cols) + r" \\")
        if i < len(groups) - 1:
            lines.append(r"      \midrule")
    lines += _latex_table_closing(notes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def write_latex_holdout_classification_table(
    kfold_summary: pd.DataFrame,
    out_path: Path,
    groups: list[tuple[str, str, str]],
    caption: str,
    label: str,
    top_n: int = 4,
    notes: list[str] | None = None,
    threshold_key: str = "holdout_at_spec90",
) -> None:
    n_cols = 7
    lines = _latex_table_preamble(caption, label, "X X r r r r r")
    lines.append(
        r"      \textbf{Formatting} & \textbf{Clf.}"
        r" & \textbf{Thr.} & \textbf{Prec.} & \textbf{Rec.}"
        r" & \textbf{Spec.} & \textbf{F1} \\"
    )
    lines.append(r"      \midrule")
    for i, (group_label, roi_fp, setting) in enumerate(groups):
        lines.append(
            f"      \\multicolumn{{{n_cols}}}{{l}}{{{group_label}}} \\\\ \\addlinespace[2pt]"
        )
        rows = _select_top_rows(kfold_summary, roi_fp, setting, top_n, threshold_key=threshold_key)
        for _, row in rows.iterrows():
            ana = ANALYSIS_LABEL.get(str(row.get("analysis", "")), str(row.get("analysis", "")))
            clf = CLASSIFIER_LABEL.get(str(row.get("classifier", "")), str(row.get("classifier", "")))
            cols = [
                ana, clf,
                _fmt_paren(row.get(f"mean_{threshold_key}_threshold"), row.get(f"std_{threshold_key}_threshold")),
                _fmt_paren(row.get(f"mean_{threshold_key}_precision"), row.get(f"std_{threshold_key}_precision")),
                _fmt_paren(row.get(f"mean_{threshold_key}_sensitivity"), row.get(f"std_{threshold_key}_sensitivity")),
                _fmt_paren(row.get(f"mean_{threshold_key}_specificity"), row.get(f"std_{threshold_key}_specificity")),
                _fmt_paren(row.get(f"mean_{threshold_key}_f1"), row.get(f"std_{threshold_key}_f1")),
            ]
            lines.append("      " + " & ".join(cols) + r" \\")
        if i < len(groups) - 1:
            lines.append(r"      \midrule")
    lines += _latex_table_closing(notes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device_src = "local" if "/Users/ipeglin" in str(ANALYSIS_RESULTS_DIR) else "IDUN"
    figs_out_dir = plot_config.get_figs_output_dir() / device_src
    csv_out_dir = REPO_ROOT / "confounds" / device_src

    latex_tables_dir = REPO_ROOT / "latex_tables" / device_src
    figs_out_dir.mkdir(parents=True, exist_ok=True)
    csv_out_dir.mkdir(parents=True, exist_ok=True)

    if not ANALYSIS_RESULTS_DIR.exists():
        print(f"Error: Results directory {ANALYSIS_RESULTS_DIR} not found.")
        sys.exit(1)

    print("Loading probabilistic classification reports...")
    df = load_reports(ANALYSIS_RESULTS_DIR, classifier_filter="knn")
    if df.empty:
        print("No KNN JSON reports found.")
    else:
        summary = build_summary_table(df)
        summary_path = csv_out_dir / f"{get_name()}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"summary written: {summary_path}")

        for fingerprint, fp_df in df.groupby("roi_fingerprint"):
            roi_selection = fp_df["roi_selection"].iloc[0]
            roi_name = fp_df["roi_name"].iloc[0]
            networks_label = fp_df["roi_networks"].iloc[0]
            print(f"\n=== ROI selection: {_roi_title(roi_name, networks_label)} ({fingerprint}) ===")

            for is_mean_val, track_df in fp_df.groupby("is_mean"):
                if is_mean_val and not PLOT_MEAN_ANALYSES:
                    continue

                track_label = "Mean Vectors" if is_mean_val else "Per-ROI Vectors"
                print(f"  --- Track: {track_label} ---")

                track_roi_name = f"{roi_name}_mean" if is_mean_val else roi_name
                track_roi_sel  = f"{roi_selection}_mean" if is_mean_val else roi_selection

                for (run_val, k_val, metric_val, reducer_name_val, reduced_dim_val), group_df in track_df.groupby(
                    ["run", "k", "metric", "reducer_name", "reduced_dim"], dropna=False
                ):
                    reduced_dim_val = None if (reduced_dim_val is None or (isinstance(reduced_dim_val, float) and np.isnan(reduced_dim_val))) else int(reduced_dim_val)
                    group_df = group_df.copy()
                    group_df["analysis"] = group_df["analysis"].cat.remove_unused_categories()
                    reducer_info = _reducer_label(reducer_name_val, reduced_dim_val)
                    print(f"  run={run_val:02d}, K={k_val}, Metric={metric_val}{reducer_info}")

                    for source_val, source_df in group_df.groupby("source"):
                        if not PLOT_HOLDOUT_FIGURES:
                            continue
                        source_slug = slugify(source_val)

                        all_dir = figs_out_dir / track_roi_sel / "all" / source_slug / _config_dir_name(
                            run_val, k_val, metric_val, reducer_name_val, reduced_dim_val
                        )
                        all_dir.mkdir(parents=True, exist_ok=True)
                        plot_roc_pr_grid(source_df, all_dir, run_val, k_val, metric_val,
                                         track_roi_sel, track_roi_name, networks_label,
                                         reducer_name=reducer_name_val, reduced_dim=reduced_dim_val)
                        plot_uncertainty_grid(source_df, all_dir, run_val, k_val, metric_val,
                                              track_roi_sel, track_roi_name, networks_label,
                                              reducer_name=reducer_name_val, reduced_dim=reduced_dim_val)
                        plot_confusion_grid(source_df, all_dir, run_val, k_val, metric_val,
                                            track_roi_sel, track_roi_name, networks_label,
                                            reducer_name=reducer_name_val, reduced_dim=reduced_dim_val)
                        plot_holdout_threshold_table(
                            source_df, all_dir, run_val,
                            param_label=f"K={k_val} | {metric_val}",
                            param_slug=f"k{k_val}_{slugify(metric_val)}",
                            roi_selection=track_roi_sel,
                            roi_name=track_roi_name,
                            networks_label=networks_label,
                            filename_prefix="knn",
                            reducer_name=reducer_name_val,
                            reduced_dim=reduced_dim_val,
                        )

                        for group_key, target_analyses in ANALYSIS_SUBGROUPS.items():
                            sub_df = source_df[source_df["analysis"].isin(target_analyses)].copy()
                            if sub_df.empty:
                                continue
                            sub_df["analysis"] = sub_df["analysis"].cat.remove_unused_categories()
                            sub_dir = figs_out_dir / track_roi_sel / group_key / source_slug / _config_dir_name(
                                run_val, k_val, metric_val, reducer_name_val, reduced_dim_val
                            )
                            sub_dir.mkdir(parents=True, exist_ok=True)
                            plot_roc_pr_grid(sub_df, sub_dir, run_val, k_val, metric_val,
                                             track_roi_sel, track_roi_name, networks_label,
                                             group_key, reducer_name=reducer_name_val, reduced_dim=reduced_dim_val)
                            plot_uncertainty_grid(sub_df, sub_dir, run_val, k_val, metric_val,
                                                  track_roi_sel, track_roi_name, networks_label,
                                                  group_key, reducer_name=reducer_name_val, reduced_dim=reduced_dim_val)
                            plot_confusion_grid(sub_df, sub_dir, run_val, k_val, metric_val,
                                                track_roi_sel, track_roi_name, networks_label,
                                                group_key, reducer_name=reducer_name_val, reduced_dim=reduced_dim_val)
                            plot_holdout_threshold_table(
                                sub_df, sub_dir, run_val,
                                param_label=f"K={k_val} | {metric_val}",
                                param_slug=f"k{k_val}_{slugify(metric_val)}",
                                roi_selection=track_roi_sel,
                                roi_name=track_roi_name,
                                networks_label=networks_label,
                                filename_prefix="knn",
                                filename_suffix=group_key,
                                reducer_name=reducer_name_val,
                                reduced_dim=reduced_dim_val,
                            )

                        for analysis_val in source_df["analysis"].unique():
                            single_df = source_df[source_df["analysis"] == analysis_val].copy()
                            if single_df.empty:
                                continue
                            single_df["analysis"] = single_df["analysis"].cat.remove_unused_categories()
                            analysis_slug = slugify(analysis_val)
                            single_dir = figs_out_dir / track_roi_sel / analysis_slug / source_slug / _config_dir_name(
                                run_val, k_val, metric_val, reducer_name_val, reduced_dim_val
                            )
                            single_dir.mkdir(parents=True, exist_ok=True)
                            plot_roc_pr_grid(single_df, single_dir, run_val, k_val, metric_val,
                                             track_roi_sel, track_roi_name, networks_label,
                                             analysis_slug, reducer_name=reducer_name_val, reduced_dim=reduced_dim_val)
                            plot_uncertainty_grid(single_df, single_dir, run_val, k_val, metric_val,
                                                  track_roi_sel, track_roi_name, networks_label,
                                                  analysis_slug, reducer_name=reducer_name_val, reduced_dim=reduced_dim_val)
                            plot_confusion_grid(single_df, single_dir, run_val, k_val, metric_val,
                                                track_roi_sel, track_roi_name, networks_label,
                                                analysis_slug, reducer_name=reducer_name_val, reduced_dim=reduced_dim_val)
                            plot_holdout_threshold_table(
                                single_df, single_dir, run_val,
                                param_label=f"K={k_val} | {metric_val}",
                                param_slug=f"k{k_val}_{slugify(metric_val)}",
                                roi_selection=track_roi_sel,
                                roi_name=track_roi_name,
                                networks_label=networks_label,
                                filename_prefix="knn",
                                filename_suffix=analysis_slug,
                                reducer_name=reducer_name_val,
                                reduced_dim=reduced_dim_val,
                            )

    print("Loading k-fold classification reports...")
    kfold_df = load_kfold_reports(ANALYSIS_RESULTS_DIR, classifier_filter="knn")
    if not kfold_df.empty:
        print("\n=== Loaded k-fold data: sources × analyses per ROI ===")
        for fp, grp in kfold_df.groupby("roi_fingerprint"):
            print(f"\n  {fp}")
            for src, sgrp in grp.groupby("source"):
                analyses = sorted(sgrp["analysis"].dropna().unique().astype(str))
                print(f"    {src}: {analyses}")

        kfold_summary = build_kfold_summary_table(kfold_df)
        kfold_summary.to_csv(csv_out_dir / f"{get_name()}_kfold.csv", index=False)

        print("\nROI fingerprints found:", kfold_df["roi_fingerprint"].unique().tolist())

        for fingerprint, fp_df in kfold_df.groupby("roi_fingerprint"):
            roi_selection  = fp_df["roi_selection"].iloc[0]
            roi_name       = fp_df["roi_name"].iloc[0]
            networks_label = fp_df["roi_networks"].iloc[0]
            print(f"\n=== K-fold ROI: {_roi_title(roi_name, networks_label)} ({fingerprint}) ===")

            fp_summary = kfold_summary[kfold_summary["roi_fingerprint"] == fingerprint]
            
            for source_val, src_summary in fp_summary.groupby("source"):
                if src_summary.empty:
                    continue
                source_slug = slugify(str(source_val))
                source_label = SOURCE_LABEL.get(str(source_val).lower(), str(source_val))

                for thr_key, thr_dir, thr_slug in [
                    ("holdout_at_spec90", "spec_90", "spec90"),
                    ("holdout_at_f1",     "f1",      "f1"),
                ]:
                    write_latex_kfold_table(
                        src_summary,
                        latex_tables_dir / thr_dir / "kfold" / f"knn_{slugify(roi_selection)}_{source_slug}_kfold_{thr_slug}.tex",
                        caption=rf"K-fold holdout performance ({CRITERION_LABELS[thr_key]}, KNN, {source_label}).",
                        label=f"tab:knn_kfold_{slugify(roi_selection)}_{source_slug}_{thr_slug}",
                        threshold_key=thr_key,
                    )

            for is_mean_val, track_df in fp_df.groupby("is_mean"):
                if is_mean_val and not PLOT_MEAN_ANALYSES:
                    continue

                track_roi_name = f"{roi_name}_mean" if is_mean_val else roi_name
                track_roi_sel  = f"{roi_selection}_mean" if is_mean_val else roi_selection
                track_df = track_df.copy()
                track_df["analysis"] = track_df["analysis"].cat.remove_unused_categories()

                for source_val, source_df in track_df.groupby("source"):
                    source_slug = slugify(source_val)

                    for (reducer_name_val, reduced_dim_val), pca_df in source_df.groupby(["reducer_name", "reduced_dim"], dropna=False):
                        reduced_dim_val = None if (reduced_dim_val is None or (isinstance(reduced_dim_val, float) and np.isnan(reduced_dim_val))) else int(reduced_dim_val)

                        if PLOT_COMBINED_KFOLD_FIGURES:
                            kf_dir = figs_out_dir / track_roi_sel / "kfold" / source_slug / _config_dir_name(
                                int(pca_df["run"].iloc[0]),
                                int(pca_df["k"].iloc[0]) if "k" in pca_df.columns and not pca_df["k"].isna().all() else 0,
                                "kfold",
                                reducer_name_val if not (isinstance(reducer_name_val, float) and np.isnan(reducer_name_val)) else None,
                                reduced_dim_val,
                            )
                            kf_dir.mkdir(parents=True, exist_ok=True)

                            plot_kfold_metrics_grid(pca_df, kf_dir,
                                                    track_roi_sel, track_roi_name, networks_label,
                                                    filename_prefix="knn")
                            plot_kfold_reliability_grid(pca_df, kf_dir,
                                                        track_roi_sel, track_roi_name, networks_label,
                                                        filename_prefix="knn")
                            plot_kfold_roc_pr_grid(pca_df, kf_dir,
                                                    track_roi_sel, track_roi_name, networks_label,
                                                    filename_prefix="knn")
                            plot_kfold_threshold_table(pca_df, kf_dir,
                                                       track_roi_sel, track_roi_name, networks_label,
                                                       filename_prefix="knn")

                        for group_key, target_analyses in ANALYSIS_SUBGROUPS.items():
                            sub_df = source_df[source_df["analysis"].isin(target_analyses)].copy()
                            if sub_df.empty:
                                continue
                            sub_df["analysis"] = sub_df["analysis"].cat.remove_unused_categories()
                            sub_dir = figs_out_dir / track_roi_sel / f"kfold_{group_key}" / source_slug / _config_dir_name(
                                int(source_df["run"].iloc[0]),
                                int(source_df["k"].iloc[0]) if "k" in source_df.columns and not source_df["k"].isna().all() else 0,
                                "kfold",
                                source_df["reducer_name"].dropna().iloc[0] if not source_df["reducer_name"].isna().all() else None,
                                None if source_df["reduced_dim"].isna().all() else int(source_df["reduced_dim"].dropna().iloc[0]),
                            )
                            sub_dir.mkdir(parents=True, exist_ok=True)
                            plot_kfold_metrics_grid(sub_df, sub_dir,
                                                    track_roi_sel, track_roi_name, networks_label,
                                                    group_key, filename_prefix="knn")
                            plot_kfold_reliability_grid(sub_df, sub_dir,
                                                        track_roi_sel, track_roi_name, networks_label,
                                                        group_key, filename_prefix="knn")
                            plot_kfold_roc_pr_grid(sub_df, sub_dir,
                                                    track_roi_sel, track_roi_name, networks_label,
                                                    filename_prefix="knn",
                                                    filename_suffix=group_key)
                            plot_kfold_threshold_table(sub_df, sub_dir,
                                                        track_roi_sel, track_roi_name, networks_label,
                                                        filename_prefix="knn",
                                                        filename_suffix=group_key)

                    for analysis_val in source_df["analysis"].unique():
                        single_df = source_df[source_df["analysis"] == analysis_val].copy()
                        if single_df.empty:
                            continue
                        single_df["analysis"] = single_df["analysis"].cat.remove_unused_categories()
                        analysis_slug = slugify(analysis_val)
                        single_dir = figs_out_dir / track_roi_sel / f"kfold_{analysis_slug}" / source_slug / _config_dir_name(
                            int(source_df["run"].iloc[0]),
                            int(source_df["k"].iloc[0]) if "k" in source_df.columns and not source_df["k"].isna().all() else 0,
                            "kfold",
                            source_df["reducer_name"].dropna().iloc[0] if not source_df["reducer_name"].isna().all() else None,
                            None if source_df["reduced_dim"].isna().all() else int(source_df["reduced_dim"].dropna().iloc[0]),
                        )
                        single_dir.mkdir(parents=True, exist_ok=True)
                        plot_kfold_metrics_grid(single_df, single_dir,
                                                track_roi_sel, track_roi_name, networks_label,
                                                analysis_slug, filename_prefix="knn")
                        plot_kfold_reliability_grid(single_df, single_dir,
                                                    track_roi_sel, track_roi_name, networks_label,
                                                    analysis_slug, filename_prefix="knn")
                        plot_kfold_roc_pr_grid(single_df, single_dir,
                                               track_roi_sel, track_roi_name, networks_label,
                                               filename_prefix="knn",
                                               filename_suffix=analysis_slug)
                        plot_kfold_threshold_table(single_df, single_dir,
                                                   track_roi_sel, track_roi_name, networks_label,
                                                   filename_prefix="knn",
                                                   filename_suffix=analysis_slug)

        latex_tables_dir.mkdir(parents=True, exist_ok=True)
        GROUPS = [
            (r"\textit{vmPFC and AMY: resting-state (subject-stratified)}",  "vmpfc_and_amy_subject_stratified",          "resting"),
            (r"\textit{vmPFC and AMY: task-block (subject-stratified)}",     "vmpfc_and_amy_subject_stratified",          "task"),
            (r"\textit{Keedwell SAD vs Neutral: resting-state}",             "keedwell_sad_v_neutral_subject_stratified", "resting"),
            (r"\textit{Keedwell SAD vs Neutral: task-block}",                "keedwell_sad_v_neutral_subject_stratified", "task"),
        ]
        
        for source_val, src_summary in kfold_summary.groupby("source"):
            if src_summary.empty:
                continue
            source_slug = slugify(str(source_val))
            source_label = SOURCE_LABEL.get(str(source_val).lower(), str(source_val))

            write_latex_validation_threshold_table(
                src_summary,
                latex_tables_dir / "shared" / "threshold_tuning" / f"knn_{source_slug}_validation_threshold_selection.tex",
                groups=GROUPS,
                caption=rf"Validation-derived thresholds and discrimination metrics (KNN, {source_label}).",
                label=f"tab:knn_{source_slug}_validation_threshold_selection",
            )
            write_latex_validation_reliability_table(
                src_summary,
                latex_tables_dir / "shared" / "reliability" / f"knn_{source_slug}_validation_reliability.tex",
                groups=GROUPS,
                caption=rf"Validation calibration and reliability metrics (KNN, {source_label}).",
                label=f"tab:knn_{source_slug}_validation_reliability",
            )
            for thr_key, thr_dir, thr_caption in [
                ("holdout_at_spec90", "spec_90", rf"Hold-out classification performance at Spec.$~\geq 90\%$ (KNN, {source_label})."),
                ("holdout_at_f1",     "f1",      rf"Hold-out classification performance at F1-optimal threshold (KNN, {source_label})."),
            ]:
                write_latex_holdout_classification_table(
                    src_summary,
                    latex_tables_dir / thr_dir / "holdout_classification" / f"knn_{source_slug}_holdout_classification_{thr_dir}.tex",
                    groups=GROUPS,
                    caption=thr_caption,
                    label=f"tab:knn_{source_slug}_holdout_classification_{thr_dir}",
                    threshold_key=thr_key,
                )

    # --- Aggregate panels (pooled across all configs) ---
    print("\nLoading RF k-fold reports for cross-classifier aggregate...")
    rf_kfold_df = load_kfold_reports(ANALYSIS_RESULTS_DIR, classifier_filter=("random_forest", "rf"))
    if not kfold_df.empty or not rf_kfold_df.empty:
        combined   = pd.concat([kfold_df, rf_kfold_df], ignore_index=True)
        long_df    = build_pooled_long_table(combined)
        agg_dir    = figs_out_dir / "_aggregate"

        plot_aggregate_panel(
            long_df, axis_col="analysis_group",
            axis_order=["resting", "task"],
            axis_labels={"resting": "Resting", "task": "Task"},
            title="Resting-state vs Task (pooled across all configs)",
            save_path=agg_dir / "aggregate_resting_vs_task.pdf",
        )
        plot_aggregate_panel(
            long_df, axis_col="classifier",
            axis_order=["knn", "random_forest"],
            axis_labels={"knn": "KNN", "random_forest": "RF"},
            title="KNN vs RF (pooled across all configs)",
            save_path=agg_dir / "aggregate_knn_vs_rf.pdf",
        )
        plot_aggregate_panel(
            long_df, axis_col="source",
            axis_order=["cwt", "hht", "hht_roi"],
            axis_labels={"cwt": "CWT", "hht": "HHT", "hht_roi": "HHT-S"},
            title="CWT vs HHT vs HHT-S (pooled across all configs)",
            save_path=agg_dir / "aggregate_source.pdf",
        )
        long_df["region_group"] = long_df["roi_fingerprint"].str.replace(
            "_subject_stratified", "", regex=False
        )
        plot_aggregate_panel(
            long_df, axis_col="region_group",
            axis_order=["vmpfc_and_amy", "keedwell_sad_v_neutral"],
            axis_labels={"vmpfc_and_amy": "vmPFC-AMY", "keedwell_sad_v_neutral": "Keedwell"},
            title="Regional differences (pooled across all configs)",
            save_path=agg_dir / "aggregate_region.pdf",
        )
        long_df.to_csv(csv_out_dir / f"{get_name()}_kfold_pooled_long.csv", index=False)
        print(f"Pooled long table: {csv_out_dir / f'{get_name()}_kfold_pooled_long.csv'}")

    print(f"\nSuccess! Figures saved in: {figs_out_dir}")

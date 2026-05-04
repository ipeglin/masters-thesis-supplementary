import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lib.fs.get_script_name import get_name
from lib.fs.project_config import REPO_ROOT
# Import local configuration for REPO_ROOT and styling
from plotting import plot_config

# Directory where your JSON result files are located
RESULTS_DIR = Path("/Users/ipeglin/Documents/masters_thesis/classifier_results/")
# RESULTS_DIR = Path("/Volumes/work/classifier_results") # IDUN Network Mount

# Define the specific analysis order for professional reporting
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

# Global toggle to show figures
SHOW_FIGURES = False

# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def format_label(text):
    """Converts 'task_averaged' -> 'Task Averaged', 'ts' -> 'TS', and 'cwt' -> 'CWT'."""
    if not text:
        return text
    if text.lower() == "hht_roi":
        return "HHT ROI"
    # Ensure spectrogram and modality acronyms are fully capitalized
    if text.lower() in ['cwt', 'hht', 'ts']:
        return text.upper()

    # Replace underscores with spaces and apply Title Case
    return text.replace("_", " ").title()


def slugify(text):
    """Filesystem-safe lowercase slug for filenames."""
    if not text:
        return "unspecified"
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "unspecified"

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

# Sentinel for results that landed at RESULTS_DIR root (no roi_selection.name suffix)
UNNAMED_ROI_SELECTION = "_unnamed"

# Separator inserted by `AppConfig::resolved_classification_results_dir` between
# the roi_selection.name and the cortical_networks list. Mirror it here so the
# plot grouping decomposes `name__net-LimbicA_LimbicB` into name + networks.
NETWORK_SUFFIX_SEP = "__net-"


def parse_roi_dir(dir_name: str):
    """Split a results subdir name into (name, networks_label).

    Layout produced by `AppConfig::resolved_classification_results_dir()`:
        <name>                                 -> ("<name>", "")
        <name>__net-<NetA>_<NetB>              -> ("<name>", "NetA+NetB")
    """
    if NETWORK_SUFFIX_SEP not in dir_name:
        return dir_name, ""
    base, networks_part = dir_name.split(NETWORK_SUFFIX_SEP, 1)
    networks_label = "+".join(networks_part.split("_")) if networks_part else ""
    return base, networks_label


def load_results(directory):
    """Recursively reads JSON files under per-ROI-selection subdirs.

    Each JSON's parent dir name is treated as the roi_selection identifier
    (matches `AppConfig::resolved_classification_results_dir()` layout, which
    may suffix the configured `roi_selection.name` with `__net-<networks>`
    when `cortical_networks` is non-empty). The full dir name is the canonical
    grouping key so a `vpfc_mpfc_amy` run with no network filter never mixes
    with the same `name` filtered to LimbicA+LimbicB. When the report payload
    carries `roi_selection_fingerprint` it takes precedence for grouping.
    """
    base = Path(directory).resolve()
    records = []
    for filepath in base.rglob("*.json"):
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Extract run index from filename (e.g. run-00, run-01)
        run_match = re.search(r'run-(\d+)', filepath.stem)
        run_idx = int(run_match.group(1)) if run_match else 0

        # Parent dir relative to RESULTS_DIR encodes roi_selection.name and
        # (optionally) the active cortical_networks list.
        parent = filepath.parent.resolve()
        if parent == base:
            dir_selection = UNNAMED_ROI_SELECTION
        else:
            dir_selection = parent.relative_to(base).parts[0]

        roi_name, networks_label = parse_roi_dir(dir_selection)

        # Fingerprint from the report payload (when present) is authoritative
        # for grouping; otherwise fall back to the dir name (which already
        # encodes name + networks).
        report_fp = data.get("roi_selection_fingerprint")
        roi_fingerprint = report_fp or dir_selection

        # Format metadata attributes
        record = {
            "run": run_idx,
            "analysis": format_label(data.get("analysis")),
            "source": format_label(data.get("source")),
            "k": data.get("num_neighbors"),
            "metric": format_label(data.get("metric")),
            "roi_selection": dir_selection,
            "roi_name": roi_name,
            "roi_networks": networks_label,
            "roi_fingerprint": roi_fingerprint,
        }

        # Extract Metrics from Test and Validation splits
        for split in ['test', 'val']:
            if split in data:
                record[f"{split}_acc"] = data[split].get("accuracy")
                record[f"{split}_sens"] = data[split].get("sensitivity")
                record[f"{split}_spec"] = data[split].get("specificity")
                record[f"{split}_cm"] = data[split].get("confusion_matrix")

        records.append(record)

    df = pd.DataFrame(records)
    
    if not df.empty:
        # Add dynamically discovered analyses safely
        known_categories = ANALYSIS_ORDER
        in_df = df['analysis'].unique()
        missing = [x for x in in_df if x not in known_categories]
        all_categories = known_categories + missing

        # Enforce the specific categorical order for Analysis Types
        df['analysis'] = pd.Categorical(
            df['analysis'], 
            categories=all_categories, 
            ordered=True
        )
        # Sort values so that confusion matrices and plots follow the order
        df = df.sort_values(['analysis', 'source'])
        
    return df

# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------

def _roi_title(roi_name: str, networks_label: str) -> str:
    """`vpfc_mpfc_amy` + `LimbicA+LimbicB` -> `vpfc_mpfc_amy [LimbicA+LimbicB]`."""
    if networks_label:
        return f"{roi_name} [{networks_label}]"
    return roi_name


def plot_comparative_metrics(
    df, save_dir: Path, run_val=0,
    roi_selection: str = UNNAMED_ROI_SELECTION,
    roi_name: str = UNNAMED_ROI_SELECTION,
    networks_label: str = "",
):
    """Plots comparative bar charts with clean metric and distance labels."""
    melted = df.melt(
        id_vars=['run', 'analysis', 'source', 'k', 'metric'],
        value_vars=['test_acc', 'test_sens', 'test_spec'],
        var_name='Metric_Type', value_name='Score'
    )

    # Map internal variable names to clean legend names
    melted['Metric_Type'] = melted['Metric_Type'].replace({
        'test_acc': 'Accuracy', 'test_sens': 'Sensitivity', 'test_spec': 'Specificity'
    })

    # Create the comparison grid
    g = sns.catplot(
        data=melted,
        x='analysis', y='Score', hue='source',
        col='Metric_Type', row='metric',
        kind='bar', palette='viridis',
        height=4, aspect=1.2, margin_titles=True,
        order=df['analysis'].cat.categories
    )

    # FIXED: Using correct keys {col_name} and {row_name}
    g.set_titles(col_template="{col_name}", row_template="Distance Metric: {row_name}")

    # Global title with K-neighbor info
    k_val = df['k'].iloc[0]
    metric_val = df['metric'].iloc[0]
    g.fig.suptitle(
        f"KNN Performance Comparison (ROI={_roi_title(roi_name, networks_label)}, "
        f"Run {run_val:02d}, K={k_val}, Metric={metric_val})",
        y=1.05, fontsize=16,
    )

    g.set_axis_labels("Analysis Type", "Score (0-1)")
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Save to PDF and then display to screen
    safe_metric = slugify(metric_val)
    safe_roi = slugify(roi_selection)
    filename = f"knn_performance_roi-{safe_roi}_run{run_val:02d}_k{k_val}_{safe_metric}.pdf"
    g.savefig(save_dir / filename)

    if SHOW_FIGURES:
        plt.show()

    plt.close()


def plot_confusion_matrices(
    df, save_dir: Path, run_val=0,
    roi_selection: str = UNNAMED_ROI_SELECTION,
    roi_name: str = UNNAMED_ROI_SELECTION,
    networks_label: str = "",
):
    """Plots confusion matrices in the specified categorical order."""
    num_plots = len(df)
    cols = 4
    rows = int(np.ceil(num_plots / cols))
    
    # fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), squeeze=False)
    fig, axes = plt.subplots(rows, cols, squeeze=False)
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(df.iterrows()):
        cm = np.array(row['test_cm'])
        ax = axes[idx]
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=['HC', 'AN'], yticklabels=['HC', 'AN'])
        
        # Build descriptive Title-Cased header
        title = (f"{row['analysis']}\n"
                 f"Source: {row['source']} | Run: {run_val:02d} | K={row['k']}\n"
                 f"Distance Metric: {row['metric']}")
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    # Hide empty subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    k_val = df['k'].iloc[0]
    metric_val = df['metric'].iloc[0]
    safe_metric = slugify(metric_val)
    safe_roi = slugify(roi_selection)
    fig.suptitle(
        f"ROI Selection: {_roi_title(roi_name, networks_label)}",
        fontsize=12, y=1.0,
    )
    filename = f"knn_cm_roi-{safe_roi}_run{run_val:02d}_k{k_val}_{safe_metric}.pdf"
    fig.savefig(save_dir / filename, dpi=300)

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)

def plot_run_development(
    df, save_dir: Path,
    roi_selection: str = UNNAMED_ROI_SELECTION,
    roi_name: str = UNNAMED_ROI_SELECTION,
    networks_label: str = "",
):
    """Plots cross-run development for equal conditions as individual figures."""
    melted = df.melt(
        id_vars=['run', 'analysis', 'source', 'k', 'metric'], 
        value_vars=['test_acc', 'test_sens', 'test_spec'],
        var_name='Metric_Type', value_name='Score'
    )
    
    metrics_map = {
        'test_acc': 'Accuracy',
        'test_sens': 'Sensitivity', 
        'test_spec': 'Specificity'
    }
    
    k_val = df['k'].iloc[0]
    metric_val = df['metric'].iloc[0]
    safe_metric = slugify(metric_val)
    safe_roi = slugify(roi_selection)

    for metric_key, metric_title in metrics_map.items():
        subset = melted[melted['Metric_Type'] == metric_key]
        if subset.empty:
            continue
            
        fig, ax = plt.subplots()
        sns.lineplot(
            data=subset,
            x='run', y='Score', hue='analysis', style='source',
            markers=True, dashes=True, ax=ax
        )
        
        ax.set_title(
            f"{metric_title} Cross-Run Dev\n"
            f"ROI={_roi_title(roi_name, networks_label)}, K={k_val}, Metric={metric_val}",
            fontsize=12,
        )
        ax.set_xticks(df['run'].unique())
        
        # Place legend outside to maintain plot area aspect ratio
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        fig.tight_layout()
        filename = f"knn_run_dev_{metric_title.lower()}_roi-{safe_roi}_k{k_val}_{safe_metric}.pdf"
        fig.savefig(save_dir / filename)
        
        if SHOW_FIGURES:
            plt.show()
            
        plt.close(fig)

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = plot_config.get_figs_output_dir()

    if not RESULTS_DIR.exists():
        print(f"Error: Results directory {RESULTS_DIR} not found.")
        sys.exit(1)

    print("Loading and sorting results...")
    results_df = load_results(RESULTS_DIR)
    
    if results_df.empty:
        print("No valid JSON results found in the specified directory.")
    else:
        # Outer partition: roi_fingerprint. Plots never mix selections (so
        # `vpfc_mpfc_amy` with no network filter and `vpfc_mpfc_amy` filtered
        # to LimbicA+LimbicB stay disjoint even though `roi_name` is shared).
        for fingerprint, fp_df in results_df.groupby('roi_fingerprint'):
            roi_selection_name = fp_df['roi_selection'].iloc[0]
            roi_name = fp_df['roi_name'].iloc[0]
            networks_label = fp_df['roi_networks'].iloc[0]
            label = _roi_title(roi_name, networks_label)
            print(f"\n=== ROI selection: {label} (dir={roi_selection_name}, fp={fingerprint}) ===")

            # Cross-run dev plots: group by K + distance metric within selection
            for (k_val, metric_val), group_df in fp_df.groupby(['k', 'metric']):
                if len(group_df['run'].unique()) > 1:
                    print(f"  cross-run dev: K={k_val}, Metric={metric_val}")
                    dev_df = group_df.copy()
                    dev_df['analysis'] = dev_df['analysis'].cat.remove_unused_categories()
                    plot_run_development(
                        dev_df, out_dir,
                        roi_selection=roi_selection_name,
                        roi_name=roi_name,
                        networks_label=networks_label,
                    )

            # Within-run plots: group by Run + K + distance metric within selection
            for (run_val, k_val, metric_val), group_df in fp_df.groupby(['run', 'k', 'metric']):
                print(f"  run={run_val:02d}, K={k_val}, Metric={metric_val}")
                group_df = group_df.copy()
                group_df['analysis'] = group_df['analysis'].cat.remove_unused_categories()

                plot_comparative_metrics(
                    group_df, out_dir, run_val,
                    roi_selection=roi_selection_name,
                    roi_name=roi_name,
                    networks_label=networks_label,
                )
                plot_confusion_matrices(
                    group_df, out_dir, run_val,
                    roi_selection=roi_selection_name,
                    roi_name=roi_name,
                    networks_label=networks_label,
                )

        print(f"\nSuccess! Figures saved in: {out_dir}")
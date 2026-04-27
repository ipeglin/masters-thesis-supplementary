import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup Paths & Matplotlib environment
# ---------------------------------------------------------------------------
# Resolve the project root (assuming this script is in /scripts)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Setup a local matplotlib cache to avoid conflicts
_matplotlib_cache_dir = _REPO_ROOT / ".cache" / "matplotlib"
_matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import local configuration for REPO_ROOT and styling
import plot_config as plot_config  # noqa: F401

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
    """Converts 'task_averaged' -> 'Task Averaged' and 'cwt' -> 'CWT'."""
    if not text:
        return text
    # Ensure spectrogram acronyms are fully capitalized
    if text.lower() in ['cwt', 'hht']:
        return text.upper()
      
    # Replace underscores with spaces and apply Title Case
    return text.replace("_", " ").title()

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_results(directory):
    """Reads JSON files, applies Title Case formatting, and enforces analysis order."""
    records = []
    for filepath in Path(directory).glob("*.json"):
        with open(filepath, 'r') as f:
            data = json.load(f)
            
            # Extract run index from filename (e.g. run-00, run-01)
            run_match = re.search(r'run-(\d+)', filepath.stem)
            run_idx = int(run_match.group(1)) if run_match else 0
            
            # Format metadata attributes
            record = {
                "run": run_idx,
                "analysis": format_label(data.get("analysis")),
                "source": format_label(data.get("source")),
                "k": data.get("num_neighbors"),
                "metric": format_label(data.get("metric")),
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

def plot_comparative_metrics(df, save_dir: Path, run_val=0):
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
    g.fig.suptitle(f"KNN Performance Comparison (Run {run_val:02d}, K={k_val}, Metric={metric_val})", y=1.05, fontsize=16)
    
    g.set_axis_labels("Analysis Type", "Score (0-1)")
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Save to PDF and then display to screen
    safe_metric = metric_val.lower().replace(' ', '_')
    filename = f"knn_performance_run{run_val:02d}_k{k_val}_{safe_metric}.pdf"
    g.savefig(save_dir / filename, bbox_inches='tight')
    
    if SHOW_FIGURES:
        plt.show()
        
    plt.close()


def plot_confusion_matrices(df, save_dir: Path, run_val=0):
    """Plots confusion matrices in the specified categorical order."""
    num_plots = len(df)
    cols = 4
    rows = int(np.ceil(num_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), squeeze=False)
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
        
    plt.tight_layout()
    k_val = df['k'].iloc[0]
    metric_val = df['metric'].iloc[0]
    safe_metric = metric_val.lower().replace(' ', '_')
    filename = f"knn_cm_run{run_val:02d}_k{k_val}_{safe_metric}.pdf"
    fig.savefig(save_dir / filename, dpi=300)
    
    if SHOW_FIGURES:
        plt.show()
        
    plt.close(fig)

def plot_run_development(df, save_dir: Path):
    """Plots cross-run development for equal conditions."""
    melted = df.melt(
        id_vars=['run', 'analysis', 'source', 'k', 'metric'], 
        value_vars=['test_acc', 'test_sens', 'test_spec'],
        var_name='Metric_Type', value_name='Score'
    )
    
    melted['Metric_Type'] = melted['Metric_Type'].replace({
        'test_acc': 'Accuracy', 'test_sens': 'Sensitivity', 'test_spec': 'Specificity'
    })
    
    g = sns.relplot(
        data=melted,
        x='run', y='Score', hue='analysis', style='source',
        col='Metric_Type', row='metric',
        kind='line', markers=True, dashes=True,
        height=4, aspect=1.2
    )
    
    g.set_titles(col_template="{col_name}", row_template="Distance Metric: {row_name}")
    
    # Global title
    k_val = df['k'].iloc[0]
    metric_val = df['metric'].iloc[0]
    g.fig.suptitle(f"KNN Cross-Run Dev (K={k_val}, Metric={metric_val})", y=1.05, fontsize=16)
    
    # Use distinct x-ticks for categorical runs
    for ax in g.axes.flat:
        ax.set_xticks(df['run'].unique())
    
    safe_metric = metric_val.lower().replace(' ', '_')
    filename = f"knn_run_dev_k{k_val}_{safe_metric}.pdf"
    g.savefig(save_dir / filename, bbox_inches='tight')
    
    if SHOW_FIGURES:
        plt.show()
        
    plt.close()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Target project-root figures directory
    out_dir = plot_config.REPO_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not RESULTS_DIR.exists():
        print(f"Error: Results directory {RESULTS_DIR} not found.")
        sys.exit(1)

    print("Loading and sorting results...")
    results_df = load_results(RESULTS_DIR)
    
    if results_df.empty:
        print("No valid JSON results found in the specified directory.")
    else:
        # Group by K and distance metric to plot cross-run development
        for (k_val, metric_val), group_df in results_df.groupby(['k', 'metric']):
            if len(group_df['run'].unique()) > 1:
                print(f"Generating cross-run dev plot for K={k_val}, Metric={metric_val}...")
                dev_df = group_df.copy()
                dev_df['analysis'] = dev_df['analysis'].cat.remove_unused_categories()
                plot_run_development(dev_df, out_dir)

        # Group by Run, K, and distance metric for within-run comparison
        groups = results_df.groupby(['run', 'k', 'metric'])
        for (run_val, k_val, metric_val), group_df in groups:
            print(f"Generating plots for Run={run_val:02d}, K={k_val}, Metric={metric_val}...")
            # Drop unused categories for cleaner plots
            group_df = group_df.copy()
            group_df['analysis'] = group_df['analysis'].cat.remove_unused_categories()
            
            plot_comparative_metrics(group_df, out_dir, run_val)
            plot_confusion_matrices(group_df, out_dir, run_val)
        
        print(f"Success! Figures saved in: {out_dir}")
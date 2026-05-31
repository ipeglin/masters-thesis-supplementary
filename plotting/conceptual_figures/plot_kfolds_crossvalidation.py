import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from lib.fs.get_script_name import get_name
from plotting import plot_config
from plotting.plot_config import FIG_WIDTH_INCHES, get_figs_output_dir


def _plot_kfolds_crossvalidation(
    *,
    k: int,
    mode: str,
    title: str,
    out_suffix: str,
):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, FIG_WIDTH_INCHES * 0.5))

    bar_height = 0.7
    bar_spacing = 0.3
    bar_width = 8.0
    
    _cp = plot_config.CONCEPT_PALETTE
    train_color = _cp["pale"]
    oob_color = _cp["accent_a"]
    calibration_color = _cp["accent_b"]
    validation_color = _cp["mid"]

    # Draw k-fold splits
    for fold in range(k):
        y_pos = k - fold - 1
        fold_width = bar_width / k

        if mode == "general":
            oob_fold = fold
            skip_folds = {oob_fold}
        elif mode == "implementation":
            validation_fold = fold
            calibration_fold = (fold + 1) % k
            skip_folds = {calibration_fold, validation_fold}
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Draw training folds
        for i in range(k):
            if i not in skip_folds:
                rect = mpatches.Rectangle(
                    (i * fold_width, y_pos - bar_height / 2),
                    fold_width - 0.05,
                    bar_height,
                    facecolor=train_color,
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.6,
                    zorder=2,
                )
                ax.add_patch(rect)
        
        if mode == "general":
            rect_oob = mpatches.Rectangle(
                (oob_fold * fold_width, y_pos - bar_height / 2),
                fold_width - 0.05,
                bar_height,
                facecolor=oob_color,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.8,
                zorder=3,
            )
            ax.add_patch(rect_oob)
        else:
            # Draw calibration fold
            rect_calibration = mpatches.Rectangle(
                (calibration_fold * fold_width, y_pos - bar_height / 2),
                fold_width - 0.05,
                bar_height,
                facecolor=calibration_color,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.8,
                zorder=3,
            )
            ax.add_patch(rect_calibration)

            # Draw validation fold
            rect_validation = mpatches.Rectangle(
                (validation_fold * fold_width, y_pos - bar_height / 2),
                fold_width - 0.05,
                bar_height,
                facecolor=validation_color,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.8,
                zorder=3,
            )
            ax.add_patch(rect_validation)
        
        # Add iteration label
        ax.text(-0.6, y_pos, f"Iter {fold + 1}", ha="right", va="center", fontsize=9)
    
    # Add legend below the folds
    train_patch = mpatches.Patch(facecolor=train_color, edgecolor="black", alpha=0.6, label="Training Set")
    if mode == "general":
        oob_patch = mpatches.Patch(facecolor=oob_color, edgecolor="black", alpha=0.8, label="Out-of-Bag")
        legend_handles = [train_patch, oob_patch]
        legend_cols = 2
    else:
        calibration_patch = mpatches.Patch(
            facecolor=calibration_color, edgecolor="black", alpha=0.8, label="Validation Set"
        )
        validation_patch = mpatches.Patch(
            facecolor=validation_color, edgecolor="black", alpha=0.8, label="Holdout Set"
        )
        legend_handles = [train_patch, calibration_patch, validation_patch]
        legend_cols = 3

    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
        fontsize=9,
        framealpha=0.95,
        ncol=legend_cols,
    )
    
    # Set axis properties
    ax.set_xlim(-1.2, bar_width + 0.5)
    ax.set_ylim(-0.65, k - 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    
    # Add title
    ax.text(
        bar_width / 2,
        k - 0.05,
        title,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    
    # Save figure
    out_dir = get_figs_output_dir()
    out_path = out_dir / f"{get_name()}{out_suffix}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {out_path}")


def plot_kfolds_crossvalidation_general():
    _plot_kfolds_crossvalidation(
        k=5,
        mode="general",
        title="5-Fold Cross-Validation (1 Out-of-Bag)",
        out_suffix="_general",
    )


def plot_kfolds_crossvalidation_implementation():
    _plot_kfolds_crossvalidation(
        k=7,
        mode="implementation",
        title="Nested Cross-Validation (1 Validation, 1 Holdout)",
        out_suffix="_implementation",
    )


if __name__ == "__main__":
    plot_kfolds_crossvalidation_general()
    plot_kfolds_crossvalidation_implementation()

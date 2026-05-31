import matplotlib.pyplot as plt
from plot_config import get_figs_output_dir, RESULTS_PALETTE

def plot_datasplit(train_frac=0.7, val_frac=0.15, test_frac=0.15):
    """Plot data split block."""
    splits = [train_frac, val_frac, test_frac]
    labels = ['Training', 'Hold-out', 'Calibration']
    
    colors = [RESULTS_PALETTE[0], RESULTS_PALETTE[1], RESULTS_PALETTE[2]]
    
    fig, ax = plt.subplots(
                            figsize=(6, 1) # Example dimensions
                           )
    
    left = 0
    for i, (frac, label) in enumerate(zip(splits, labels)):
        ax.barh(0, frac, left=left, color=colors[i], edgecolor='black')
        # ax.barh(0, frac, left=left, label=label, edgecolor='black', alpha=0.4)
        ax.text(left + frac / 2, 0, f"{label}\n{frac*100:.0f}%", 
                ha='center', va='center')
        left += frac
        
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    out_dir = get_figs_output_dir()
    fig.savefig(out_dir / "datasplit.pdf", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    plot_datasplit()

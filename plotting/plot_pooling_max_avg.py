import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from lib.fs.get_script_name import get_name
from plotting.plot_config import FIG_WIDTH_INCHES, get_figs_output_dir

def draw_grid(ax, x_start, y_start, data, title, colors=None):
    rows, cols = data.shape
    for i in range(rows):
        for j in range(cols):
            val = data[i, j]
            
            # Map colors based on 2x2 pool regions
            c = 'white'
            if colors:
                if data.shape == (2, 2):
                    c = colors[i][j]
                else:
                    c = colors[i // (rows // 2)][j // (cols // 2)]
                    
            rect = patches.Rectangle(
                (x_start + j, y_start - i - 1), 1, 1, 
                linewidth=1.5, edgecolor='black', facecolor=c, alpha=0.6
            )
            ax.add_patch(rect)
            
            # Format number display
            text_val = f"{val:.1f}" if isinstance(val, float) else str(val)
            text_val = text_val.rstrip('0').rstrip('.') if '.' in text_val else text_val
            
            ax.text(x_start + j + 0.5, y_start - i - 0.5, text_val, 
                    ha='center', va='center', fontsize=11, weight='medium')
            
    ax.text(x_start + cols / 2, y_start + 0.2, title, 
            ha='center', va='bottom', fontsize=11, weight='bold')

def generate_pooling_types_plot():
    out_dir = get_figs_output_dir()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES * 1.2, FIG_WIDTH_INCHES * 0.5))
    
    # Input matrix (4x4)
    input_data = np.array([
        [12, 20, 30,  0],
        [ 8, 12,  2,  0],
        [34, 70, 37,  4],
        [112, 100, 25, 12]
    ])
    
    # Max Pool (2x2)
    max_data = np.array([
        [20, 30],
        [112, 37]
    ])
    
    # Avg Pool (2x2)
    avg_data = np.array([
        [13,  8],
        [79, 19.5]
    ])
    
    # Colors for the 4 pooling quadrants
    region_colors = [
        ['#ffadad', '#a2d2ff'], 
        ['#fdffb6', '#caffbf']
    ]
    
    # Draw grids
    draw_grid(ax, 0, 4, input_data, "Input Feature Map\n(4x4)", region_colors)
    draw_grid(ax, 6, 5.5, max_data, "Max Pooling\n(2x2)", region_colors)
    draw_grid(ax, 6, 2.0, avg_data, "Average Pooling\n(2x2)", region_colors)
    
    # Draw logic arrows
    ax.annotate('', xy=(5.5, 4.5), xytext=(4.3, 3.0), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7))
    ax.annotate('', xy=(5.5, 1.0), xytext=(4.3, 2.0), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7))
    
    ax.text(4.8, 3.8, 'Max\n(stride 2)', ha='right', va='bottom', fontsize=9, rotation=35)
    ax.text(4.8, 1.3, 'Average\n(stride 2)', ha='right', va='top', fontsize=9, rotation=-35)

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 6.5)
    ax.axis('off')
    
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.suptitle("CNN Pooling Layers")
    fig.savefig(out_dir / f'{get_name()}.pdf', bbox_inches='tight', pad_inches=0.05)

if __name__ == '__main__':
    generate_pooling_types_plot()

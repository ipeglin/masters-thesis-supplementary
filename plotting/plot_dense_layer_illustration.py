import os
import matplotlib.pyplot as plt
import numpy as np

from plotting.plot_config import FIG_WIDTH_INCHES, get_figs_output_dir
from lib.fs.get_script_name import get_name

def draw_layer(ax, x_pos, num_nodes, node_radius=0.12, color='blue', label='Layer'):
    # Center y-positions around 0
    y_positions = np.linspace(-num_nodes/2 + 0.5, num_nodes/2 - 0.5, num_nodes)
    for y in y_positions:
        circle = plt.Circle((x_pos, y), node_radius, color=color, zorder=3, ec='black', linewidth=1)
        ax.add_patch(circle)
    
    # Add layer label
    max_y = max(y_positions) if len(y_positions) > 0 else 0
    ax.text(x_pos, num_nodes/2 + 0.3, label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    return x_pos, y_positions

def plot_dense_network():
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, FIG_WIDTH_INCHES * 0.6))

    layers = [
        {"num": 5, "name": "Input Layer", "color": "lavender"},
        {"num": 7, "name": "Hidden Layer 1", "color": "honeydew"},
        {"num": 7, "name": "Hidden Layer 2", "color": "honeydew"},
        {"num": 3, "name": "Output Layer", "color": "mistyrose"}
    ]

    x_spacing = 3.0
    node_radius = 0.15
    layer_data = []

    # Draw nodes
    current_x = 0
    for i, layer_info in enumerate(layers):
        x, y_list = draw_layer(
            ax, 
            current_x, 
            layer_info["num"], 
            node_radius, 
            layer_info["color"], 
            layer_info["name"]
        )
        layer_data.append((x, y_list))
        current_x += x_spacing

    # Draw fully connected weights
    for idx in range(len(layer_data) - 1):
        x1, y1_list = layer_data[idx]
        x2, y2_list = layer_data[idx + 1]
        
        for y1 in y1_list:
            for y2 in y2_list:
                ax.plot([x1, x2], [y1, y2], color='gray', linestyle='-', alpha=0.25, linewidth=0.8, zorder=1)

    ax.set_aspect('equal')
    ax.axis('off')
    
    # Determine bounds based on sizes
    max_nodes = max(l.get("num", 0) for l in layers)
    last_x = layer_data[-1][0]
    ax.set_xlim(-1.5, last_x + 1.5)
    ax.set_ylim(-max_nodes/2 - 0.5, max_nodes/2 + 1)
    
    # Save figure
    script_name = get_name()
    out_dir = get_figs_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, f"{script_name}.pdf")
    plt.tight_layout()
    plt.suptitle('Shallow Dense-layered Neural Network')
    plt.savefig(out_path, format="pdf", bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    plot_dense_network()
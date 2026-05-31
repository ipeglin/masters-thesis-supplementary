import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib.fs.get_script_name import get_name
from plotting import plot_config
from plotting.plot_config import FIG_WIDTH_INCHES, get_figs_output_dir

def prism_verts(x, y, z, dx, dy, dz):
    v = [
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
    ]
    return [
        [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]], 
        [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]], 
        [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]
    ]

def generate_cnn_3d_plot():
    out_dir = get_figs_output_dir()
    
    fig = plt.figure(figsize=(FIG_WIDTH_INCHES * 1.2, FIG_WIDTH_INCHES * 0.55))
    ax = fig.add_subplot(111, projection='3d')
    
    # Layers for DenseNet201 (conceptual)
    # [label, dims, depth(x), height(z), width(y), color]
    _cp = plot_config.CONCEPT_PALETTE
    layers = [
        ("Input", "224x224x3", 0.2, 10, 10, _cp["wash"]),
        ("Stem\nConv/Pool", "56x56x64", 0.8, 8, 8, _cp["pale"]),
        ("Dense\nBlock 1", "56x56x256", 1.5, 8, 8, _cp["accent_a"]),
        ("Trans 1", "28x28x128", 0.5, 6, 6, _cp["mid"]),
        ("Dense\nBlock 2", "28x28x512", 2.0, 6, 6, _cp["accent_a"]),
        ("Trans 2", "14x14x256", 0.5, 4, 4, _cp["mid"]),
        ("Dense\nBlock 3", "14x14x1792", 3.0, 4, 4, _cp["accent_a"]),
        ("Trans 3", "7x7x896", 0.5, 2, 2, _cp["mid"]),
        ("Dense\nBlock 4", "7x7x1920", 2.0, 2, 2, _cp["accent_a"]),
        ("GAP", "1x1x1920", 0.5, 1, 1, _cp["mid"]),
        ("Feature\nVector", "1920", 2.0, 0.5, 0.5, _cp["accent_b"]),
    ]
    
    pos_x = 0
    gap = 2.0  # Reduced to fit narrower width
    
    all_faces = []
    all_colors = []
    labels = []
    
    for i, (label, dims, d, h, w, c) in enumerate(layers):
        x = pos_x - d/2
        y = -w/2
        z = -h/2
        faces = prism_verts(x, y, z, d, w, h)
        all_faces.extend(faces)
        all_colors.extend([c] * 6)
        
        if i < len(layers) - 1:
            ax.plot([pos_x + d/2 + 0.1, pos_x + d/2 + gap - 0.1], [0, 0], [0, 0], color='k', lw=1.5, linestyle='--')
            pos_x += d/2 + gap + layers[i+1][2]/2

    poly = Poly3DCollection(all_faces, facecolors=all_colors, edgecolors='k', linewidths=0.5, alpha=0.8)
    ax.add_collection3d(poly)

    ax.set_xlim(-2, pos_x)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-8, 6)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-65)  # Slight angle adjustment
    ax.set_title('DenseNet201 Head Architecture', pad=0)
    
    legend_elements = [
        mpatches.Patch(facecolor=_cp["wash"],     edgecolor='k', label='Input Image'),
        mpatches.Patch(facecolor=_cp["pale"],     edgecolor='k', label='Conv / Pool'),
        mpatches.Patch(facecolor=_cp["accent_a"], edgecolor='k', label='Dense Block'),
        mpatches.Patch(facecolor=_cp["mid"],      edgecolor='k', label='Transition / GAP'),
        mpatches.Patch(facecolor=_cp["accent_b"], edgecolor='k', label='Feature Vector'),
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.95, 0.5), ncol=1, frameon=False, fontsize=8)
    
    fig.subplots_adjust(left=0, right=0.85, top=0.95, bottom=0)
    fig.savefig(out_dir / f'{get_name()}.pdf', bbox_inches='tight')

if __name__ == '__main__':
    generate_cnn_3d_plot()


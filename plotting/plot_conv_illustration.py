import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib.fs.get_script_name import get_name
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

def generate_conv_plot():
    out_dir = get_figs_output_dir()
    # Use full square figure size. Matplotlib fits the 3D axes to this. 
    # bbox_inches='tight' will crop the top and bottom empty space perfectly, 
    # preserving the full width without horizontal squishing.
    fig = plt.figure(figsize=(FIG_WIDTH_INCHES, FIG_WIDTH_INCHES))
    ax = fig.add_subplot(111, projection='3d')

    # Input tensor
    f_in = prism_verts(0, -3, -3, 1, 6, 6)
    # Output tensor
    f_out = prism_verts(6, -2, -2, 1, 4, 4)
    
    # Receptive field in input
    rf = prism_verts(0.9, 0, 0, 0.2, 2, 2)
    # Output pixel
    px = prism_verts(5.9, -0.5, -0.5, 0.2, 1, 1)

    # Lines connecting receptive field corners to output pixel corners
    for iy, iz in [(0, 0), (2, 0), (0, 2), (2, 2)]:
        oy = -0.5 if iy == 0 else 0.5
        oz = -0.5 if iz == 0 else 0.5
        ax.plot([1.1, 5.9], [iy, oy], [iz, oz], 'k--', lw=1, alpha=0.5)

    ax.add_collection3d(Poly3DCollection(f_in, facecolors='#d3d3d3', edgecolors='k', alpha=0.3, linewidths=0.5))
    ax.add_collection3d(Poly3DCollection(f_out, facecolors='#add8e6', edgecolors='k', alpha=0.3, linewidths=0.5))
    ax.add_collection3d(Poly3DCollection(rf, facecolors='r', edgecolors='k', alpha=0.6, linewidths=1.0))
    ax.add_collection3d(Poly3DCollection(px, facecolors='r', edgecolors='k', alpha=0.6, linewidths=1.0))

    ax.text(-1.5, 0, -4.5, 'Input Feature Map', zdir='x', ha='center', va='top', fontsize=9)
    ax.text(5.5, 0, -3.0, 'Output Feature Map', zdir='x', ha='center', va='top', fontsize=9)

    legend_elements = [mpatches.Patch(facecolor='r', edgecolor='k', alpha=0.6, label='Kernel Application')]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.85, 0.5), frameon=False, fontsize=9)

    ax.set_xlim(-1, 8)
    ax.set_ylim(-4.5, 4.5)
    ax.set_zlim(-4.5, 4.5)
    ax.set_box_aspect((12, 9, 9))  # Elongate X slightly so it spans wider horizontally
    ax.set_axis_off()
    
    # Zoom in camera to crop out matplotlib 3D white margins
    ax.dist = 7
    ax.view_init(elev=22, azim=-55)
    ax.set_title('Convolutional Layer Concept', pad=-20)
    
    fig.subplots_adjust(left=0, right=0.85, top=1.0, bottom=0)
    fig.savefig(out_dir / f'{get_name()}.pdf', bbox_inches='tight', pad_inches=0.0)

if __name__ == '__main__':
    generate_conv_plot()

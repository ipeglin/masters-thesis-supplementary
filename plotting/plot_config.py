"""
Global matplotlib configuration for consistent thesis figures.
Optimized for 12pt font and 150mm linewidth with a MATLAB-like aesthetic.
"""

import inspect
from pathlib import Path

import matplotlib as mpl
from cycler import cycler

# Assuming REPO_ROOT logic is handled in your project structure
try:
    from lib.fs.project_config import REPO_ROOT
except ImportError:
    # Fallback if the lib path isn't set up in the environment
    REPO_ROOT = Path(__file__).resolve().parents[1]


def get_figs_output_dir():
    caller_path = inspect.stack()[1].filename
    caller_name = Path(caller_path).stem
    output_dir = REPO_ROOT / "figures" / caller_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# --- LaTeX Geometry Setup ---
LINEWIDTH_MM = 150.0
INCHES_PER_MM = 1 / 25.4
FIG_WIDTH_INCHES = LINEWIDTH_MM * INCHES_PER_MM
ASPECT_RATIO = 0.75
FIG_HEIGHT_INCHES = FIG_WIDTH_INCHES * ASPECT_RATIO

# --- Font & LaTeX Settings ---
mpl.rcParams.update(
    {
        "font.family": "serif",
        # cmr10 is the Computer Modern Roman font
        "font.serif": ["cmr10", "Computer Modern Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    }
)

FONT_SIZE = 12
mpl.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": 11,
        "figure.titlesize": FONT_SIZE,
    }
)

# --- MATLAB-like "Closed Box" Spines & Ticks ---
mpl.rcParams.update(
    {
        # 1. Show all four boundaries (The "Box")
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.8,
        # 2. Configure Ticks to point inward on all sides
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,  # Show ticks on top
        "ytick.right": True,  # Show ticks on right
        # Optional: ensure minor ticks also follow this (if used)
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
    }
)

# --- Figure & Layout ---
mpl.rcParams.update(
    {
        "figure.figsize": (FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES),
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.08,
        "figure.constrained_layout.w_pad": 0.08,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# MATLAB Color Cycle
matlab_colors = [
    "#0072BD",
    "#D95319",
    "#EDB120",
    "#7E2F8E",
    "#77AC30",
    "#4DBEEE",
    "#A2142F",
]
mpl.rcParams["axes.prop_cycle"] = cycler(color=matlab_colors)
mpl.rcParams["lines.linewidth"] = 1.25

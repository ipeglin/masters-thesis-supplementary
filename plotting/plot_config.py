"""
Global matplotlib configuration for consistent thesis figures.
Optimized for 12pt font and 150mm linewidth.
"""
import inspect
import os
from pathlib import Path

import matplotlib as mpl

# Use the REPO_ROOT logic from your original config
from lib.fs.project_config import REPO_ROOT


def get_figs_output_dir():
    caller_path = inspect.stack()[1].filename
    caller_name = Path(caller_path).stem
    output_dir = REPO_ROOT / "figures" / caller_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# --- LaTeX Geometry Setup ---
# Based on packages.sty: total={150mm, 245mm}
LINEWIDTH_MM = 150.0
INCHES_PER_MM = 1 / 25.4
FIG_WIDTH_INCHES = LINEWIDTH_MM * INCHES_PER_MM

# We increase the aspect ratio from 0.618 to 0.75.
# This provides more vertical space for titles and X-axis labels to breathe.
ASPECT_RATIO = 0.75 
FIG_HEIGHT_INCHES = FIG_WIDTH_INCHES * ASPECT_RATIO

# --- Global rcParams ---
mpl.rcParams["figure.figsize"] = (FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES)

# Font settings
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["cmr10", "STIXGeneral", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "cm"

FONT_SIZE = 12  # Matches 12pt thesis document class
mpl.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": 11,      # Slightly smaller to prevent legend overlap
    "figure.titlesize": FONT_SIZE,
})

# --- Layout Engine ---
# Constrained layout is more robust than tight_layout for preventing overlaps.
mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["figure.constrained_layout.h_pad"] = 0.08  # Padding between elements
mpl.rcParams["figure.constrained_layout.w_pad"] = 0.08

# --- Automatic Save Settings ---
# Replaced manual save arguments with global defaults
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.05

# Additional styling
mpl.rcParams["axes.unicode_minus"] = False
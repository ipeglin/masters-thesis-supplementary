"""
Global matplotlib configuration for consistent thesis figures.
Optimized for 12pt font and 150mm linewidth with a MATLAB-like aesthetic.
"""

import inspect
import shutil
from pathlib import Path

import matplotlib as mpl
from cycler import cycler

# Full LaTeX rendering needs a `latex` binary on PATH; HPC nodes often lack it.
# Fall back to mathtext (cm fontset) when absent so figures still render.
_HAS_LATEX = shutil.which("latex") is not None

try:
    from lib.fs.project_config import REPO_ROOT
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[1]


_PLOT_SUBDIRS = {"conceptual_figures", "implementation_figures", "result_figures"}


def get_figs_output_dir():
    caller_path = Path(inspect.stack()[1].filename)
    caller_name = caller_path.stem
    parent_name = caller_path.parent.name
    if parent_name in _PLOT_SUBDIRS:
        output_dir = REPO_ROOT / "figures" / parent_name / caller_name
    else:
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
        "font.serif": ["cmr10", "Computer Modern Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": _HAS_LATEX,
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
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
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

# ---------------------------------------------------------------------------
# Palette system
# ---------------------------------------------------------------------------

# Reserved: encode control vs anhedonic subjects ONLY. Never use elsewhere.
COHORT = {
    "control": "#0072BD",
    "anhedonic": "#D95319",
}

# Non-cohort categorical data (signals, modes, arbitrary multi-line plots).
# Okabe-Ito + Tol supplements, with entries too close to cohort removed.
# Excluded: #56B4E9, #0072B2, #E69F00, #D55E00 (too similar to cohort blue/orange).
RESULTS_PALETTE = [
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#000000",  # black
    "#882255",  # wine
    "#44AA99",  # teal
]

# Conceptual / schematic figures: grayscale-dominant + two accent colors.
CONCEPT_PALETTE = {
    "ink":      "#222222",
    "mid":      "#555555",
    "soft":     "#888888",
    "pale":     "#BBBBBB",
    "wash":     "#EEEEEE",
    "accent_a": "#44AA99",  # teal
    "accent_b": "#AA4499",  # purple
}

# Colormap conventions
CMAP_SEQ  = "viridis"   # sequential intensity (heatmaps, spectrograms)
CMAP_SEQ2 = "magma"     # alt sequential when viridis already used in figure family
CMAP_DIV  = "RdBu_r"    # diverging (correlation, t-stat, signed effects)
CMAP_CONF = "Greys"     # confusion matrices

# Deprecated alias — points at RESULTS_PALETTE for backwards compat.
matlab_colors = RESULTS_PALETTE


def cohort_color(label):
    """Return hex color for a cohort label.

    Accepts "control"/"anhedonic" (str) or 0/1 (int, control=0).
    """
    if label in (0, "control"):
        return COHORT["control"]
    if label in (1, "anhedonic"):
        return COHORT["anhedonic"]
    raise ValueError(f"Unknown cohort label: {label!r}. Use 'control'/'anhedonic' or 0/1.")


def use_results_cycle():
    """Set axes.prop_cycle to RESULTS_PALETTE for the current script."""
    mpl.rcParams["axes.prop_cycle"] = cycler(color=RESULTS_PALETTE)


def use_concept_style():
    """Set default line color to ink and prop_cycle to concept accents."""
    mpl.rcParams["axes.prop_cycle"] = cycler(
        color=[CONCEPT_PALETTE["accent_a"], CONCEPT_PALETTE["accent_b"],
               CONCEPT_PALETTE["mid"], CONCEPT_PALETTE["pale"]]
    )


# Default cycle: results palette (avoids leaking cohort colors into arbitrary plots)
use_results_cycle()
mpl.rcParams["lines.linewidth"] = 1.25

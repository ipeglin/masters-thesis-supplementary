"""
Global matplotlib configuration for consistent thesis figures.

Import this module before any other plotting code:
    import plot_config  # noqa: F401

Also exposes REPO_ROOT (the directory containing this file) so scripts in
subdirectories can build repo-root-relative output paths.
"""

from pathlib import Path

import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parent

# --- Font ---
# cmr10 is Computer Modern Roman bundled with matplotlib (the LaTeX default).
# STIXGeneral is the fallback; it is designed to match CM aesthetics.
mpl.pyplot.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["cmr10", "STIXGeneral", "DejaVu Serif"]

FONT_SIZE = 12  # points (matches standard LaTeX 12pt document class)
mpl.rcParams["font.size"] = FONT_SIZE
mpl.rcParams["axes.titlesize"] = FONT_SIZE
mpl.rcParams["axes.labelsize"] = FONT_SIZE
mpl.rcParams["xtick.labelsize"] = FONT_SIZE
mpl.rcParams["ytick.labelsize"] = FONT_SIZE
mpl.rcParams["legend.fontsize"] = FONT_SIZE
mpl.rcParams["figure.titlesize"] = FONT_SIZE

# Keep math rendering consistent with the body font
mpl.rcParams["mathtext.fontset"] = "cm"

# --- Axes ---
mpl.rcParams["axes.unicode_minus"] = False  # avoids glyph-missing warnings with cmr10

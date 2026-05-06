"""
Variational Mode Decomposition — step-by-step illustration.
Restored annotations and spacing with updated mode colors.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from lib.fs.get_script_name import get_name
from lib.fs.project_config import REPO_ROOT
from plotting import plot_config
from plotting.plot_config import FIG_WIDTH_INCHES, get_figs_output_dir

out_dir = get_figs_output_dir()

# --------------------------------------------------------------------------- #
# Signal parameters
# --------------------------------------------------------------------------- #
FS = 1000
T = 2.0
FM = 6.0
AM_DEPTH = 0.35

MODE_FREQS = [50, 150, 280]
MODE_AMPS = [1.0, 0.70, 0.50]

# Updated Palette: Distinctly different from MATLAB blue (#0072BD)
MODE_COLORS = ["#d95f02", "#7570b3", "#1b9e77"]
ALPHA_FILL = 0.18


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fwhm_bounds(freqs, spectrum):
    """Return (f_left, f_right, half_max) of the dominant spectral peak."""
    pk = np.argmax(spectrum)
    hm = spectrum[pk] / 2.0
    left = (
        freqs[np.where(spectrum[:pk] < hm)[0][-1]]
        if np.any(spectrum[:pk] < hm)
        else freqs[0]
    )
    right_tail = spectrum[pk:]
    right = (
        freqs[pk + np.where(right_tail < hm)[0][0]]
        if np.any(right_tail < hm)
        else freqs[-1]
    )
    return left, right, hm


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def generate_vmd_illustration():
    t = np.linspace(0, T, int(T * FS), endpoint=False)
    N = len(t)

    modes = [
        a * (1.0 + AM_DEPTH * np.cos(2 * np.pi * FM * t)) * np.cos(2 * np.pi * w * t)
        for a, w in zip(MODE_AMPS, MODE_FREQS)
    ]
    f = sum(modes)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1.0 / FS))

    F_twosided = np.abs(np.fft.fftshift(np.fft.fft(f))) / N
    f_analytic = hilbert(f)
    F_analytic_abs = np.abs(np.fft.fftshift(np.fft.fft(f_analytic))) / N

    bb_specs = []
    for wk, uk in zip(MODE_FREQS, modes):
        uk_a = hilbert(uk)
        uk_bb = uk_a * np.exp(-1j * 2 * np.pi * wk * t)
        bb_specs.append(np.abs(np.fft.fftshift(np.fft.fft(uk_bb))) / N)

    # ── Figure layout (Original hspace and wspace restored) ────────────────── #
    fig = plt.figure(figsize=(FIG_WIDTH_INCHES * 1.2, FIG_WIDTH_INCHES * 1.85))
    gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.72, wspace=1)

    t_show = slice(0, int(0.4 * FS))
    xlim_sp = 340
    xlim_bb = 50

    # ── Row 0: Composite signal ────────────────────────────────────────────── #
    ax_f = fig.add_subplot(gs[0, :])
    ax_f.plot(t[t_show], f[t_show])
    ax_f.set_title(r"Composite signal $\mathbf{x}(t) = \sum_{k=1}^{M} u_m(t)$")
    ax_f.set_xlabel("Time (s)")
    ax_f.set_ylabel("Amplitude")

    # ── Row 1: Spectra (With restored "negative frequencies" annotation) ───── #
    ax_2s = fig.add_subplot(gs[1, :3])
    ax_2s.plot(freqs, F_twosided)
    ax_2s.set_title(r"Two-sided spectrum $|F(\omega)|$")
    ax_2s.set_xlabel("Frequency (Hz)")
    ax_2s.set_ylabel("Magnitude")
    ax_2s.set_xlim(-xlim_sp, xlim_sp)

    ax_1s = fig.add_subplot(gs[1, 3:], sharey=ax_2s)
    ax_1s.plot(freqs, F_analytic_abs)
    ax_1s.set_title(r"Analytic spectrum $|F_{\!a}(\omega)|$")
    ax_1s.set_xlabel("Frequency (Hz)")
    ax_1s.set_xlim(-xlim_sp, xlim_sp)
    ax_1s.annotate(
        "negative\nfrequencies\nzeroed",
        xy=(-160, F_analytic_abs.max() * 0.06),
        xycoords="data",
        xytext=(0.18, 0.78),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
        fontsize=8,
        color="k",
        ha="center",
        va="center",
    )

    # ── Row 2: Baseband Spectra (With restored arrows and formulas) ────────── #
    col_slices = [slice(0, 2), slice(2, 4), slice(4, 6)]
    for k, (col_sl, color, wk, bbs) in enumerate(
        zip(col_slices, MODE_COLORS, MODE_FREQS, bb_specs)
    ):
        ax = fig.add_subplot(gs[2, col_sl])
        mask = (freqs >= -xlim_bb) & (freqs <= xlim_bb)
        f_win = freqs[mask]
        bbs_win = bbs[mask]

        ax.plot(f_win, bbs_win, color=color, lw=1.0)
        ax.fill_between(f_win, bbs_win, alpha=ALPHA_FILL, color=color)

        # Restored FWHM span bracket
        fl, fr, hm = _fwhm_bounds(f_win, bbs_win)
        ax.annotate(
            "",
            xy=(fr, hm),
            xytext=(fl, hm),
            arrowprops=dict(
                arrowstyle="<->", color="gray", lw=0.9, shrinkA=0, shrinkB=0
            ),
        )

        # Restored bandwidth formula text
        ax.text(
            0.75,
            0.58,
            r"$B_m = \|\partial_t\tilde{u}_m\|_2^2$",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="k",
            transform=ax.transAxes,
        )

        ax.set_title(
            rf"Mode {k + 1} at baseband" + "\n" + rf"$\omega_{k + 1} = {wk}$ Hz"
        )
        ax.set_xlabel("Frequency (Hz)")
        if k == 0:
            ax.set_ylabel("Magnitude")
        ax.set_xlim(-xlim_bb, xlim_bb)

    # ── Row 3: Time-domain modes ────────────────────────────────────────────── #
    for k, (col_sl, color) in enumerate(zip(col_slices, MODE_COLORS)):
        ax = fig.add_subplot(gs[3, col_sl])
        ax.plot(t[t_show], modes[k][t_show], color=color, lw=0.8)
        ax.set_title(rf"$u_{k + 1}(t)$")
        ax.set_xlabel("Time (s)")
        if k == 0:
            ax.set_ylabel("Amplitude")

    fig.savefig(out_dir / f"{get_name()}.pdf")


if __name__ == "__main__":
    generate_vmd_illustration()
    print("VMD illustration generated with all annotations and proper spacing.")

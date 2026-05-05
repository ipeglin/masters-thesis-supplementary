"""
Variational Mode Decomposition — step-by-step illustration.

Shows:
  1. Composite signal f(t) = sum_m u_m(t) and its two-sided spectrum.
  2. Analytic signal: negative frequencies zeroed → single-sided spectrum.
  3. Harmonic mixing: each mode's analytic signal shifted to baseband
     by multiplication with e^{-j2π ω_m t}; bandwidth visible at 0.
  4. Decomposed modes u_m(t) in time domain.
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

FS = 1000          # sampling frequency (Hz)
T = 2.0            # duration (s)
FM = 6.0           # AM modulation frequency (Hz) — gives visible sideband bandwidth
AM_DEPTH = 0.35    # AM modulation depth

MODE_FREQS = [50, 150, 280]     # carrier frequencies (Hz)
MODE_AMPS  = [1.0, 0.70, 0.50]  # carrier amplitudes

# Colorbrewer-inspired palette, printer-friendly
MODE_COLORS = ['#2166ac', '#d6604d', '#4dac26']
ALPHA_FILL  = 0.18


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _fwhm_bounds(freqs, spectrum):
    """Return (f_left, f_right, half_max) of the dominant spectral peak."""
    pk = np.argmax(spectrum)
    hm = spectrum[pk] / 2.0
    left = freqs[np.where(spectrum[:pk] < hm)[0][-1]] if np.any(spectrum[:pk] < hm) else freqs[0]
    right_tail = spectrum[pk:]
    right = freqs[pk + np.where(right_tail < hm)[0][0]] if np.any(right_tail < hm) else freqs[-1]
    return left, right, hm


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def generate_vmd_illustration():
    t  = np.linspace(0, T, int(T * FS), endpoint=False)
    N  = len(t)

    # AM-modulated sinusoidal modes — gives visible spectral bandwidth
    modes = [
        a * (1.0 + AM_DEPTH * np.cos(2 * np.pi * FM * t)) * np.cos(2 * np.pi * w * t)
        for a, w in zip(MODE_AMPS, MODE_FREQS)
    ]
    f = sum(modes)

    # Shared frequency axis (centred)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1.0 / FS))

    # ── Spectrum of composite (two-sided) ────────────────────────────────── #
    F_raw      = np.fft.fft(f)
    F_twosided = np.abs(np.fft.fftshift(F_raw)) / N

    # ── Analytic signal of composite (one-sided via scipy Hilbert) ────────── #
    # scipy.signal.hilbert gives the full analytic signal in time domain;
    # its FFT has energy only at non-negative frequencies.
    f_analytic     = hilbert(f)
    F_analytic_abs = np.abs(np.fft.fftshift(np.fft.fft(f_analytic))) / N

    # ── Baseband spectra: s_a_m(t) · e^{-j2π ω_m t} ─────────────────────── #
    bb_specs = []
    for wk, uk in zip(MODE_FREQS, modes):
        uk_a  = hilbert(uk)
        uk_bb = uk_a * np.exp(-1j * 2 * np.pi * wk * t)
        bb_specs.append(np.abs(np.fft.fftshift(np.fft.fft(uk_bb))) / N)

    # ── Figure layout ──────────────────────────────────────────────────────── #
    fig = plt.figure(
        figsize=(FIG_WIDTH_INCHES * 1.2, FIG_WIDTH_INCHES * 1.85)
        )
    gs  = gridspec.GridSpec(4, 6, figure=fig, hspace=0.72, wspace=1)

    t_show  = slice(0, int(0.4 * FS))   # first 0.4 s for time-domain panels
    xlim_sp = 340                        # frequency window for full spectra (Hz)
    xlim_bb = 50                         # frequency window for baseband (Hz)

    # ── Row 0: composite signal (time domain) ──────────────────────────────── #
    ax_f = fig.add_subplot(gs[0, :])
    ax_f.plot(t[t_show], f[t_show], 'k-', lw=0.8)
    ax_f.set_title(r'Composite signal $\mathbf{x}(t) = \sum_{k=1}^{M} u_m(t)$')
    ax_f.set_xlabel('Time (s)')
    ax_f.set_ylabel('Amplitude')
    _clean_ax(ax_f)

    # ── Row 1: two-sided  |  analytic (one-sided) spectrum ─────────────────── #
    ax_2s = fig.add_subplot(gs[1, :3])
    ax_2s.plot(freqs, F_twosided, 'k-', lw=1.0)
    ax_2s.set_title(r'Two-sided spectrum $|F(\omega)|$')
    ax_2s.set_xlabel('Frequency (Hz)')
    ax_2s.set_ylabel('Magnitude')
    ax_2s.set_xlim(-xlim_sp, xlim_sp)
    _clean_ax(ax_2s)

    ax_1s = fig.add_subplot(gs[1, 3:], sharey=ax_2s)
    ax_1s.plot(freqs, F_analytic_abs, 'k-', lw=1.0)
    ax_1s.set_title(r'Analytic spectrum $|F_{\!a}(\omega)|$')
    ax_1s.set_xlabel('Frequency (Hz)')
    # ax_1s.set_ylabel('Magnitude')
    ax_1s.set_xlim(-xlim_sp, xlim_sp)
    ax_1s.annotate(
        'negative\nfrequencies\nzeroed',
        xy=(-160, F_analytic_abs.max() * 0.06),
        xycoords='data',
        xytext=(0.18, 0.78),
        textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='k', lw=0.8),
        fontsize=8, color='k', ha='center', va='center',
    )
    _clean_ax(ax_1s)

    # ── Row 2: baseband spectra ─────────────────────────────────────────────── #
    col_slices = [slice(0, 2), slice(2, 4), slice(4, 6)]

    for k, (col_sl, color, wk, bbs) in enumerate(
        zip(col_slices, MODE_COLORS, MODE_FREQS, bb_specs)
    ):
        ax = fig.add_subplot(gs[2, col_sl])

        mask = (freqs >= -xlim_bb) & (freqs <= xlim_bb)
        f_win  = freqs[mask]
        bbs_win = bbs[mask]

        ax.plot(f_win, bbs_win, color=color, lw=1.0)
        ax.fill_between(f_win, bbs_win, alpha=ALPHA_FILL, color=color)

        # FWHM span bracket at half-maximum height
        fl, fr, hm = _fwhm_bounds(f_win, bbs_win)
        ax.annotate(
            '', xy=(fr, hm), xytext=(fl, hm),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=0.9,
                            shrinkA=0, shrinkB=0),
        )
        # Label in axes-fraction coordinates so it never collides with data
        ax.text(
            0.8, 0.58,
            r'$B_m = \|\partial_t\tilde{u}_m\|_2^2$',
            ha='center', va='bottom', fontsize=7.5, color='k',
            transform=ax.transAxes,
        )

        ax.set_title(
            rf'Mode {k + 1} at baseband'
            '\n'
            rf'$\omega_{k + 1} = {wk}$ Hz'
        )
        ax.set_xlabel('Frequency (Hz)')
        if k == 0:
            ax.set_ylabel('Magnitude')
        ax.set_xlim(-xlim_bb, xlim_bb)
        _clean_ax(ax)

    # ── Row 3: time-domain modes ────────────────────────────────────────────── #
    for k, (col_sl, color) in enumerate(zip(col_slices, MODE_COLORS)):
        ax = fig.add_subplot(gs[3, col_sl])
        ax.plot(t[t_show], modes[k][t_show], color=color, lw=0.8)
        ax.set_title(rf'$u_{k + 1}(t)$')
        ax.set_xlabel('Time (s)')
        if k == 0:
            ax.set_ylabel('Amplitude')
        _clean_ax(ax)

    fig.savefig(out_dir / f'{get_name()}.pdf')


if __name__ == '__main__':
    generate_vmd_illustration()
    print("VMD illustration generated.")

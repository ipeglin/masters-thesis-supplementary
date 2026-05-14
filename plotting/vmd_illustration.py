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
from lib.mvmd.mvmd import mvmd
from plotting import plot_config
from plotting.plot_config import FIG_WIDTH_INCHES, get_figs_output_dir

out_dir = get_figs_output_dir()

# --------------------------------------------------------------------------- #
# Signal parameters
# --------------------------------------------------------------------------- #

FS = 2000          # sampling frequency (Hz)
T = 1.0            # duration (s)

MODE_FREQS = [2, 24, 288]       # carrier frequencies (Hz)
MODE_AMPS  = [1.0, 0.25, 1/16]  # carrier amplitudes
NOISE_STD  = 0.1

ALPHA      = 2000

# Colorbrewer-inspired palette, printer-friendly
MODE_COLORS = ['#d6604d', '#4dac26', "#ac26a8"]
ALPHA_FILL  = 0.18


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

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

    # Noisy triharmonic signal: cos(4pi t) + 1/4 cos(48pi t) + 1/16 cos(576pi t) + N
    # 4pi t -> 2 Hz, 48pi t -> 24 Hz, 576pi t -> 288 Hz
    true_modes = [
        a * np.cos(2 * np.pi * w * t)
        for a, w in zip(MODE_AMPS, MODE_FREQS)
    ]
    f_clean = sum(true_modes)
    np.random.seed(42)
    noise = np.random.normal(0, NOISE_STD, N)
    f = f_clean + noise

    # Use actual MVMD
    modes_list, modes_hat, omega = mvmd(f.reshape(1, N), len(MODE_FREQS), ALPHA, init=0, tau=0)
    modes = [modes_list[k, 0, :] for k in range(len(MODE_FREQS))]
    est_freqs = omega[-1, :] * FS

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
    for wk, uk in zip(est_freqs, modes):
        uk_a  = hilbert(uk)
        uk_bb = uk_a * np.exp(-1j * 2 * np.pi * wk * t)
        bb_specs.append(np.abs(np.fft.fftshift(np.fft.fft(uk_bb))) / N)

    # ── Figure layout ──────────────────────────────────────────────────────── #
    fig = plt.figure(
        figsize=(FIG_WIDTH_INCHES * 1.2, FIG_WIDTH_INCHES * 1.85)
        )
    gs  = gridspec.GridSpec(4, 6, figure=fig, hspace=0.72, wspace=1)

    xlim_sp = 340                        # frequency window for full spectra (Hz)
    xlim_bb = 50                         # frequency window for baseband (Hz)

    # ── Row 0: composite signal (time domain) ──────────────────────────────── #
    ax_f = fig.add_subplot(gs[0, :])
    ax_f.plot(t, f)
    ax_f.set_title(r'Composite signal $\mathbf{x}(t)$')
    ax_f.set_xlabel('Time (s)')
    ax_f.set_ylabel('Amplitude')
    ax_f.set_xlim(0, T)
    
    col_slices = [slice(0, 2), slice(2, 4), slice(4, 6)]

    # ── Row 1: time-domain modes ────────────────────────────────────────────── #
    for k, (col_sl, color) in enumerate(zip(col_slices, MODE_COLORS)):
        ax = fig.add_subplot(gs[1, col_sl])
        ax.plot(t, modes[k], color=color)
        ax.set_title(rf'$u_{k + 1}(t)$')
        ax.set_xlabel('Time (s)')
        ax.set_xlim(0, T)
        if k == 0:
            ax.set_ylabel('Amplitude')
    
    # ── Row 2: spectrum of composite ───────────────────────────────────────── #
    ax_s_orig = fig.add_subplot(gs[2, :])
    r_freqs = np.fft.rfftfreq(N, 1.0 / FS)
    S_orig = np.abs(np.fft.rfft(f)) / N
    # Filter 0 freq for log scale
    valid = r_freqs > 0
    ax_s_orig.plot(r_freqs[valid], S_orig[valid], color='k')
    ax_s_orig.set_title(r'Spectrum of composite signal')
    ax_s_orig.set_xlabel('Frequency (Hz)')
    ax_s_orig.set_ylabel('Magnitude')
    ax_s_orig.set_xscale('log')
    ax_s_orig.set_yscale('log')
    ax_s_orig.set_xlim(left=1, right=xlim_sp)

    # ── Row 3: modes spectra ───────────────────────────────────────────────── #
    ax_s_modes = fig.add_subplot(gs[3, :], sharey=ax_s_orig, sharex=ax_s_orig)
    for m, mode in enumerate(modes):
        S_mode = np.abs(np.fft.rfft(mode)) / N
        ax_s_modes.plot(r_freqs[valid], S_mode[valid], color=MODE_COLORS[m], label=f'Mode {m+1}')
    ax_s_modes.set_title(r'Spectra of derived modes')
    ax_s_modes.set_xlabel('Frequency (Hz)')
    ax_s_modes.set_ylabel('Magnitude')
    ax_s_modes.set_xscale('log')
    ax_s_modes.set_yscale('log')
    ax_s_modes.set_xlim(left=1, right=xlim_sp)
    ax_s_modes.legend(loc='upper right', fontsize=8)
        

    fig.savefig(out_dir / f'{get_name()}.pdf')

    # ── Baseband Figure ─────────────────────────────────────────────────────── #
    fig_bb = plt.figure(figsize=(FIG_WIDTH_INCHES * 1.2, FIG_WIDTH_INCHES * 0.46))
    gs_bb  = gridspec.GridSpec(1, 6, figure=fig_bb, wspace=1)

    for k, (col_sl, color, wk, bbs) in enumerate(
        zip(col_slices, MODE_COLORS, est_freqs, bb_specs)
    ):
        ax = fig_bb.add_subplot(gs_bb[0, col_sl])

        mask = (freqs >= -xlim_bb) & (freqs <= xlim_bb)
        f_win  = freqs[mask]
        bbs_win = bbs[mask]

        ax.plot(f_win, bbs_win, color=color)
        ax.fill_between(f_win, bbs_win, alpha=ALPHA_FILL, color=color)

        fl, fr, hm = _fwhm_bounds(f_win, bbs_win)
        ax.annotate(
            '', xy=(fr, hm), xytext=(fl, hm),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=0.9,
                            shrinkA=0, shrinkB=0),
        )
        ax.text(
            0.8, 0.58,
            r'$B_m = \|\partial_t\tilde{u}_m\|_2^2$',
            ha='center', va='bottom', fontsize=7.5, color='k',
            transform=ax.transAxes,
        )

        ax.set_title(
            rf'Mode {k + 1} at baseband'
            '\n'
            rf'$\omega_{k + 1} = {wk:.1f}$ Hz'
        )
        ax.set_xlabel('Frequency (Hz)')
        if k == 0:
            ax.set_ylabel('Magnitude')
        ax.set_xlim(-xlim_bb, xlim_bb)

    fig_bb.savefig(out_dir / f'{get_name()}_baseband.pdf')


if __name__ == '__main__':
    generate_vmd_illustration()
    print("VMD illustration generated.")

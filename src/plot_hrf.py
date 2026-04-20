"""
Plot a realistic haemodynamic response function (HRF) with initial dip,
and its frequency-domain magnitude spectrum to illustrate the low-pass filter
characteristic of the BOLD HRF (Friston et al. 1998).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import plot_config  # noqa: F401 – applies global rcParams

import numpy as np
from scipy.special import gamma as gamma_fn
import matplotlib.pyplot as plt


def _gamma_pdf(t, a, b):
    """Gamma probability density function parameterised by shape a and scale b."""
    return (t ** (a - 1) * np.exp(-t / b)) / (b ** a * gamma_fn(a))


def hrf(t,
        a1=6.0, b1=1.0,         # main positive peak: shape & scale
        a2=16.0, b2=1.0,        # post-stimulus undershoot: shape & scale
        c=0.1,                   # undershoot ratio
        a0=1.5, b0=0.5, d=0.08  # initial-dip term: shape, scale, amplitude
        ):
    """
    Canonical double-gamma HRF with an initial dip.

    Parameters
    ----------
    t  : array_like, time in seconds (t >= 0)

    Returns
    -------
    h  : ndarray, normalised HRF amplitude
    """
    t = np.asarray(t, dtype=float)
    pos = np.where(t >= 0, t, 0.0)  # gamma is undefined for t < 0

    main = _gamma_pdf(pos, a1, b1) - c * _gamma_pdf(pos, a2, b2)
    dip  = -d * _gamma_pdf(pos, a0, b0)

    h = main + dip
    h /= np.abs(h).max()  # normalise to unit peak amplitude
    return h


def plot_hrf_time():
    t = np.linspace(0, 32, 500)
    h = hrf(t)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.3)

    ax.plot(t, h, color="#1f77b4", linewidth=2)

    ax.fill_between(t, h, 0, where=(h > 0), alpha=0.15, color="#1f77b4")
    ax.fill_between(t, h, 0, where=(h < 0), alpha=0.15, color="#d62728")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title("Haemodynamic Response Function")
    ax.set_xlim(t[0], t[-1])

    fig.tight_layout()
    plt.show()


def plot_hrf_spectrum():
    """
    Plot the magnitude spectrum of the HRF to illustrate its low-pass filter
    characteristic. High-frequency neural signals are strongly attenuated by
    the sluggish haemodynamic response, limiting the temporal bandwidth of
    BOLD fMRI.
    """
    # Use a fine, long time vector for good frequency resolution
    dt = 0.05          # 20 Hz sampling — well above any fMRI-relevant frequency
    t = np.arange(0, 64, dt)
    h = hrf(t)

    n = len(h)
    freqs = np.fft.rfftfreq(n, d=dt)          # positive frequencies only (Hz)
    H = np.abs(np.fft.rfft(h))
    H_db = 20 * np.log10(H / H.max())         # normalised magnitude in dB

    # Find the -3 dB cutoff
    cutoff_idx = np.where(H_db <= -3)[0]
    f_cutoff = freqs[cutoff_idx[0]] if len(cutoff_idx) else None

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(freqs, H_db, color="#1f77b4", linewidth=2)

    ax.axhline(-3, color="#d62728", linewidth=1.0, linestyle="--", alpha=0.8,
               label="-3 dB")
    if f_cutoff is not None:
        ax.axvline(f_cutoff, color="#d62728", linewidth=1.0, linestyle=":",
                   alpha=0.8, label=f"$f_{{-3\,\mathrm{{dB}}}}$ = {f_cutoff:.3f} Hz")

    ax.fill_between(freqs, H_db, -80, alpha=0.10, color="#1f77b4")

    ax.set_xlim(0, 0.5)
    ax.set_ylim(-80, 5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("HRF Frequency Response (Low-Pass Filter)")
    ax.legend(frameon=False)

    fig.tight_layout()
    plt.show()


plot_hrf_time()
plot_hrf_spectrum()

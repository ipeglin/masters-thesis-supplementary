import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from lib.fs.get_script_name import get_name
from lib.fs.project_config import REPO_ROOT
from plotting import plot_config

out_dir = plot_config.get_figs_output_dir()


def generate_signal_spectrum():
    fs = 1000  # Sampling frequency
    t = np.linspace(0, 2, 2 * fs)

    # Create a simple real-valued signal with a single frequency
    fc = 50  # Center frequency (Hz)
    s_t = np.cos(2 * np.pi * fc * t)

    # Hilbert transform
    s_hat = hilbert(s_t)
    s_t_hilbert = s_hat.imag

    # Analytical signal
    s_a = s_t + 1j * s_t_hilbert

    # Frequency axis for plotting (full spectrum)
    freqs = np.fft.fftfreq(len(t), 1 / fs)
    freqs = np.fft.fftshift(freqs)

    # Compute FFTs and shift for centered display
    S_f = np.fft.fftshift(np.fft.fft(s_t))
    S_hat_f = np.fft.fftshift(np.fft.fft(s_t_hilbert))
    S_a_f = np.fft.fftshift(np.fft.fft(s_a))

    # Create the spectrum plot with MATLAB-like styling
    plt.figure(
        # figsize=(12, 8)
    )

    # Original signal spectrum
    plt.subplot(2, 1, 1)
    plt.plot(freqs, np.abs(S_f))
    plt.title("Spectrum of Original Signal s(t)")
    plt.ylabel("Magnitude")
    plt.grid(False)
    plt.xlim(-100, 100)

    # Analytical signal spectrum
    plt.subplot(2, 1, 2)
    plt.plot(freqs, np.abs(S_a_f))
    plt.title("Spectrum of Analytic Signal $s_a(t) = s(t) + j\\hat{s}(t)$")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(False)
    plt.xlim(-100, 100)

    plt.savefig(out_dir / f"{get_name()}.pdf")
    # plt.show()


if __name__ == "__main__":
    generate_signal_spectrum()
    print("Signal spectrum figure generated successfully!")
